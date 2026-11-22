# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Desktop app.

Layout: voice library on the left; tabs for Speak, Long-form, History and Connect.
Threading: QObject workers moved onto QThreads, held in _worker_refs so they can't be
garbage-collected mid-run. Results come back through a main-thread Bridge object, so
every UI update happens on the GUI thread.
"""
import json
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QGuiApplication, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QFormLayout, QFrame, QGridLayout,
                             QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QSlider, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                             QVBoxLayout, QWidget)

from . import __version__, audio, config, text as textmod
from .engine import Cancelled, Result, TTSEngine
from .jobs import JobManager, history, save_result
from .recorder import Recorder, RecorderError, list_inputs
from .voices import VoiceError, VoiceStore, effective_settings

# ---- design system -----------------------------------------------------------------
OBSIDIAN, PANEL, BORDER = "#0b0f14", "#11161d", "#1e2831"
TEAL, PHOSPHOR, AMBER, RED, TEXT, DIM = "#2fd6c3", "#4be08a", "#ffb454", "#ff6b6b", "#c8d3da", "#6b7a86"
VERDICT_COLOUR = {"great": PHOSPHOR, "ok": AMBER, "poor": RED}

STYLE = f"""
* {{ font-family: 'JetBrains Mono', 'Cascadia Mono', Consolas, monospace; font-size: 12px; }}
QWidget {{ background: {OBSIDIAN}; color: {TEXT}; }}
QFrame#panel, QTabWidget::pane {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QLabel {{ background: transparent; }}
QLabel#h1 {{ color: {TEAL}; font-size: 13px; font-weight: bold; letter-spacing: 1px; }}
QLabel#dim {{ color: {DIM}; }}
QLabel#script {{ font-size: 15px; color: {TEXT}; background: {OBSIDIAN};
                 border: 1px solid {BORDER}; padding: 14px; }}
QPlainTextEdit, QLineEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {{
    background: {OBSIDIAN}; border: 1px solid {BORDER}; border-radius: 0; padding: 4px;
    selection-background-color: {TEAL}; selection-color: {OBSIDIAN}; }}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus {{ border-color: {TEAL}; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QListWidget::item {{ padding: 6px 4px; }}
QListWidget::item:selected {{ background: {BORDER}; color: {TEAL}; }}
QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 0;
               padding: 7px 14px; }}
QPushButton:hover {{ border-color: {TEAL}; color: {TEAL}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {TEAL}; color: {OBSIDIAN}; border-color: {TEAL}; font-weight: bold; }}
QPushButton#primary:hover {{ background: {PHOSPHOR}; border-color: {PHOSPHOR}; color: {OBSIDIAN}; }}
QPushButton#primary:disabled {{ background: {BORDER}; color: {DIM}; }}
QPushButton#rec {{ border-color: {RED}; color: {RED}; font-weight: bold; }}
QPushButton#rec:checked {{ background: {RED}; color: {OBSIDIAN}; }}
QPushButton#flat {{ border: none; color: {DIM}; padding: 4px; }}
QPushButton#flat:hover {{ color: {TEAL}; }}
QTabBar::tab {{ background: {OBSIDIAN}; border: 1px solid {BORDER}; border-bottom: none;
                padding: 8px 18px; color: {DIM}; border-radius: 0; }}
QTabBar::tab:selected {{ background: {PANEL}; color: {TEAL}; }}
QProgressBar {{ background: {OBSIDIAN}; border: 1px solid {BORDER}; border-radius: 0;
                text-align: center; color: {TEXT}; height: 14px; }}
QProgressBar::chunk {{ background: {TEAL}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; background: {TEAL}; border-radius: 0; }}
QSlider::sub-page:horizontal {{ background: {TEAL}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER}; background: {OBSIDIAN}; }}
QCheckBox::indicator:checked {{ background: {TEAL}; border-color: {TEAL}; }}
QHeaderView::section {{ background: {PANEL}; color: {DIM}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 5px; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {DIM}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {BORDER}; color: {TEAL}; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {TEAL}; }}
"""

SLIDERS = (  # key, label, tooltip, lo, hi, scale
    ("exaggeration", "Emotion", "How animated the delivery is. 0.5 is neutral; higher is "
     "more expressive (and a bit faster).", 25, 200, 100),
    ("cfg_weight", "Pacing", "How closely it follows the reference voice's pacing. Lower is "
     "slower and more deliberate; 0 avoids accent bleed across languages.", 0, 100, 100),
    ("temperature", "Variation", "Randomness. Lower is steadier and more repeatable.", 5, 150, 100),
    ("speed", "Speed", "Playback tempo without changing pitch.", 50, 200, 100),
    ("sentence_pause", "Pause", "Silence between sentences, in seconds.", 0, 150, 100),
)
AUDIO_FILTER = "Audio (*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.webm)"


# ---- threading ---------------------------------------------------------------------
class Worker(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @pyqtSlot()
    def run(self):
        try:
            self.done.emit(self.fn(self.progress.emit))
        except Exception as e:
            self.failed.emit(str(e) or type(e).__name__)
        finally:
            self.finished.emit()


class Bridge(QObject):
    """Lives on the GUI thread; turns worker signals into plain callbacks there."""

    def __init__(self, on_done, on_fail, on_progress, on_finish):
        super().__init__()
        self._d, self._f, self._p, self._x = on_done, on_fail, on_progress, on_finish

    @pyqtSlot(object)
    def done(self, r):
        self._d and self._d(r)

    @pyqtSlot(str)
    def failed(self, msg):
        self._f and self._f(msg)

    @pyqtSlot(int, int)
    def progress(self, a, b):
        self._p and self._p(a, b)

    @pyqtSlot()
    def finished(self):
        self._x and self._x()


def h1(text):
    lbl = QLabel(text.upper())
    lbl.setObjectName("h1")
    return lbl


def dim(text):
    lbl = QLabel(text)
    lbl.setObjectName("dim")
    lbl.setWordWrap(True)
    return lbl


def panel():
    f = QFrame()
    f.setObjectName("panel")
    return f


def button(text, slot=None, kind=None, tip=None):
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if slot:
        b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


class Threads:
    """Mixin: run a callable off the GUI thread."""

    def _init_threads(self):
        self._worker_refs = []

    def run_bg(self, fn, on_done=None, on_fail=None, on_progress=None, on_finish=None):
        th = QThread()
        w = Worker(fn)
        br = Bridge(on_done, on_fail, on_progress, on_finish)
        w.moveToThread(th)
        th.started.connect(w.run)
        w.done.connect(br.done)
        w.failed.connect(br.failed)
        w.progress.connect(br.progress)
        w.finished.connect(br.finished)
        w.finished.connect(th.quit)
        ref = (th, w, br)
        self._worker_refs.append(ref)
        th.finished.connect(lambda r=ref: self._worker_refs.remove(r) if r in self._worker_refs else None)
        th.start()
        return ref

    def wait_threads(self, ms=3000):
        for th, _, _ in list(self._worker_refs):
            th.quit()
            th.wait(ms)


# ---- new-voice dialog --------------------------------------------------------------
class RecordDialog(QDialog, Threads):
    TARGET, LIMIT = 12.0, 30.0

    def __init__(self, store: VoiceStore, parent=None):
        super().__init__(parent)
        self._init_threads()
        self.store = store
        self.rec = Recorder()
        self.samples = None
        self.voice = None
        self.setWindowTitle("New voice")
        self.setMinimumSize(660, 640)
        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.addWidget(h1("1 · Record about 10–15 seconds"))
        v.addWidget(dim("Quiet room, normal speaking voice, about a hand's width from the mic. "
                        "Read this aloud (or say anything you like):"))
        script = QLabel(config.READING_SCRIPT)
        script.setObjectName("script")
        script.setWordWrap(True)
        script.setMinimumHeight(120)
        v.addWidget(script)

        row = QHBoxLayout()
        self.mic = QComboBox()
        self.mic.addItem("Default microphone", None)
        for idx, name in list_inputs():
            self.mic.addItem(name, idx)
        row.addWidget(QLabel("Mic"))
        row.addWidget(self.mic, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        self.btn_rec = button("● Record", self._toggle, "rec")
        self.btn_rec.setCheckable(True)
        self.btn_play = button("▶ Listen back", self._play)
        self.btn_play.setEnabled(False)
        self.btn_file = button("…or use an audio file", self._from_file)
        row.addWidget(self.btn_rec)
        row.addWidget(self.btn_play)
        row.addStretch(1)
        row.addWidget(self.btn_file)
        v.addLayout(row)

        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.length = QProgressBar()
        self.length.setRange(0, int(self.TARGET * 10))
        self.length.setFormat("0.0 s")
        v.addWidget(dim("Input level (keep it out of the red)"))
        v.addWidget(self.level)
        v.addWidget(dim("Length (aim to fill the bar)"))
        v.addWidget(self.length)

        self.report = QLabel("")
        self.report.setWordWrap(True)
        v.addWidget(self.report)

        v.addWidget(h1("2 · Name it"))
        self.name = QLineEdit(placeholderText="e.g. My voice")
        v.addWidget(self.name)
        self.consent = QCheckBox("This is my own voice, or I have the speaker's explicit "
                                 "permission to clone it.")
        v.addWidget(self.consent)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Cancel", self.reject))
        self.btn_save = button("Save voice", self._save, "primary")
        self.btn_save.setEnabled(False)
        row.addWidget(self.btn_save)
        v.addLayout(row)
        for w in (self.name,):
            w.textChanged.connect(self._validate)
        self.consent.toggled.connect(self._validate)

        self.timer = QTimer(self, interval=60, timeout=self._tick)
        self.player = None

    def _validate(self):
        self.btn_save.setEnabled(self.samples is not None and bool(self.name.text().strip())
                                 and self.consent.isChecked())

    def _toggle(self, on):
        try:
            if on:
                self.rec.start(self.mic.currentData())
                self.samples = None
                self.report.setText("")
                self.btn_rec.setText("■ Stop")
                self.timer.start()
            else:
                self._stop()
        except RecorderError as e:
            self.btn_rec.setChecked(False)
            self.btn_rec.setText("● Record")
            self.timer.stop()
            self._show_error(str(e))

    def _stop(self):
        self.timer.stop()
        self.btn_rec.setChecked(False)
        self.btn_rec.setText("● Record again")
        self.level.setValue(0)
        try:
            raw = self.rec.stop()
        except RecorderError as e:
            return self._show_error(str(e))
        self._set_samples(raw, self.rec.sample_rate)

    def _tick(self):
        lvl = min(1.0, self.rec.level)
        self.level.setValue(int(lvl * 100))
        colour = RED if lvl > 0.95 else TEAL
        self.level.setStyleSheet(f"QProgressBar::chunk {{ background: {colour}; }}")
        sec = self.rec.seconds
        self.length.setValue(min(self.length.maximum(), int(sec * 10)))
        self.length.setFormat(f"{sec:.1f} s" + ("  ✓ enough" if sec >= 10 else ""))
        if sec >= self.LIMIT:
            self._stop()

    def _set_samples(self, raw, sr):
        a = audio.trim_silence(raw, sr, threshold_db=-40.0, pad=0.15)
        self.samples = (raw, sr)
        sec = len(a) / float(sr)
        self.length.setValue(min(self.length.maximum(), int(sec * 10)))
        self.length.setFormat(f"{sec:.1f} s of speech")
        r = audio.analyze_reference(a, sr)
        col = VERDICT_COLOUR[r.verdict]
        tips = "<br>".join(f"• {t}" for t in r.tips)
        self.report.setText(f"<b style='color:{col}'>{r.summary()}</b><br>{tips}")
        self.btn_play.setEnabled(True)
        self._validate()

    def _from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Voice sample", "", AUDIO_FILTER)
        if not path:
            return
        try:
            a, sr = audio.read_any(path)
        except audio.AudioError as e:
            return self._show_error(str(e))
        if not self.name.text().strip():
            self.name.setText(Path(path).stem.replace("_", " ")[:40])
        self._set_samples(a, sr)

    def _play(self):
        if self.samples is None:
            return
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(self.samples[0], self.samples[1])
        except Exception as e:
            self._show_error(f"Playback failed: {e}")

    def _show_error(self, msg):
        self.report.setText(f"<b style='color:{RED}'>{msg}</b>")

    def _save(self):
        try:
            self.voice = self.store.add_audio(self.name.text(), self.samples[0], self.samples[1],
                                              self.consent.isChecked())
        except VoiceError as e:
            return self._show_error(str(e))
        self.accept()

    def done(self, r):
        if self.rec.recording:
            try:
                self.rec.stop()
            except RecorderError:
                pass
        self.timer.stop()
        super().done(r)


# ---- main window -------------------------------------------------------------------
class MainWindow(QMainWindow, Threads):
    def __init__(self, device=None):
        super().__init__()
        self._init_threads()
        self.setWindowTitle(f"Chatterbox Studio {__version__}")
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self.engine = TTSEngine(device)
        self.store = VoiceStore()
        self.jobs = JobManager(self.engine)
        self.server = None
        self.last = None              # (Result, settings, text, voice_id)
        self.stop_evt = threading.Event()
        self.speaking = False
        self.job_id = None

        split = QSplitter()
        split.addWidget(self._library())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._speak_tab(), "Speak")
        self.tabs.addTab(self._long_tab(), "Long-form")
        self.tabs.addTab(self._history_tab(), "History")
        self.tabs.addTab(self._connect_tab(), "Connect")
        self.tabs.currentChanged.connect(lambda i: i == 2 and self._refresh_history())
        split.addWidget(self.tabs)
        split.setSizes([300, 880])
        split.setChildrenCollapsible(False)
        self.setCentralWidget(split)

        self.status = QLabel("Ready.")
        self.model_lbl = QLabel()
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.model_lbl)
        self.btn_preload = button("Load model now", self._preload, "flat",
                                  "Load the model in the background so the first Speak is quick.")
        self.statusBar().addPermanentWidget(self.btn_preload)
        self.poll = QTimer(self, interval=500, timeout=self._poll)
        self.poll.start()
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._speak)
        QShortcut(QKeySequence("Escape"), self, activated=self._stop)
        self._refresh_voices()
        self._poll()

    # ---- voice library ---------------------------------------------------------
    def _library(self):
        p = panel()
        v = QVBoxLayout(p)
        v.addWidget(h1("Voices"))
        self.voice_list = QListWidget()
        self.voice_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.voice_list.customContextMenuRequested.connect(self._voice_menu)
        self.voice_list.itemDoubleClicked.connect(lambda it: self._select_voice(it.data(Qt.ItemDataRole.UserRole)))
        v.addWidget(self.voice_list, 1)
        self.empty_hint = dim("No voices yet. Click “New voice” and read a short passage — "
                              "about 15 seconds. You can also drop an audio file here.")
        v.addWidget(self.empty_hint)
        v.addWidget(button("＋ New voice", self._new_voice, "primary"))
        row = QHBoxLayout()
        row.addWidget(button("Preview", self._preview_voice, tip="Hear the selected voice"))
        row.addWidget(button("Delete", self._delete_voice))
        v.addLayout(row)
        p.setMinimumWidth(240)
        return p

    def _refresh_voices(self, select=None):
        keep = select or self.voice_combo.currentData()
        self.voice_list.clear()
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        self.voice_combo.addItem("Built-in voice (English)", "")
        voices = self.store.list()
        for v in voices:
            verdict = (v.quality or {}).get("verdict", "")
            it = QListWidgetItem(f"{v.name}\n{v.duration:.0f}s · {verdict or 'unchecked'}")
            it.setData(Qt.ItemDataRole.UserRole, v.id)
            if verdict:
                it.setForeground(QColor(VERDICT_COLOUR.get(verdict, TEXT)))
            tips = "\n".join((v.quality or {}).get("tips", []))
            it.setToolTip(f"{v.name}  ({v.id})\n{tips}")
            self.voice_list.addItem(it)
            self.voice_combo.addItem(v.name, v.id)
        self.long_voice.clear()
        for i in range(self.voice_combo.count()):
            self.long_voice.addItem(self.voice_combo.itemText(i), self.voice_combo.itemData(i))
        self.empty_hint.setVisible(not voices)
        i = self.voice_combo.findData(keep) if keep else (1 if voices else 0)
        self.voice_combo.setCurrentIndex(max(i, 0))
        self.long_voice.setCurrentIndex(max(i, 0))
        self.voice_combo.blockSignals(False)
        self._voice_changed()

    def _selected_voice_id(self):
        it = self.voice_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else self.voice_combo.currentData()

    def _select_voice(self, vid):
        i = self.voice_combo.findData(vid)
        if i >= 0:
            self.voice_combo.setCurrentIndex(i)
            self.tabs.setCurrentIndex(0)

    def _new_voice(self):
        dlg = RecordDialog(self.store, self)
        if dlg.exec() and dlg.voice:
            self._refresh_voices(select=dlg.voice.id)
            self._msg(f"Saved voice “{dlg.voice.name}”. It's selected — type something and press Speak.")

    def _voice_menu(self, pos):
        it = self.voice_list.itemAt(pos)
        if not it:
            return
        vid = it.data(Qt.ItemDataRole.UserRole)
        m = QMenu(self)
        m.addAction("Use this voice", lambda: self._select_voice(vid))
        m.addAction("Preview", self._preview_voice)
        m.addAction("Play original recording", lambda: self._play_file(self.store.get(vid).path))
        m.addAction("Rename…", lambda: self._rename_voice(vid))
        m.addAction("Open folder", lambda: self._open(Path(self.store.get(vid).path).parent))
        m.addSeparator()
        m.addAction("Delete…", self._delete_voice)
        m.exec(self.voice_list.mapToGlobal(pos))

    def _rename_voice(self, vid):
        v = self.store.get(vid)
        name, ok = QInputDialog.getText(self, "Rename voice", "Name", text=v.name)
        if ok:
            try:
                self.store.rename(vid, name)
                self._refresh_voices(select=vid)
            except VoiceError as e:
                self._msg(str(e), error=True)

    def _delete_voice(self):
        vid = self._selected_voice_id()
        if not vid:
            return
        v = self.store.get(vid)
        if QMessageBox.question(self, "Delete voice", f"Delete “{v.name}”? This can't be undone."
                                ) == QMessageBox.StandardButton.Yes:
            self.store.delete(vid)
            self._refresh_voices()
            self._msg(f"Deleted “{v.name}”.")

    def _preview_voice(self):
        vid = self._selected_voice_id()
        self._select_voice(vid or "")
        self._speak(text="Hi there! This is how I sound. Pretty good, right?")

    # drag & drop an audio file onto the window -> new voice
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            name, ok = QInputDialog.getText(self, "New voice from file", "Name",
                                            text=Path(path).stem.replace("_", " ")[:40])
            if not ok:
                continue
            if QMessageBox.question(self, "Consent", "Is this your own voice, or do you have the "
                                    "speaker's explicit permission to clone it?"
                                    ) != QMessageBox.StandardButton.Yes:
                continue
            try:
                v = self.store.add_file(name, path, True)
            except VoiceError as err:
                self._msg(str(err), error=True)
                continue
            self._refresh_voices(select=v.id)
            self._msg(f"Imported “{v.name}”: {v.quality.get('verdict', '')}. "
                      + " ".join(v.quality.get("tips", [])[:1]))

    # ---- speak tab -------------------------------------------------------------
    def _speak_tab(self):
        p = panel()
        v = QVBoxLayout(p)
        g = QGridLayout()
        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._voice_changed)
        self.lang_combo = QComboBox()
        for code, name in config.LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        self.model_combo = QComboBox()
        for k, label in config.MODELS.items():
            self.model_combo.addItem(label.split(" — ")[0], k)
            self.model_combo.setItemData(self.model_combo.count() - 1, label,
                                         Qt.ItemDataRole.ToolTipRole)
        self.preset_combo = QComboBox()
        for k, pr in config.PRESETS.items():
            self.preset_combo.addItem(pr["label"], k)
        self.preset_combo.addItem("Custom", "custom")
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        for col, (lbl, w) in enumerate((("Voice", self.voice_combo), ("Language", self.lang_combo),
                                        ("Model", self.model_combo), ("Style", self.preset_combo))):
            g.addWidget(dim(lbl), 0, col)
            g.addWidget(w, 1, col)
        v.addLayout(g)

        self.text = QPlainTextEdit()
        self.text.setPlaceholderText(
            "Type or paste what the voice should say…\n\n"
            "Tips:  [pause 1s] adds a pause  ·  a blank line starts a new paragraph  ·  "
            "with the Turbo model, [laugh] [chuckle] [cough] add reactions  ·  Ctrl+Enter speaks")
        self.text.textChanged.connect(self._text_stats)
        v.addWidget(self.text, 1)
        self.stats = dim("")
        v.addWidget(self.stats)

        self.adv_btn = button("▸ Fine-tune", self._toggle_adv, "flat")
        v.addWidget(self.adv_btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.adv = QWidget()
        ag = QGridLayout(self.adv)
        ag.setContentsMargins(0, 0, 0, 0)
        self.sliders, self.slider_vals = {}, {}
        for i, (key, label, tip, lo, hi, scale) in enumerate(SLIDERS):
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.setToolTip(tip)
            val = QLabel()
            val.setMinimumWidth(48)
            sl.valueChanged.connect(lambda x, l=val, s=scale, k=key: self._slider_moved(k, l, x / s))
            self.sliders[key] = (sl, scale)
            r, c = divmod(i, 2)
            lab = QLabel(label)
            lab.setToolTip(tip)
            ag.addWidget(lab, r, c * 3)
            ag.addWidget(sl, r, c * 3 + 1)
            ag.addWidget(val, r, c * 3 + 2)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setSpecialValueText("random")
        self.seed.setToolTip("Same seed + same settings = the same take. 'random' = a new take each time.")
        ag.addWidget(QLabel("Seed"), 2, 3)
        ag.addWidget(self.seed, 2, 4)
        self.adv.setVisible(False)
        v.addWidget(self.adv)

        row = QHBoxLayout()
        self.btn_speak = button("▶  Speak", self._speak, "primary")
        self.btn_speak.setMinimumWidth(140)
        self.btn_stop = button("■ Stop", self._stop)
        self.btn_stop.setEnabled(False)
        self.btn_save = button("Save as…", self._save_as)
        self.btn_save.setEnabled(False)
        self.btn_replay = button("↻ Replay", lambda: self.last and self._play_result(self.last[0]))
        self.btn_replay.setEnabled(False)
        self.stream_cb = QCheckBox("Start playing straight away")
        self.stream_cb.setToolTip("Plays each sentence as soon as it's ready. On slower "
                                  "machines you may hear small gaps.")
        self.stream_cb.setChecked(True)
        row.addWidget(self.btn_speak)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_replay)
        row.addWidget(self.btn_save)
        row.addStretch(1)
        row.addWidget(self.stream_cb)
        row.addWidget(button("Save as voice default", self._save_defaults, "flat",
                             "Remember these settings for the selected voice."))
        v.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        self._apply_settings({})
        return p

    def _toggle_adv(self):
        show = not self.adv.isVisible()
        self.adv.setVisible(show)
        self.adv_btn.setText(("▾" if show else "▸") + " Fine-tune")

    def _slider_moved(self, key, label, value):
        label.setText(f"{value:.2f}" + ("×" if key == "speed" else "s" if key == "sentence_pause" else ""))
        if getattr(self, "_applying", False) or key not in ("exaggeration", "cfg_weight", "temperature"):
            return
        i = self.preset_combo.findData("custom")
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(i)
        self.preset_combo.blockSignals(False)

    def _preset_changed(self):
        k = self.preset_combo.currentData()
        if k in config.PRESETS:
            self._applying = True
            for key in ("exaggeration", "cfg_weight", "temperature"):
                sl, scale = self.sliders[key]
                sl.setValue(int(round(config.PRESETS[k][key] * scale)))
            self._applying = False

    def _apply_settings(self, st):
        st = {**config.DEFAULT_SETTINGS, **st}
        self._applying = True
        for key, (sl, scale) in self.sliders.items():
            sl.setValue(int(round(float(st[key]) * scale)))
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(st["language"])))
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(st["model"])))
        match = "custom"
        for k, pr in config.PRESETS.items():
            if all(abs(pr[x] - float(st[x])) < 1e-6 for x in ("exaggeration", "cfg_weight", "temperature")):
                match = k
                break
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(self.preset_combo.findData(match))
        self.preset_combo.blockSignals(False)
        self._applying = False

    def current_settings(self):
        d = {k: sl.value() / scale for k, (sl, scale) in self.sliders.items()}
        d["language"] = self.lang_combo.currentData()
        d["model"] = self.model_combo.currentData()
        return d

    def _voice_changed(self, _=None):
        vid = self.voice_combo.currentData()
        try:
            self._apply_settings(self.store.get(vid).settings if vid else {})
        except Exception:
            self._apply_settings({})
        for i in range(self.voice_list.count()):
            if self.voice_list.item(i).data(Qt.ItemDataRole.UserRole) == vid:
                self.voice_list.setCurrentRow(i)

    def _save_defaults(self):
        vid = self.voice_combo.currentData()
        if not vid:
            return self._msg("Pick one of your voices first — the built-in voice has no saved defaults.",
                             error=True)
        try:
            self.store.update_settings(vid, self.current_settings())
        except VoiceError as e:
            return self._msg(str(e), error=True)
        self._msg("Saved. These settings now load whenever you pick this voice (also via the API).")

    def _text_stats(self):
        t = self.text.toPlainText()
        if not t.strip():
            return self.stats.setText("")
        est = textmod.estimate_seconds(t)
        hint = "  ·  long text? the Long-form tab adds progress, subtitles and cancel" \
            if len(t) > 3000 else ""
        self.stats.setText(f"{len(t):,} characters  ·  about {est:.0f} s of speech{hint}")

    def _speak(self, text=None):
        if self.speaking:
            return
        text = text if isinstance(text, str) else self.text.toPlainText()
        if not text.strip():
            self.text.setFocus()
            return self._msg("Type something to say first.", error=True)
        vid = self.voice_combo.currentData()
        try:
            voice = self.store.get(vid) if vid else None
        except Exception:
            return self._msg("That voice no longer exists.", error=True)
        st = effective_settings(None, self.current_settings())
        seed = self.seed.value() or None
        stream = self.stream_cb.isChecked()
        self.stop_evt.clear()
        self._stop_player()

        def job(progress):
            parts, cues, t, sink, sr = [], [], 0.0, None, None
            it = self.engine.render(text, voice.path if voice else None, st, seed=seed,
                                    cancel=self.stop_evt, on_progress=progress,
                                    chunk_limit=200 if stream else 250)
            try:
                for piece in it:
                    sr = piece.sr
                    dur = len(piece.audio) / sr
                    if piece.kind == "speak":
                        cues.append((t, t + dur, piece.text))
                    t += dur
                    parts.append(piece.audio)
                    if stream:
                        if sink is None:
                            import sounddevice as sd
                            sink = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                            sink.start()
                        data = piece.audio.reshape(-1, 1)
                        step = max(1, int(sr * 0.1))         # 100 ms slices keep Stop responsive
                        for i in range(0, len(data), step):
                            if self.stop_evt.is_set():
                                break
                            sink.write(data[i:i + step])
                    if self.stop_evt.is_set():
                        break
            except Cancelled:
                pass                                       # Stop pressed: keep what we have
            finally:
                it.close()
                if sink is not None:
                    (sink.abort if self.stop_evt.is_set() else sink.stop)()
                    sink.close()
            if not parts:
                return None
            res = Result(audio.normalize_peak(np.concatenate(parts)), sr, cues)
            paths = save_result(res, "wav", text=text, voice=vid or "", settings=st)
            return res, paths, stream, self.stop_evt.is_set()

        self.speaking = True
        self.btn_speak.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._msg("Loading the model (the first run downloads it — a few minutes)…"
                  if not self.engine.loaded else "Generating…")
        self.run_bg(job, lambda r: self._spoken(r, st, text, vid), self._speak_failed,
                    self._speak_progress, self._speak_finished)

    def _speak_progress(self, done, total):
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"{done}/{total}")
        if done:
            self._msg("Speaking…" if self.stream_cb.isChecked() else "Generating…")

    def _speak_failed(self, msg):
        self._msg(msg, error=True)

    def _speak_finished(self):
        self.speaking = False
        self.btn_speak.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setVisible(False)

    def _spoken(self, r, st, text, vid):
        if r is None:
            return self._msg("Stopped.")
        res, paths, streamed, stopped = r
        self.last = (res, st, text, vid)
        self.btn_save.setEnabled(True)
        self.btn_replay.setEnabled(True)
        if not streamed and not stopped:
            self._play_file(paths["audio"])
        self._msg(("Stopped — partial take kept. " if stopped else "Done. ")
                  + f"{res.seconds:.1f}s, saved to History.")

    def _stop(self):
        self.stop_evt.set()
        self._stop_player()

    def _save_as(self):
        if not self.last:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save audio", str(Path.home() / "speech.mp3"),
            "MP3 (*.mp3);;WAV (*.wav);;FLAC (*.flac);;OGG (*.ogg)")
        if not dest:
            return
        fmt = Path(dest).suffix.lstrip(".").lower() or "wav"
        try:
            Path(dest).write_bytes(audio.encode(self.last[0].audio, self.last[0].sr, fmt))
        except audio.AudioError as e:
            return self._msg(str(e), error=True)
        self._msg(f"Saved → {dest}")

    # ---- playback --------------------------------------------------------------
    def _play_file(self, path):
        try:
            a, sr = audio.read_any(str(path))
        except audio.AudioError as e:
            return self._msg(str(e), error=True)
        self._play_array(a, sr)

    def _play_result(self, res):
        self._play_array(res.audio, res.sr)

    def _play_array(self, a, sr):
        """One audio path for everything (sounddevice). Non-blocking; Stop/Esc halts it."""
        try:
            import sounddevice as sd
            sd.stop()
            sd.play(a, sr)
        except Exception as e:
            self._msg(f"Playback failed: {e}", error=True)

    def _stop_player(self):
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    # ---- long-form tab ---------------------------------------------------------
    def _long_tab(self):
        p = panel()
        v = QVBoxLayout(p)
        v.addWidget(h1("Long-form narration"))
        v.addWidget(dim("Articles, scripts, chapters. Runs in the background with progress, "
                        "and can write subtitles (.srt) that line up with the audio — handy "
                        "for videos. Markdown headings and formatting are cleaned up for you."))
        row = QHBoxLayout()
        row.addWidget(button("Open text file…", self._open_text))
        self.long_voice = QComboBox()
        row.addWidget(QLabel("Voice"))
        row.addWidget(self.long_voice, 1)
        self.long_fmt = QComboBox()
        for f in ("mp3", "wav", "flac", "ogg"):
            self.long_fmt.addItem(f.upper(), f)
        row.addWidget(QLabel("Format"))
        row.addWidget(self.long_fmt)
        self.long_srt = QCheckBox("Subtitles (.srt)")
        self.long_srt.setChecked(True)
        row.addWidget(self.long_srt)
        v.addLayout(row)
        self.long_text = QPlainTextEdit()
        self.long_text.setPlaceholderText("Paste or open your text. Uses the Speak tab's "
                                          "language, model and style settings.")
        self.long_text.textChanged.connect(lambda: self.long_stats.setText(
            f"{len(self.long_text.toPlainText()):,} characters · about "
            f"{textmod.estimate_seconds(self.long_text.toPlainText()) / 60:.1f} min of speech"))
        v.addWidget(self.long_text, 1)
        self.long_stats = dim("")
        v.addWidget(self.long_stats)
        row = QHBoxLayout()
        self.btn_long = button("▶  Narrate", self._long_start, "primary")
        self.btn_long_cancel = button("Cancel", self._long_cancel)
        self.btn_long_cancel.setEnabled(False)
        self.btn_long_open = button("Show file", lambda: self._long_last and self._open(Path(self._long_last).parent))
        self.btn_long_open.setEnabled(False)
        self._long_last = None
        row.addWidget(self.btn_long)
        row.addWidget(self.btn_long_cancel)
        row.addWidget(self.btn_long_open)
        row.addStretch(1)
        v.addLayout(row)
        self.long_bar = QProgressBar()
        self.long_bar.setVisible(False)
        v.addWidget(self.long_bar)
        return p

    def _open_text(self):
        path, _ = QFileDialog.getOpenFileName(self, "Text file", "", "Text (*.txt *.md *.markdown);;All (*)")
        if path:
            try:
                self.long_text.setPlainText(Path(path).read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                self.long_text.setPlainText(Path(path).read_text(encoding="cp1252", errors="replace"))

    def _long_start(self):
        text = self.long_text.toPlainText()
        if not text.strip():
            return self._msg("Add some text first.", error=True)
        if len(text) > config.MAX_JOB_CHARS:
            return self._msg(f"That's over {config.MAX_JOB_CHARS:,} characters; split it into parts.",
                             error=True)
        vid = self.long_voice.currentData()
        voice = self.store.get(vid) if vid else None
        st = effective_settings(voice, self.current_settings())
        job = self.jobs.submit(text, vid or "", voice.path if voice else None, st,
                               self.long_fmt.currentData(), self.long_srt.isChecked(),
                               self.seed.value() or None)
        self.job_id = job.id
        self._job_t0 = time.time()
        self.btn_long.setEnabled(False)
        self.btn_long_cancel.setEnabled(True)
        self.long_bar.setVisible(True)
        self.long_bar.setRange(0, 0)
        self._msg("Narrating in the background — you can keep using the app.")

    def _long_cancel(self):
        if self.job_id:
            self.jobs.cancel(self.job_id)

    def _poll_job(self):
        if not self.job_id:
            return
        j = self.jobs.get(self.job_id)
        if j.total:
            self.long_bar.setRange(0, j.total)
            self.long_bar.setValue(j.done)
            el = time.time() - (j.started or self._job_t0)
            eta = el / j.done * (j.total - j.done) if j.done else 0
            self.long_bar.setFormat(f"{j.done}/{j.total} chunks · {el / 60:.1f} min elapsed"
                                    + (f" · ~{eta / 60:.1f} min left" if j.done else ""))
        if j.status in ("done", "failed", "cancelled"):
            self.job_id = None
            self.btn_long.setEnabled(True)
            self.btn_long_cancel.setEnabled(False)
            self.long_bar.setVisible(False)
            if j.status == "done":
                self._long_last = j.paths["audio"]
                self.btn_long_open.setEnabled(True)
                self._msg(f"Narration ready: {j.seconds / 60:.1f} min → {j.paths['audio']}")
            elif j.status == "failed":
                self._msg(j.error, error=True)
            else:
                self._msg("Narration cancelled.")

    # ---- history tab -----------------------------------------------------------
    def _history_tab(self):
        p = panel()
        v = QVBoxLayout(p)
        v.addWidget(h1("History"))
        self.hist = QTableWidget(0, 4)
        self.hist.setHorizontalHeaderLabels(["When", "Text", "Voice", "Length"])
        self.hist.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.hist.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.hist.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hist.verticalHeader().setVisible(False)
        self.hist.cellDoubleClicked.connect(lambda r, c: self._hist_play())
        v.addWidget(self.hist, 1)
        row = QHBoxLayout()
        row.addWidget(button("▶ Play", self._hist_play))
        row.addWidget(button("Use this text again", self._hist_reuse))
        row.addWidget(button("Show file", lambda: self._hist_item() and self._open(Path(self._hist_item()["audio"]).parent)))
        row.addStretch(1)
        row.addWidget(button("Delete", self._hist_delete))
        v.addLayout(row)
        return p

    def _refresh_history(self):
        self._hist_items = history() + [
            {"created": j.finished, "text": j.text, "voice": j.voice_id, "seconds": j.seconds,
             **j.paths} for j in self.jobs.list() if j.status == "done"]
        self._hist_items.sort(key=lambda d: d.get("created") or 0, reverse=True)
        names = {v.id: v.name for v in self.store.list()}
        self.hist.setRowCount(len(self._hist_items))
        for r, d in enumerate(self._hist_items):
            when = time.strftime("%d %b %H:%M", time.localtime(d.get("created") or 0))
            cells = (when, " ".join(d.get("text", "").split())[:160],
                     names.get(d.get("voice", ""), "built-in" if not d.get("voice") else d["voice"]),
                     f"{d.get('seconds', 0):.1f}s")
            for c, val in enumerate(cells):
                self.hist.setItem(r, c, QTableWidgetItem(val))

    def _hist_item(self):
        r = self.hist.currentRow()
        items = getattr(self, "_hist_items", [])
        return items[r] if 0 <= r < len(items) else None

    def _hist_play(self):
        d = self._hist_item()
        if d:
            self._play_file(d["audio"])

    def _hist_reuse(self):
        d = self._hist_item()
        if d:
            self.text.setPlainText(d.get("text", ""))
            if d.get("settings"):
                self._apply_settings(d["settings"])
            self.tabs.setCurrentIndex(0)

    def _hist_delete(self):
        d = self._hist_item()
        if not d:
            return
        self._stop_player()
        for k in ("audio", "srt", "meta"):
            if d.get(k):
                Path(d[k]).unlink(missing_ok=True)
        self._refresh_history()

    # ---- connect tab -----------------------------------------------------------
    def _connect_tab(self):
        p = panel()
        v = QVBoxLayout(p)
        v.addWidget(h1("Use your voices from other apps"))
        v.addWidget(dim("Start the local server, then any app on this computer can talk with "
                        "your voices: Claude Desktop (MCP), scripts, home automation, or "
                        "anything that speaks the OpenAI speech API."))
        f = QFormLayout()
        self.srv_scope = QComboBox()
        self.srv_scope.addItem("This computer only (recommended)", "127.0.0.1")
        self.srv_scope.addItem("My home network (needs a token)", "0.0.0.0")
        self.srv_scope.currentIndexChanged.connect(self._scope_changed)
        self.srv_port = QSpinBox()
        self.srv_port.setRange(1024, 65535)
        self.srv_port.setValue(config.API_PORT)
        trow = QHBoxLayout()
        self.srv_token = QLineEdit(config.API_TOKEN)
        self.srv_token.setPlaceholderText("optional on this computer, required on a network")
        trow.addWidget(self.srv_token, 1)
        trow.addWidget(button("Generate", lambda: self.srv_token.setText(secrets.token_urlsafe(24))))
        f.addRow("Who can connect", self.srv_scope)
        f.addRow("Port", self.srv_port)
        f.addRow("Token", trow)
        v.addLayout(f)
        self.btn_server = button("Start server", self._toggle_server, "primary")
        self.btn_server.setCheckable(True)
        v.addWidget(self.btn_server, 0, Qt.AlignmentFlag.AlignLeft)
        self.srv_status = dim("Server is off.")
        v.addWidget(self.srv_status)

        v.addWidget(h1("Copy a ready-made setup"))
        grid = QGridLayout()
        snippets = (("Claude Desktop (MCP) config", self._snip_mcp,
                     "Paste into claude_desktop_config.json, then restart Claude Desktop."),
                    ("Python (OpenAI SDK)", self._snip_openai,
                     "Works with any OpenAI-compatible speech client."),
                    ("curl", self._snip_curl, "Quick test from a terminal."),
                    ("API docs", lambda: QDesktopServices.openUrl(QUrl(self._base_url() + "/docs")),
                     "Interactive docs (server must be running)."))
        for i, (label, fn, tip) in enumerate(snippets):
            grid.addWidget(button(label, fn, tip=tip), i, 0)
            grid.addWidget(dim(tip), i, 1)
        v.addLayout(grid)
        v.addStretch(1)
        return p

    def _scope_changed(self):
        if self.srv_scope.currentData() != "127.0.0.1" and not self.srv_token.text():
            self.srv_token.setText(secrets.token_urlsafe(24))

    def _base_url(self, for_clients=True):
        host = self.srv_scope.currentData()
        if host == "0.0.0.0" and for_clients:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("10.255.255.255", 1))
                    host = s.getsockname()[0]
            except OSError:
                host = "127.0.0.1"
        return f"http://{host}:{self.srv_port.value()}"

    def _copy(self, text, what):
        QGuiApplication.clipboard().setText(text)
        self._msg(f"Copied {what} to the clipboard.")

    def _snip_mcp(self):
        env = {"CBX_URL": self._base_url()}
        if self.srv_token.text():
            env["CBX_TOKEN"] = self.srv_token.text()
        cfg = {"mcpServers": {"chatterbox": {"command": sys.executable,
                                             "args": ["-m", "chatterbox_studio", "mcp"],
                                             "env": env}}}
        self._copy(json.dumps(cfg, indent=2), "the MCP config")

    def _snip_openai(self):
        vid = self.voice_combo.currentData() or "alloy"
        tok = self.srv_token.text() or "not-needed"
        self._copy(
            "from openai import OpenAI\n"
            f'client = OpenAI(base_url="{self._base_url()}/v1", api_key="{tok}")\n'
            "with client.audio.speech.with_streaming_response.create(\n"
            f'        model="chatterbox", voice="{vid}", input="Hello from my own voice!") as r:\n'
            '    r.stream_to_file("hello.mp3")\n', "a Python example")

    def _snip_curl(self):
        vid = self.voice_combo.currentData()
        auth = f' -H "Authorization: Bearer {self.srv_token.text()}"' if self.srv_token.text() else ""
        body = json.dumps({"text": "Hello from my own voice!", "voice_id": vid, "format": "mp3"})
        self._copy(f"curl -X POST {self._base_url()}/tts{auth} -H \"Content-Type: application/json\" "
                   f"-d '{body}' -o hello.mp3", "a curl command")

    def _toggle_server(self, on):
        import uvicorn
        from .api import check_bind, create_app
        if on:
            host, port, token = self.srv_scope.currentData(), self.srv_port.value(), self.srv_token.text().strip()
            try:
                check_bind(host, token, config.ALLOW_INSECURE)
            except RuntimeError as e:
                self.btn_server.setChecked(False)
                return self._msg(str(e), error=True)
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    self.btn_server.setChecked(False)
                    return self._msg(f"Port {port} is already in use — is another copy running?",
                                     error=True)
            config.API_TOKEN = token
            app = create_app(self.engine, self.store, self.jobs)
            self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
            threading.Thread(target=self.server.run, daemon=True, name="cbx-api").start()
            self.btn_server.setText("Stop server")
            for w in (self.srv_scope, self.srv_port, self.srv_token):
                w.setEnabled(False)
            self.srv_status.setText(f"Running at {self._base_url()}  ·  docs at /docs"
                                    + ("  ·  token required" if token else ""))
            self._msg("Server started.")
        elif self.server:
            self.server.should_exit = True
            self.server = None
            self.btn_server.setText("Start server")
            for w in (self.srv_scope, self.srv_port, self.srv_token):
                w.setEnabled(True)
            self.srv_status.setText("Server is off.")
            self._msg("Server stopped.")

    # ---- misc ------------------------------------------------------------------
    def _preload(self):
        if self.engine.loaded or self.engine.busy:
            return
        st = self.current_settings()
        self.btn_preload.setEnabled(False)
        self._msg("Loading the model in the background (first run downloads it)…")
        self.run_bg(lambda _p: self.engine.load(st["model"], st["language"]),
                    lambda _r: self._msg("Model loaded — ready."),
                    lambda m: self._msg(m, error=True),
                    on_finish=lambda: self.btn_preload.setEnabled(True))

    def _poll(self):
        s = self.engine.status()
        state = f"{s['model']} on {s['device']}" if s["model_loaded"] else f"not loaded · {s['device']}"
        self.model_lbl.setText(f"model: {state}" + ("  · busy" if s["busy"] else ""))
        self.btn_preload.setVisible(not s["model_loaded"])
        self._poll_job()

    def _open(self, path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _msg(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet(f"color:{RED};" if error else f"color:{TEXT};")

    def closeEvent(self, e):
        self.stop_evt.set()
        if self.job_id:
            self.jobs.cancel(self.job_id)
        if self.server:
            self.server.should_exit = True
        self._stop_player()
        self.wait_threads()
        super().closeEvent(e)


def run_gui(device=None):
    app = QApplication(sys.argv)
    app.setApplicationName("Chatterbox Studio")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("JetBrains Mono", 10))
    w = MainWindow(device)
    w.show()
    sys.exit(app.exec())
