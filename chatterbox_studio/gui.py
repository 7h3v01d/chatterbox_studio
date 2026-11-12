import sys
import threading
import time

from PyQt6.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow,
                             QMessageBox, QProgressBar, QPushButton, QSlider, QTextEdit,
                             QVBoxLayout, QWidget)

from . import config
from .api import create_app, wav_bytes
from .engine import TTSEngine
from .recorder import Recorder, RecorderError
from .voices import VoiceError, VoiceStore

STYLE = """
QWidget { background:#16181d; color:#e6e6e6; font-size:13px; }
QGroupBox { border:1px solid #2c303a; border-radius:6px; margin-top:10px; padding-top:8px; }
QGroupBox::title { subcontrol-origin: margin; left:10px; color:#8ab4ff; }
QTextEdit, QLineEdit, QListWidget, QComboBox { background:#1e2128; border:1px solid #2c303a; border-radius:4px; padding:4px; }
QPushButton { background:#2a3040; border:1px solid #3a4152; border-radius:4px; padding:6px 12px; }
QPushButton:hover { background:#344058; } QPushButton:disabled { color:#666; }
QPushButton#rec { background:#7a2430; } QPushButton#rec:checked { background:#d93a4e; }
QProgressBar { border:1px solid #2c303a; border-radius:3px; text-align:center; }
QProgressBar::chunk { background:#4caf7d; }
"""


class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self, device=None):
        super().__init__()
        self.setWindowTitle("Chatterbox Studio")
        self.resize(980, 640)
        self.engine = TTSEngine(device)
        self.store = VoiceStore()
        self.rec = Recorder()
        self.recorded = None
        self.last_wav = None
        self.server = None
        self.worker = None
        self.stop_evt = threading.Event()

        self.player = QMediaPlayer()
        self.audio_out = QAudioOutput()
        self.player.setAudioOutput(self.audio_out)

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.addWidget(self._left_panel(), 1)
        outer.addWidget(self._right_panel(), 2)

        self.status = QLabel("Ready.")
        self.statusBar().addWidget(self.status, 1)
        self.timer = QTimer(self, interval=80, timeout=self._tick)
        self._refresh_voices()

    # ---------- layout ----------
    def _left_panel(self):
        box = QGroupBox("1 · Voice")
        v = QVBoxLayout(box)
        self.btn_rec = QPushButton("● Record")
        self.btn_rec.setObjectName("rec")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self._toggle_record)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.rec_time = QLabel("0.0 s  (aim for 10–20 s of clean speech)")
        self.btn_play_rec = QPushButton("Play recording")
        self.btn_play_rec.clicked.connect(self._play_recording)
        self.btn_play_rec.setEnabled(False)
        self.name = QLineEdit(placeholderText="Voice name")
        # QCheckBox can't word-wrap, so pair an empty checkbox with a wrapping, clickable label
        self.consent = QCheckBox()
        consent_lbl = QLabel("This is my voice (or I have explicit permission to clone it)")
        consent_lbl.setWordWrap(True)
        consent_lbl.mousePressEvent = lambda e: self.consent.toggle()
        consent_row = QWidget()
        cr = QHBoxLayout(consent_row)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.addWidget(self.consent, 0, Qt.AlignmentFlag.AlignTop)
        cr.addWidget(consent_lbl, 1)
        self.btn_save = QPushButton("Save as voice")
        self.btn_save.clicked.connect(self._save_voice)
        self.btn_import = QPushButton("Import audio file…")
        self.btn_import.clicked.connect(self._import_voice)
        self.voice_list = QListWidget()
        self.btn_del = QPushButton("Delete selected voice")
        self.btn_del.clicked.connect(self._delete_voice)
        for w in (self.btn_rec, self.level, self.rec_time, self.btn_play_rec, self.name,
                  consent_row, self.btn_save, self.btn_import, QLabel("Library:"),
                  self.voice_list, self.btn_del):
            v.addWidget(w)
        return box

    def _right_panel(self):
        box = QGroupBox("2 · Speak")
        v = QVBoxLayout(box)
        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._voice_changed)
        self.lang_combo = QComboBox()
        for code, name in config.LANGUAGES.items():
            self.lang_combo.addItem(f"{name} ({code})", code)
        self.text = QTextEdit(placeholderText="Type what the voice should say…")
        self.sliders = {}
        for key, label, lo, hi in (("exaggeration", "Exaggeration", 25, 200),
                                   ("cfg_weight", "Pace / CFG", 0, 100),
                                   ("temperature", "Temperature", 5, 200)):
            lbl, sl = QLabel(), QSlider(Qt.Orientation.Horizontal)
            sl.setRange(lo, hi)
            sl.valueChanged.connect(lambda x, l=lbl, t=label: l.setText(f"{t} {x/100:.2f}"))
            self.sliders[key] = sl
            self._slider_labels = getattr(self, "_slider_labels", [])
            self._slider_labels.append((lbl, sl))
        self.stream_cb = QCheckBox("Stream (start playing as it generates)")
        self.btn_gen = QPushButton("Generate && play")
        self.btn_gen.clicked.connect(self._generate)
        self.btn_stop = QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_saveas = QPushButton("Save WAV…")
        self.btn_saveas.clicked.connect(self._save_wav)
        self.btn_saveas.setEnabled(False)
        self.btn_defaults = QPushButton("Save as this voice's defaults")
        self.btn_defaults.clicked.connect(self._save_defaults)
        self.btn_server = QPushButton(f"Start API server (:{config.API_PORT})")
        self.btn_server.setCheckable(True)
        self.btn_server.clicked.connect(self._toggle_server)
        row = QHBoxLayout()
        row.addWidget(self.btn_gen)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_saveas)
        v.addWidget(QLabel("Voice:"))
        v.addWidget(self.voice_combo)
        v.addWidget(QLabel("Language:"))
        v.addWidget(self.lang_combo)
        v.addWidget(self.text)
        for lbl, sl in self._slider_labels:
            v.addWidget(lbl)
            v.addWidget(sl)
        v.addWidget(self.btn_defaults)
        v.addWidget(self.stream_cb)
        v.addLayout(row)
        v.addWidget(self.btn_server)
        v.addWidget(QLabel("MCP: run  python -m chatterbox_studio mcp  (needs the API server running)"))
        self._apply_settings(config.DEFAULT_SETTINGS)
        return box

    def _apply_settings(self, st):
        st = {**config.DEFAULT_SETTINGS, **st}
        for key, sl in self.sliders.items():
            sl.setValue(int(round(st[key] * 100)))
        i = self.lang_combo.findData(st["language"])
        self.lang_combo.setCurrentIndex(max(i, 0))

    def _current_settings(self):
        d = {k: sl.value() / 100 for k, sl in self.sliders.items()}
        d["language"] = self.lang_combo.currentData()
        return d

    def _voice_changed(self, _=None):
        vid = self.voice_combo.currentData()
        try:
            self._apply_settings(self.store.get(vid).settings if vid else {})
        except Exception as e:
            self._msg(f"Could not load voice settings: {e}", error=True)

    def _save_defaults(self):
        vid = self.voice_combo.currentData()
        if not vid:
            return self._msg("Select a cloned voice first (the built-in voice has no saved defaults).", error=True)
        try:
            self.store.update_settings(vid, self._current_settings())
        except Exception as e:
            return self._msg(str(e), error=True)
        self._msg("Saved these settings as the voice's defaults.")

    # ---------- helpers ----------
    def _msg(self, text, error=False):
        self.status.setText(text)
        self.status.setStyleSheet("color:#ff6b6b;" if error else "")

    def _refresh_voices(self):
        self.voice_list.clear()
        keep = self.voice_combo.currentData() if hasattr(self, "voice_combo") else None
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        self.voice_combo.addItem("Default voice", "")
        for v in self.store.list():
            self.voice_list.addItem(f"{v.name}  ({v.duration:.0f}s)  [{v.id}]")
            self.voice_combo.addItem(v.name, v.id)
        i = self.voice_combo.findData(keep) if keep else 0
        self.voice_combo.setCurrentIndex(max(i, 0))
        self.voice_combo.blockSignals(False)

    def _run(self, fn, on_done, busy_msg):
        if self.worker and self.worker.isRunning():
            return
        self._msg(busy_msg)
        self.worker = Worker(fn)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(lambda e: self._msg(e, error=True))
        self.worker.finished.connect(lambda: self.btn_stop.setEnabled(False))
        self.worker.start()

    # ---------- recording ----------
    def _toggle_record(self, on):
        try:
            if on:
                self.rec.start()
                self.timer.start()
                self.btn_rec.setText("■ Stop")
                self._msg("Recording…")
            else:
                self.recorded = (self.rec.stop(), self.rec.sample_rate)
                self.timer.stop()
                self.level.setValue(0)
                self.btn_rec.setText("● Record")
                self.btn_play_rec.setEnabled(True)
                self._msg(f"Captured {len(self.recorded[0]) / self.recorded[1]:.1f} s.")
        except RecorderError as e:
            self.btn_rec.setChecked(False)
            self.btn_rec.setText("● Record")
            self.timer.stop()
            self._msg(str(e), error=True)

    def _tick(self):
        self.level.setValue(int(min(1.0, self.rec.level) * 100))
        self.rec_time.setText(f"{self.rec.seconds:.1f} s")

    def _play_recording(self):
        if not self.recorded:
            return
        import sounddevice as sd
        sd.play(self.recorded[0], self.recorded[1])

    def _save_voice(self):
        if not self.recorded:
            return self._msg("Record something first.", error=True)
        try:
            v = self.store.add_audio(self.name.text(), self.recorded[0], self.recorded[1],
                                     self.consent.isChecked())
        except VoiceError as e:
            return self._msg(str(e), error=True)
        self._refresh_voices()
        self._msg(f"Saved voice '{v.name}'.")

    def _import_voice(self):
        path, _ = QFileDialog.getOpenFileName(self, "Voice sample", "", "Audio (*.wav *.flac *.ogg *.mp3)")
        if not path:
            return
        try:
            v = self.store.add_file(self.name.text(), path, self.consent.isChecked())
        except VoiceError as e:
            return self._msg(str(e), error=True)
        self._refresh_voices()
        self._msg(f"Imported voice '{v.name}'.")

    def _delete_voice(self):
        row = self.voice_list.currentRow()
        voices = self.store.list()
        if row < 0 or row >= len(voices):
            return
        if QMessageBox.question(self, "Delete", f"Delete '{voices[row].name}'?") == QMessageBox.StandardButton.Yes:
            self.store.delete(voices[row].id)
            self._refresh_voices()

    # ---------- synthesis ----------
    def _generate(self):
        text = self.text.toPlainText()
        vid = self.voice_combo.currentData()
        path = self.store.get(vid).path if vid else None
        st = self._current_settings()
        stream = self.stream_cb.isChecked()
        self.stop_evt.clear()

        def job():
            import torch
            out = config.OUT_DIR / f"tts_{int(time.time())}.wav"
            parts, sr, sink = [], None, None
            gen = self.engine.synthesize_stream(
                text, path, st["exaggeration"], st["cfg_weight"],
                st["temperature"], st["language"])
            try:
                for wav, sr in gen:
                    parts.append(wav)
                    if stream:
                        import sounddevice as sd
                        if sink is None:
                            sink = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                            sink.start()
                        data = wav.squeeze().cpu().numpy().astype("float32").reshape(-1, 1)
                        step = max(1, int(sr * 0.1))  # 100 ms slices keep Stop responsive
                        for i in range(0, len(data), step):
                            if self.stop_evt.is_set():
                                break
                            sink.write(data[i:i + step])
                    if self.stop_evt.is_set():
                        break
            finally:
                gen.close()  # releases the engine lock immediately
                if sink is not None:
                    sink.abort() if self.stop_evt.is_set() else sink.stop()
                    sink.close()
            stopped = self.stop_evt.is_set()
            if not parts:
                if stopped:
                    return None, stream, True
                raise RuntimeError("No audio was produced.")
            out.write_bytes(wav_bytes(torch.cat(parts, dim=-1), sr))
            return str(out), stream, stopped

        busy = ("Loading model (first run downloads weights)…" if not self.engine.loaded
                else "Generating…")
        self._run(job, self._generated, busy)
        if self.worker and self.worker.isRunning():
            self.btn_stop.setEnabled(True)

    def _stop(self):
        self.stop_evt.set()
        self.player.stop()
        self.btn_stop.setEnabled(False)
        self._msg("Stopping… (takes effect once the current chunk finishes generating)")

    def _generated(self, result):
        path, streamed, stopped = result
        if path is None:
            return self._msg("Stopped before any audio was produced.")
        self.last_wav = path
        self.btn_saveas.setEnabled(True)
        if stopped:
            return self._msg(f"Stopped. Partial audio saved → {path}")
        if not streamed:
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
        self._msg(f"Done → {path}")

    def _save_wav(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save WAV", "speech.wav", "WAV (*.wav)")
        if dest and self.last_wav:
            import shutil
            shutil.copyfile(self.last_wav, dest)

    # ---------- API server (shares this process's model) ----------
    def _toggle_server(self, on):
        import uvicorn
        if on:
            app = create_app(self.engine, self.store)
            self.server = uvicorn.Server(uvicorn.Config(app, host=config.API_HOST,
                                                        port=config.API_PORT, log_level="warning"))
            threading.Thread(target=self.server.run, daemon=True).start()
            self.btn_server.setText(f"API running on http://{config.API_HOST}:{config.API_PORT}  (click to stop)")
            self._msg("API server started.")
        elif self.server:
            self.server.should_exit = True
            self.server = None
            self.btn_server.setText(f"Start API server (:{config.API_PORT})")
            self._msg("API server stopped.")

    def closeEvent(self, e):
        if self.server:
            self.server.should_exit = True
        super().closeEvent(e)


def run_gui(device=None):
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    w = MainWindow(device)
    w.show()
    sys.exit(app.exec())
