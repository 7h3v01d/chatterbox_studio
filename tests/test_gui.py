# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Offscreen GUI smoke tests: build the real window, drive it with the fake model."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")

from conftest import SR, speech_like  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pump(app, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def win(app, factory, tmp_path):
    from chatterbox_studio import gui
    from chatterbox_studio.engine import TTSEngine
    from chatterbox_studio.jobs import JobManager
    from chatterbox_studio.voices import VoiceStore
    app.setStyleSheet(gui.STYLE)
    w = gui.MainWindow("cpu")
    w.engine = TTSEngine("cpu", model_factory=factory)
    w.jobs = JobManager(w.engine, tmp_path / "jobs")
    w.store = VoiceStore(tmp_path / "voices")
    w._refresh_voices()
    yield w
    w.close()


def test_speak_without_streaming(app, win):
    win.stream_cb.setChecked(False)
    win.text.setPlainText("Hello there. This is a test.")
    win._speak()
    assert _pump(app, lambda: not win.speaking)
    assert win.last is not None and win.last[0].seconds > 0
    assert win.btn_save.isEnabled()
    assert "Done" in win.status.text()


def test_empty_text_is_friendly(app, win):
    win.text.setPlainText("")
    win._speak()
    assert "Type something" in win.status.text() and not win.speaking


def test_presets_drive_sliders(app, win):
    win.preset_combo.setCurrentIndex(win.preset_combo.findData("dramatic"))
    assert win.current_settings()["exaggeration"] == 1.2
    win.sliders["exaggeration"][0].setValue(70)
    assert win.preset_combo.currentData() == "custom"


def test_record_dialog_saves_voice(app, win):
    from chatterbox_studio.gui import RecordDialog
    dlg = RecordDialog(win.store, win)
    dlg._set_samples(speech_like(12), SR)
    assert not dlg.btn_save.isEnabled()          # needs name + consent
    dlg.name.setText("Me")
    dlg.consent.setChecked(True)
    assert dlg.btn_save.isEnabled()
    dlg._save()
    assert dlg.voice and win.store.get(dlg.voice.id).name == "Me"
    win._refresh_voices(select=dlg.voice.id)
    assert win.voice_combo.currentData() == dlg.voice.id


def test_voice_defaults_roundtrip(app, win):
    v = win.store.add_audio("Def", speech_like(), SR, True)
    win._refresh_voices(select=v.id)
    win.preset_combo.setCurrentIndex(win.preset_combo.findData("calm"))
    win._save_defaults()
    assert win.store.get(v.id).settings["exaggeration"] == 0.3
    win.voice_combo.setCurrentIndex(0)
    win.voice_combo.setCurrentIndex(win.voice_combo.findData(v.id))
    assert win.preset_combo.currentData() == "calm"


def test_long_form_job(app, win):
    win.long_text.setPlainText("# Chapter\n\nFirst paragraph here.\n\nSecond one.")
    win.long_fmt.setCurrentIndex(win.long_fmt.findData("wav"))
    win._long_start()
    assert _pump(app, lambda: (win._poll(), win.job_id is None)[1])
    assert win._long_last and win._long_last.endswith(".wav")
    win._refresh_history()
    assert win.hist.rowCount() >= 1


def test_server_refuses_lan_without_token(app, win):
    win.srv_scope.setCurrentIndex(1)
    win.srv_token.setText("")
    win.btn_server.setChecked(True)
    win._toggle_server(True)
    assert not win.btn_server.isChecked() and "token" in win.status.text()


def test_mcp_snippet(app, win):
    import json
    win._snip_mcp()
    cfg = json.loads(QtWidgets.QApplication.clipboard().text())
    assert cfg["mcpServers"]["chatterbox"]["args"] == ["-m", "chatterbox_studio", "mcp"]
