# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Regression tests for the 0.5.0 adversarial review (CBX-01..06) and the issues found
while verifying it against chatterbox-tts 0.1.7. Each one fails on the 0.5.0 code."""
import io
import os
import sys
import threading
import types
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from chatterbox_studio import config
from chatterbox_studio.api import create_app
from chatterbox_studio.engine import TTSEngine

from conftest import SR, FakeModel, TurboFake, speech_like


def _wav():
    buf = io.BytesIO()
    sf.write(buf, speech_like(), SR, format="WAV")
    return buf.getvalue()


def _blocking_engine():
    gate = threading.Event()

    class Blocking(FakeModel):
        def generate(self, text, **kw):
            gate.wait(10)
            return super().generate(text, **kw)
    return TTSEngine("cpu", model_factory=lambda k, d: Blocking(k)), gate


# ---- CBX-01: MCP may only write new files inside approved folders -------------------
class TestMcpWrites:
    @pytest.fixture
    def mcp(self, monkeypatch):
        pytest.importorskip("mcp")
        from chatterbox_studio import mcp_server as m
        calls = []

        class R:
            content = b"RIFF fake audio"
            headers = {"X-Audio-Seconds": "1.0"}
        monkeypatch.setattr(m, "_call", lambda *a, **k: (calls.append(a), R())[1])
        monkeypatch.setattr(config, "MCP_OUTPUT_DIRS", "", raising=False)
        m.calls = calls
        return m

    def test_absolute_path_outside_is_refused(self, mcp, tmp_path):
        target = tmp_path / "elsewhere" / "victim.wav"
        with pytest.raises(RuntimeError, match="outside the allowed folders"):
            mcp.speak("hi", save_to=str(target))
        assert not target.exists() and mcp.calls == []      # refused before any work

    def test_existing_file_is_never_overwritten(self, mcp):
        victim = config.OUT_DIR / "keep_me.wav"
        victim.write_bytes(b"precious")
        with pytest.raises(RuntimeError, match="already exists"):
            mcp.speak("hi", save_to=str(victim))
        assert victim.read_bytes() == b"precious"

    def test_traversal_and_wrong_extension(self, mcp):
        with pytest.raises(RuntimeError, match="outside"):
            mcp.speak("hi", save_to="../../../escape.wav")
        with pytest.raises(RuntimeError, match=r"\.wav"):
            mcp.speak("hi", save_to="settings.json")

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
    def test_symlink_escape(self, mcp, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        link = config.OUT_DIR / f"link_{os.getpid()}"
        link.symlink_to(outside, target_is_directory=True)
        try:
            with pytest.raises(RuntimeError, match="outside"):
                mcp.speak("hi", save_to=str(link / "x.wav"))
            assert not (outside / "x.wav").exists()
        finally:
            link.unlink()

    def test_user_approved_folder_and_relative_name(self, mcp, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MCP_OUTPUT_DIRS", str(tmp_path), raising=False)
        out = tmp_path / "ok.wav"
        assert str(out.resolve()) in mcp.speak("hi", save_to=str(out))
        assert out.read_bytes().startswith(b"RIFF")
        rel = f"rel_{os.getpid()}.wav"
        mcp.speak("hi", save_to=rel)
        assert (config.OUT_DIR / rel).exists()


# ---- CBX-02: bounded admission ----------------------------------------------------------
class TestAdmission:
    def test_job_count_limit(self, tmp_path):
        from chatterbox_studio.jobs import JobManager, QueueFull
        eng, gate = _blocking_engine()
        jm = JobManager(eng, tmp_path, max_pending=5)
        try:
            for _ in range(5):
                jm.submit("Hello.", "", None, {})
            with pytest.raises(QueueFull):
                jm.submit("Hello.", "", None, {})
        finally:
            gate.set()

    def test_text_volume_limit(self, tmp_path):
        from chatterbox_studio.jobs import JobManager, QueueFull
        eng, gate = _blocking_engine()
        jm = JobManager(eng, tmp_path, max_pending=100, max_pending_chars=250_000)
        try:
            jm.submit("x" * 200_000, "", None, {})
            with pytest.raises(QueueFull, match="Too much text"):
                jm.submit("x" * 100_000, "", None, {})
        finally:
            gate.set()

    def test_api_returns_429(self, tmp_path, store):
        from chatterbox_studio.jobs import JobManager
        eng, gate = _blocking_engine()
        jm = JobManager(eng, tmp_path, max_pending=2)
        c = TestClient(create_app(eng, store, jm), base_url="http://127.0.0.1:8765")
        try:
            codes = [c.post("/jobs", json={"text": "Hi."}).status_code for _ in range(3)]
            assert codes[:2] == [200, 200] and codes[2] == 429
            r = c.post("/jobs", json={"text": "Hi."})
            assert r.headers.get("retry-after") == "30"
        finally:
            gate.set()

    def test_capacity_frees_up_and_text_is_released(self, engine, tmp_path):
        from chatterbox_studio.jobs import JobManager
        jm = JobManager(engine, tmp_path, max_pending=1)
        j = jm.wait(jm.submit("First.", "", None, {}).id)
        assert j.status == "done" and j.text == ""          # memory released
        assert jm.wait(jm.submit("Second.", "", None, {}).id).status == "done"


# ---- CBX-03: verified NOT a defect; pin the invariant it relies on ----------------------
def test_exaggeration_reaches_every_generate_call(engine, factory, voice):
    engine.synthesize("a.", voice.path, {"exaggeration": 0.3})
    engine.synthesize("b.", voice.path, {"exaggeration": 1.7})
    m = factory.made[0]
    assert len(m.prepared) == 1                               # conditioning cached...
    assert [c["exaggeration"] for c in m.calls] == [0.3, 1.7]  # ...emotion still updated


# ---- CBX-04: commit boundary and atomic outputs ----------------------------------------
class TestCommit:
    def test_cancel_after_render_before_commit_writes_nothing(self, engine, tmp_path, monkeypatch):
        from chatterbox_studio.jobs import JobManager
        jm = JobManager(engine, tmp_path)
        real = engine.synthesize

        def synth_then_cancel(*a, **k):
            res = real(*a, **k)
            k["cancel"].set()                    # cancel lands after the last chunk check
            return res
        monkeypatch.setattr(engine, "synthesize", synth_then_cancel)
        j = jm.wait(jm.submit("Hello there.", "", None, {}).id)
        assert j.status == "cancelled"
        assert list(tmp_path.glob("*")) == []

    def test_cancel_after_commit_is_a_noop(self, engine, tmp_path, monkeypatch):
        from chatterbox_studio import jobs as J
        jm = J.JobManager(engine, tmp_path)
        real = J.save_result

        def save_and_cancel(*a, **k):
            jm.cancel(jm.list()[0].id)           # arrives while committing
            return real(*a, **k)
        monkeypatch.setattr(J, "save_result", save_and_cancel)
        j = jm.wait(jm.submit("Hello there.", "", None, {}).id)
        assert j.status == "done" and Path(j.paths["audio"]).is_file()

    def test_failed_write_leaves_no_partial_file(self, engine, tmp_path, monkeypatch):
        from chatterbox_studio.jobs import save_result
        res = engine.synthesize("Hello.", None)

        def boom(src, dst):
            raise OSError("disk full")
        monkeypatch.setattr(os, "fsync", lambda fd: boom(fd, None))   # every write path syncs
        with pytest.raises(OSError):
            save_result(res, "wav", tmp_path / "out.wav")
        assert list(tmp_path.iterdir()) == []


# ---- CBX-05: takes never overwrite each other ---------------------------------------
def test_same_text_same_second_keeps_both(engine, tmp_path):
    from chatterbox_studio.jobs import history, save_result
    a = save_result(engine.synthesize("Take one.", None), text="Same words", out_dir=tmp_path)
    b = save_result(engine.synthesize("Take two, longer.", None), text="Same words", out_dir=tmp_path)
    assert a["audio"] != b["audio"]
    assert len(list(tmp_path.glob("*.wav"))) == 2 and len(history(tmp_path)) == 2


# ---- CBX-06: web pages can't drive the API ------------------------------------------
class TestBrowserGuard:
    @pytest.fixture
    def app(self, engine, store, tmp_path):
        from chatterbox_studio.jobs import JobManager
        return create_app(engine, store, JobManager(engine, tmp_path))

    def test_cross_origin_form_post_blocked(self, app, store):
        c = TestClient(app, base_url="http://127.0.0.1:8765")
        r = c.post("/voices", data={"name": "planted", "consent": "true"},
                   files={"file": ("x.wav", _wav())}, headers={"Origin": "https://evil.example"})
        assert r.status_code == 403 and store.list() == []

    def test_dns_rebinding_host_blocked_without_token(self, app):
        c = TestClient(app, base_url="http://attacker.example:8765")
        assert c.get("/voices").status_code == 403

    def test_legit_clients_still_work(self, app, monkeypatch):
        c = TestClient(app, base_url="http://localhost:8765")
        assert c.get("/voices").status_code == 200                        # scripts: no Origin
        assert c.get("/voices", headers={"Origin": "http://localhost:8765"}).status_code == 200  # /docs
        monkeypatch.setattr(config, "ALLOWED_ORIGINS", "https://my.app", raising=False)
        assert c.get("/voices", headers={"Origin": "https://my.app"}).status_code == 200

    def test_lan_host_allowed_with_token(self, app, monkeypatch):
        monkeypatch.setattr(config, "API_TOKEN", "tok")
        c = TestClient(app, base_url="http://192.168.0.163:8765")
        assert c.get("/voices", headers={"Authorization": "Bearer tok"}).status_code == 200


# ---- found while verifying against chatterbox-tts 0.1.7 -----------------------------
def test_turbo_is_not_sent_ignored_controls(engine, factory):
    TurboFake.warnings = 0
    engine.synthesize("One. Two. Three.", None, {"model": "turbo"}, chunk_limit=5)
    assert TurboFake.warnings == 0
    assert all(c["exaggeration"] == 0.0 and c["cfg_weight"] == 0.0 for c in factory.made[0].calls)


def test_nano_unavailable_gives_accurate_message(monkeypatch):
    from chatterbox_studio import engine as E

    class Turbo017:                               # 0.1.7: from_pretrained(cls, device)
        @classmethod
        def from_pretrained(cls, device):
            return FakeModel("turbo")
    mod = types.ModuleType("chatterbox.tts_turbo")
    mod.ChatterboxTurboTTS = Turbo017
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", mod)
    with pytest.raises(E.EngineError, match="GitHub"):
        E.default_factory("nano", "cpu")
    assert E.default_factory("turbo", "cpu").kind == "turbo"


def test_multilingual_v2_loaded_without_v3_flag(monkeypatch):
    from chatterbox_studio import engine as E
    seen = {}

    class Mtl017:
        @classmethod
        def from_pretrained(cls, device):
            seen["ok"] = True
            return FakeModel("multilingual")
    mod = types.ModuleType("chatterbox.mtl_tts")
    mod.ChatterboxMultilingualTTS = Mtl017
    monkeypatch.setitem(sys.modules, "chatterbox", types.ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mod)
    E.default_factory("multilingual", "cpu")
    assert seen == {"ok": True}
