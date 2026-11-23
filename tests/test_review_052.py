# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Regression tests for the 0.5.1 re-review (CBX-07..10). Each fails on 0.5.1."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

from chatterbox_studio import config, jobs as J
from chatterbox_studio.api import create_app


# ---- CBX-07: no-overwrite publication must be atomic -------------------------------
def test_competing_writer_is_never_overwritten(tmp_path, monkeypatch):
    """A rival process creates the destination after our existence check, while our temp
    file is being flushed. The rival's file must survive and we must report the clash."""
    dest = tmp_path / "take.wav"
    real_fsync = os.fsync

    def fsync_then_rival(fd):
        real_fsync(fd)
        if not dest.exists():
            dest.write_bytes(b"RIVAL")                  # interleaved writer
    monkeypatch.setattr(os, "fsync", fsync_then_rival)
    with pytest.raises(FileExistsError):
        J.atomic_write(dest, b"OURS", overwrite=False)
    assert dest.read_bytes() == b"RIVAL"
    assert [p.name for p in tmp_path.iterdir()] == ["take.wav"]      # no temp left behind


def test_overwrite_true_still_replaces(tmp_path):
    dest = tmp_path / "x.wav"
    dest.write_bytes(b"old")
    J.atomic_write(dest, b"new", overwrite=True)
    assert dest.read_bytes() == b"new"


# ---- CBX-08: audio + subtitles + sidecar commit together or not at all ---------------
def _fail_on(suffix, monkeypatch):
    real = J.atomic_write

    def flaky(path, data, overwrite=True):
        if str(path).endswith(suffix):
            raise OSError(f"disk full writing {suffix}")
        return real(path, data, overwrite)
    monkeypatch.setattr(J, "atomic_write", flaky)


@pytest.mark.parametrize("failing", [".srt", ".json"])
def test_partial_outputs_rolled_back(engine, tmp_path, monkeypatch, failing):
    res = engine.synthesize("Hello there.", None)
    _fail_on(failing, monkeypatch)
    with pytest.raises(OSError):
        J.save_result(res, "wav", text="Hello", srt=True, out_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_rollback_restores_preexisting_files(engine, tmp_path, monkeypatch):
    """Explicit destination over existing files (overwrite policy) + later failure:
    the user's original files come back unchanged, nothing is deleted."""
    res = engine.synthesize("Hello there.", None)
    dest = tmp_path / "mine.wav"
    dest.write_bytes(b"ORIGINAL AUDIO")
    dest.with_suffix(".srt").write_bytes(b"ORIGINAL SUBS")
    _fail_on(".json", monkeypatch)
    with pytest.raises(OSError):
        J.save_result(res, "wav", dest, srt=True)
    assert dest.read_bytes() == b"ORIGINAL AUDIO"
    assert dest.with_suffix(".srt").read_bytes() == b"ORIGINAL SUBS"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mine.srt", "mine.wav"]


def test_job_failure_leaves_no_orphans(engine, tmp_path, monkeypatch):
    jm = J.JobManager(engine, tmp_path)
    _fail_on(".srt", monkeypatch)
    j = jm.wait(jm.submit("Hello there.", "", None, {}, srt=True).id)
    assert j.status == "failed" and "disk full" in j.error
    assert list(tmp_path.glob("*")) == []


# ---- CBX-09: long-text delegation keeps every setting --------------------------------
class TestMcpDelegation:
    @pytest.fixture
    def mcp(self, monkeypatch):
        pytest.importorskip("mcp")
        from chatterbox_studio import mcp_server as m
        sent = []

        class R:
            content = b"RIFF"
            text = "1\n00:00:00,000 --> 00:00:01,000\nx\n"
            headers = {"X-Audio-Seconds": "1"}

            def __init__(self, body=None):
                self._body = body

            def json(self):
                return {"id": "job1", "status": "done", "fmt": "wav", "seconds": 1.0}

        def fake_call(method, path, timeout=600, **kw):
            if "json" in kw:
                sent.append((path, kw["json"]))
            return R()
        monkeypatch.setattr(m, "_call", fake_call)
        m.sent = sent
        return m

    def test_short_and_long_send_the_same_options(self, mcp):
        opts = dict(voice="v", language="en", preset="dramatic", model="standard",
                    speed=1.2, exaggeration=1.4, cfg_weight=0.2, temperature=0.6)
        mcp.speak("Short.", **opts)
        mcp.speak("Long. " * 500, **opts)
        (p1, short), (p2, long_) = mcp.sent
        assert (p1, p2) == ("/tts", "/jobs")
        keys = ("voice_id", "language", "preset", "model", "speed",
                "exaggeration", "cfg_weight", "temperature")
        assert {k: short.get(k) for k in keys} == {k: long_.get(k) for k in keys}
        assert long_["exaggeration"] == 1.4 and long_["cfg_weight"] == 0.2


# ---- CBX-10: only the API's own origin (plus configured ones) -----------------------
class TestExactOrigin:
    @pytest.fixture
    def client(self, engine, store, tmp_path):
        return TestClient(create_app(engine, store, J.JobManager(engine, tmp_path)),
                          base_url="http://127.0.0.1:8765")

    @pytest.mark.parametrize("origin", ["http://localhost:9999", "http://127.0.0.1:3000",
                                        "https://127.0.0.1:8765", "http://[::1]:8765",
                                        "null"])
    def test_other_local_origins_rejected(self, client, origin):
        assert client.get("/voices", headers={"Origin": origin}).status_code == 403

    def test_own_origin_and_configured_origin_allowed(self, client, monkeypatch):
        assert client.get("/voices", headers={"Origin": "http://127.0.0.1:8765"}).status_code == 200
        monkeypatch.setattr(config, "ALLOWED_ORIGINS", "http://localhost:3000")
        assert client.get("/voices", headers={"Origin": "http://localhost:3000"}).status_code == 200
        assert client.get("/voices", headers={"Origin": "http://localhost:3001"}).status_code == 403


# ---- MCP: validation and creation are separate steps; the write re-checks ----------
@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin on Windows")
class TestMcpWriteRaces:
    @pytest.fixture
    def m(self, monkeypatch):
        pytest.importorskip("mcp")
        from chatterbox_studio import mcp_server as m
        monkeypatch.setattr(config, "MCP_OUTPUT_DIRS", "")
        return m

    def test_symlink_planted_after_validation(self, m, tmp_path, monkeypatch):
        outside = tmp_path / "victim.wav"
        outside.write_bytes(b"PRECIOUS")
        dest = config.OUT_DIR / f"race_{os.getpid()}.wav"

        class R:
            content = b"RIFF"
            headers = {}

        def call_and_plant(*a, **k):                 # runs between validate and write
            dest.symlink_to(outside)
            return R()
        monkeypatch.setattr(m, "_call", call_and_plant)
        try:
            with pytest.raises(FileExistsError):
                m.speak("hi", save_to=dest.name)
            assert outside.read_bytes() == b"PRECIOUS"
        finally:
            dest.unlink(missing_ok=True)

    def test_parent_swapped_to_symlink_after_validation(self, m, tmp_path, monkeypatch):
        outside = tmp_path / "outside"
        outside.mkdir()
        sub = config.OUT_DIR / f"sub_{os.getpid()}"
        sub.mkdir()

        class R:
            content = b"RIFF"
            headers = {}

        def call_and_swap(*a, **k):
            sub.rmdir()
            sub.symlink_to(outside, target_is_directory=True)
            return R()
        monkeypatch.setattr(m, "_call", call_and_swap)
        try:
            with pytest.raises(RuntimeError, match="outside"):
                m.speak("hi", save_to=f"{sub.name}/x.wav")
            assert list(outside.iterdir()) == []
        finally:
            sub.unlink()


def test_no_hardlink_filesystem_fallback_still_never_replaces(tmp_path, monkeypatch):
    """FAT/exFAT/some shares can't hard-link: fall back to exclusive create."""
    import errno

    def no_links(src, dst):
        raise OSError(errno.EPERM, "hard links not supported")
    monkeypatch.setattr(os, "link", no_links)
    dest = tmp_path / "a.wav"
    J.atomic_write(dest, b"ONE", overwrite=False)
    assert dest.read_bytes() == b"ONE"
    with pytest.raises(FileExistsError):
        J.atomic_write(dest, b"TWO", overwrite=False)
    assert dest.read_bytes() == b"ONE" and len(list(tmp_path.iterdir())) == 1
