# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
import io
import time

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from chatterbox_studio import config
from chatterbox_studio.api import check_bind, create_app

from conftest import SR, speech_like


@pytest.fixture
def client(engine, store, tmp_path):
    from chatterbox_studio.jobs import JobManager
    return TestClient(create_app(engine, store, JobManager(engine, tmp_path / "jobs")))


def _wav():
    buf = io.BytesIO()
    sf.write(buf, speech_like(), SR, format="WAV")
    return buf.getvalue()


def _upload(client, name="Leon", consent="true"):
    return client.post("/voices", data={"name": name, "consent": consent},
                       files={"file": ("me.wav", _wav())})


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is False


def test_voice_lifecycle(client):
    v = _upload(client).json()
    assert v["quality"]["verdict"] in ("great", "ok") and "path" not in v
    assert client.get(f"/voices/{v['id']}/audio").content[:4] == b"RIFF"
    assert client.patch(f"/voices/{v['id']}", json={"name": "L2"}).json()["name"] == "L2"
    r = client.put(f"/voices/{v['id']}/settings", json={"model": "turbo", "speed": 1.2})
    assert r.json()["settings"] == {"model": "turbo", "speed": 1.2}
    assert client.delete(f"/voices/{v['id']}").status_code == 200
    assert client.get("/voices").json() == []


def test_upload_without_consent(client):
    assert _upload(client, consent="false").status_code == 400


def test_tts_formats_and_voice_by_name(client):
    _upload(client, "Leon")
    for fmt, magic in (("wav", b"RIFF"), ("flac", b"fLaC"), ("ogg", b"OggS")):
        r = client.post("/tts", json={"text": "Hello world.", "voice_id": "leon", "format": fmt})
        assert r.status_code == 200, r.text
        assert r.content[:4] == magic and float(r.headers["x-audio-seconds"]) > 0


def test_tts_errors(client):
    assert client.post("/tts", json={"text": "x", "voice_id": "ghost"}).status_code == 404
    assert client.post("/tts", json={"text": "x", "format": "aiff"}).status_code == 400
    assert client.post("/tts", json={"text": "x", "preset": "nope"}).status_code == 400
    r = client.post("/tts", json={"text": "x" * (config.MAX_CHARS + 1)})
    assert r.status_code == 413 and "/jobs" in r.json()["detail"]
    r = client.post("/tts", json={"text": "Bonjour", "language": "fr"})
    assert r.status_code == 500 and "built-in voice" in r.json()["detail"]


def test_preset_applies(client, factory):
    client.post("/tts", json={"text": "Hi.", "preset": "dramatic"})
    call = factory.made[0].calls[0]
    assert call["exaggeration"] == 1.2 and call["cfg_weight"] == 0.3


def test_stream(client):
    r = client.post("/tts/stream", json={"text": "One. Two. Three. " * 20})
    assert r.status_code == 200 and r.content[:4] == b"RIFF" and len(r.content) > 44


def test_jobs_end_to_end(client):
    j = client.post("/jobs", json={"text": "Long one. " * 50, "format": "mp3", "srt": True}).json()
    for _ in range(200):
        j = client.get(f"/jobs/{j['id']}").json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.02)
    assert j["status"] == "done", j
    assert client.get(f"/jobs/{j['id']}/audio").status_code == 200
    assert "-->" in client.get(f"/jobs/{j['id']}/srt").text
    assert client.post("/jobs", json={"text": "x", "model": "turbo", "language": "fr"}).status_code == 400


def test_openai_compat(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi there.", "voice": "alloy"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy",
                                              "response_format": "aac"})
    assert r.status_code == 400
    assert client.get("/v1/models").json()["data"][0]["id"] == "chatterbox"


def test_token(client, monkeypatch):
    monkeypatch.setattr(config, "API_TOKEN", "s3cret")
    assert client.get("/voices").status_code == 401
    assert client.get("/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_deny_first_bind():
    check_bind("127.0.0.1", "")
    check_bind("0.0.0.0", "tok")
    with pytest.raises(RuntimeError, match="without a token"):
        check_bind("0.0.0.0", "")


# REGRESSION (0.4): POST /voices wrote the upload to a NamedTemporaryFile and re-opened it
# by name while still open, which Windows refuses. Uploads must never touch a temp file.
def test_upload_does_not_use_tempfile(client, monkeypatch):
    import tempfile

    def boom(*a, **k):
        raise PermissionError("[Errno 13] Permission denied (Windows held-open temp file)")
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", boom)
    assert _upload(client).status_code == 200
