"""MCP server (stdio). Thin client over the local REST API so one process owns the model.
Start the API first:  python -m chatterbox_studio serve
"""
import os
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from . import config

BASE = os.environ.get("CBX_URL", f"http://{config.API_HOST}:{config.API_PORT}")
HEADERS = {"Authorization": f"Bearer {config.API_TOKEN}"} if config.API_TOKEN else {}

mcp = FastMCP("chatterbox-studio")


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE, headers=HEADERS, timeout=600)


def _check(r: httpx.Response):
    if r.status_code >= 400:
        raise RuntimeError(f"Chatterbox API {r.status_code}: {r.text}")
    return r


@mcp.tool()
def list_voices() -> list[dict]:
    """List the cloned voices available for speech synthesis."""
    with _client() as c:
        return _check(c.get("/voices")).json()


@mcp.tool()
def speak(text: str, voice_id: str = "", language: str | None = None,
          exaggeration: float | None = None, cfg_weight: float | None = None,
          play: bool = False) -> str:
    """Synthesize `text` in a cloned voice (voice_id from list_voices; empty = default voice).
    Unset options fall back to the voice's saved defaults. `language` is a code like
    en, fr, ja, zh. Returns the path of the generated WAV file."""
    body = {"text": text, "voice_id": voice_id, "language": language,
            "exaggeration": exaggeration, "cfg_weight": cfg_weight}
    with _client() as c:
        r = _check(c.post("/tts", json={k: v for k, v in body.items() if v is not None}))
    out = config.OUT_DIR / f"mcp_{int(time.time())}.wav"
    out.write_bytes(r.content)
    if play:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(str(out))
        sd.play(data, sr)
        sd.wait()
    return str(out)


@mcp.tool()
def set_voice_defaults(voice_id: str, exaggeration: float | None = None,
                       cfg_weight: float | None = None, temperature: float | None = None,
                       language: str | None = None) -> dict:
    """Save default synthesis settings on a voice (used whenever speak() leaves them unset)."""
    body = {"exaggeration": exaggeration, "cfg_weight": cfg_weight,
            "temperature": temperature, "language": language}
    with _client() as c:
        return _check(c.put(f"/voices/{voice_id}/settings",
                            json={k: v for k, v in body.items() if v is not None})).json()


@mcp.tool()
def clone_voice(name: str, audio_path: str, i_have_consent: bool) -> dict:
    """Create a voice from a local audio file (5-30s of clean speech). Only use a voice
    you own or have explicit permission to clone; i_have_consent must be true."""
    p = Path(audio_path)
    if not p.is_file():
        raise RuntimeError(f"File not found: {audio_path}")
    with _client() as c:
        r = c.post("/voices", data={"name": name, "consent": str(i_have_consent).lower()},
                   files={"file": (p.name, p.read_bytes())})
    return _check(r).json()


def main():
    mcp.run()
