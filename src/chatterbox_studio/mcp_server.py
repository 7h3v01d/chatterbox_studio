# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""MCP server (stdio): lets Claude Desktop and other MCP clients speak with your voices.

It is a thin client over the local REST API, so one process owns the model. Start the
API first (GUI -> Connect -> Start, or `python -m chatterbox_studio serve`).

Deny-first: tools can list, speak and clone (with consent). Deleting voices is not
exposed to the model.
"""
import os
import time
from pathlib import Path

import httpx
try:                                    # mcp >= 2: FastMCP was renamed MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:                     # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from . import config

BASE = os.environ.get("CBX_URL", f"http://{config.API_HOST}:{config.API_PORT}")
HEADERS = {"Authorization": f"Bearer {config.API_TOKEN}"} if config.API_TOKEN else {}

mcp = FastMCP("chatterbox-studio")


def _client(timeout: float = 600) -> httpx.Client:
    return httpx.Client(base_url=BASE, headers=HEADERS, timeout=timeout)


def _call(method: str, path: str, timeout: float = 600, **kw) -> httpx.Response:
    try:
        with _client(timeout) as c:
            r = c.request(method, path, **kw)
    except httpx.ConnectError:
        raise RuntimeError(
            f"Chatterbox Studio isn't running at {BASE}. Open the app and click "
            "Connect -> Start API server, or run:  python -m chatterbox_studio serve")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"Chatterbox Studio error {r.status_code}: {detail}")
    return r


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _play(path: str):
    import sounddevice as sd
    import soundfile as sf
    data, sr = sf.read(path)
    sd.play(data, sr)
    sd.wait()


@mcp.tool()
def status() -> dict:
    """Check that Chatterbox Studio is running and whether a model is loaded."""
    return _call("GET", "/health", timeout=10).json()


@mcp.tool()
def list_voices() -> list[dict]:
    """List the voices available for speech (id, name, length, saved defaults)."""
    return _call("GET", "/voices").json()


@mcp.tool()
def list_presets() -> dict:
    """Style presets that can be passed to speak(): natural, narrator, calm, expressive,
    dramatic, accent."""
    return {k: v["label"] for k, v in _call("GET", "/presets").json().items()}


@mcp.tool()
def speak(text: str, voice: str = "", language: str | None = None,
          preset: str | None = None, model: str | None = None,
          speed: float | None = None, exaggeration: float | None = None,
          cfg_weight: float | None = None, format: str = "wav",
          play: bool = False, save_to: str | None = None) -> str:
    """Turn `text` into speech and return the saved audio file's path.

    voice: a voice id or name from list_voices ("" = the built-in voice).
    language: code like en, fr, de, ja, zh (non-English needs a cloned voice).
    preset: natural | narrator | calm | expressive | dramatic | accent.
    model: auto | standard | turbo | nano | multilingual. Turbo is fastest and understands
           [laugh], [chuckle] and [cough] tags in the text.
    speed: 0.5-2.0. Text may include [pause 1s] markers; blank lines become paragraph pauses.
    format: wav | flac | mp3 | ogg | opus.  play: also play it out loud on this machine.
    Long texts (over ~2000 characters) are handled as a background job automatically.
    """
    opts = _clean({"voice_id": voice, "language": language, "preset": preset, "model": model,
                   "speed": speed, "exaggeration": exaggeration, "cfg_weight": cfg_weight})
    fmt = format.lower()
    if len(text) > 2000:
        return speak_long(text, voice, language, preset, model, speed, fmt, False, save_to,
                          play=play)
    r = _call("POST", "/tts", json={"text": text, "format": fmt, **opts})
    out = Path(save_to) if save_to else config.OUT_DIR / f"mcp_{int(time.time() * 1000)}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    if play:
        _play(str(out))
    return f"{out} ({r.headers.get('X-Audio-Seconds', '?')} s)"


@mcp.tool()
def speak_long(text: str, voice: str = "", language: str | None = None,
               preset: str | None = "narrator", model: str | None = None,
               speed: float | None = None, format: str = "mp3", subtitles: bool = True,
               save_to: str | None = None, play: bool = False) -> str:
    """Narrate long text (articles, scripts, chapters) as a background job and wait for
    it. Writes an audio file and, with subtitles=True, a matching .srt file. Returns the
    paths. Markdown is cleaned automatically."""
    body = _clean({"text": text, "voice_id": voice, "language": language, "preset": preset,
                   "model": model, "speed": speed, "format": format.lower(), "srt": subtitles})
    job = _call("POST", "/jobs", json=body).json()
    while job["status"] in ("queued", "running"):
        time.sleep(1.0)
        job = _call("GET", f"/jobs/{job['id']}", timeout=30).json()
    if job["status"] != "done":
        raise RuntimeError(f"Job {job['status']}: {job.get('error', '')}")
    out = Path(save_to) if save_to else config.OUT_DIR / f"mcp_{job['id']}.{job['fmt']}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_call("GET", f"/jobs/{job['id']}/audio").content)
    result = [f"audio: {out} ({job['seconds']} s)"]
    if subtitles:
        sp = out.with_suffix(".srt")
        sp.write_text(_call("GET", f"/jobs/{job['id']}/srt").text, encoding="utf-8")
        result.append(f"subtitles: {sp}")
    if play:
        _play(str(out))
    return "\n".join(result)


@mcp.tool()
def set_voice_defaults(voice_id: str, exaggeration: float | None = None,
                       cfg_weight: float | None = None, temperature: float | None = None,
                       language: str | None = None, model: str | None = None,
                       speed: float | None = None) -> dict:
    """Save default settings on a voice (used whenever speak() leaves them unset)."""
    body = _clean({"exaggeration": exaggeration, "cfg_weight": cfg_weight,
                   "temperature": temperature, "language": language, "model": model,
                   "speed": speed})
    return _call("PUT", f"/voices/{voice_id}/settings", json=body).json()


@mcp.tool()
def clone_voice(name: str, audio_path: str, i_have_consent: bool) -> dict:
    """Create a voice from a local audio file (about 10 s of clean speech is ideal).
    Only clone a voice you own or have explicit permission to clone; i_have_consent must
    be true. The result includes a quality report with tips."""
    p = Path(audio_path).expanduser()
    if not p.is_file():
        raise RuntimeError(f"File not found: {audio_path}")
    return _call("POST", "/voices", data={"name": name, "consent": str(i_have_consent).lower()},
                 files={"file": (p.name, p.read_bytes())}).json()


def main():
    mcp.run()
