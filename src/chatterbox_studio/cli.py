# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Command line.

    python -m chatterbox_studio                 open the app (same as `gui`)
    python -m chatterbox_studio say "Hello" --voice Leon -o hello.mp3
    python -m chatterbox_studio say --file chapter1.md --srt --preset narrator
    python -m chatterbox_studio voices          list voices
    python -m chatterbox_studio clone me.wav --name Leon --consent
    python -m chatterbox_studio serve [--host 0.0.0.0 --token ...]
    python -m chatterbox_studio mcp             MCP stdio server
    python -m chatterbox_studio doctor          check the install
"""
import argparse
import importlib
import platform
import shutil
import socket
import sys
from pathlib import Path

from . import __version__, config


def _out(msg=""):
    print(msg, flush=True)


def cmd_say(a):
    from .engine import EngineError, TTSEngine
    from .jobs import save_result
    from .voices import VoiceError, VoiceNotFound, VoiceStore, effective_settings

    if a.file:
        text = Path(a.file).read_text(encoding="utf-8")
    elif a.text == "-" or (a.text is None and not sys.stdin.isatty()):
        text = sys.stdin.read()
    else:
        text = a.text
    if not text or not text.strip():
        sys.exit("Nothing to say. Pass text, --file, or pipe text in.")

    store = VoiceStore()
    voice = None
    if a.voice:
        try:
            voice = store.resolve(a.voice)
        except (VoiceNotFound, VoiceError) as e:
            names = ", ".join(v.name for v in store.list()) or "(none yet)"
            sys.exit(f"Voice '{a.voice}' not found ({e}). Available: {names}")

    ov = {}
    if a.preset:
        ov.update({k: v for k, v in config.PRESETS[a.preset].items() if k != "label"})
    for k in ("language", "model", "exaggeration", "cfg_weight", "temperature", "speed"):
        if getattr(a, k) is not None:
            ov[k] = getattr(a, k)
    s = effective_settings(voice, ov)

    out = Path(a.output) if a.output else None
    fmt = a.format or (out.suffix.lstrip(".").lower() if out and out.suffix else "wav")
    if fmt not in config.FORMATS or fmt == "pcm":
        sys.exit(f"Unsupported format '{fmt}'.")

    engine = TTSEngine(a.device)

    def progress(done, total):
        if total > 1:
            print(f"\r  {done}/{total} chunks", end="", file=sys.stderr, flush=True)

    _out(f"Loading {s['model']} model on {engine.device} (first run downloads weights)…")
    try:
        res = engine.synthesize(text, voice.path if voice else None, s, seed=a.seed,
                                on_progress=progress)
    except EngineError as e:
        sys.exit(f"\nFailed: {e}")
    print(file=sys.stderr)
    paths = save_result(res, fmt, out, text=text, voice=voice.id if voice else "",
                        settings=s, srt=a.srt)
    _out(f"Saved {res.seconds:.1f}s → {paths['audio']}")
    if "srt" in paths:
        _out(f"Subtitles → {paths['srt']}")
    if a.play:
        import sounddevice as sd
        sd.play(res.audio, res.sr)
        sd.wait()


def cmd_voices(a):
    from .voices import VoiceStore
    vs = VoiceStore().list()
    if not vs:
        _out("No voices yet. Record one in the app, or: chatterbox_studio clone file.wav "
             "--name Me --consent")
    for v in vs:
        q = v.quality.get("verdict", "?") if v.quality else "?"
        _out(f"{v.name:<24} {v.duration:5.1f}s  quality:{q:<6} id:{v.id}")


def cmd_clone(a):
    from .voices import VoiceError, VoiceStore
    if not a.consent:
        sys.exit("Add --consent to confirm this is your voice or you have permission to clone it.")
    try:
        v = VoiceStore().add_file(a.name, a.audio, True)
    except VoiceError as e:
        sys.exit(str(e))
    _out(f"Saved voice '{v.name}' (id {v.id}).")
    for tip in v.quality.get("tips", []):
        _out(f"  • {tip}")


def cmd_serve(a):
    import uvicorn
    from .api import check_bind, create_app
    from .engine import TTSEngine
    from .voices import VoiceStore
    host = a.host or config.API_HOST
    port = a.port or config.API_PORT
    if a.token:
        config.API_TOKEN = a.token
    try:
        check_bind(host, config.API_TOKEN, config.ALLOW_INSECURE)
    except RuntimeError as e:
        sys.exit(str(e))
    engine = TTSEngine(a.device)
    if a.preload:
        _out(f"Preloading model on {engine.device}…")
        engine.load()
    _out(f"Chatterbox Studio API → http://{host}:{port}  (docs at /docs)")
    uvicorn.run(create_app(engine, VoiceStore()), host=host, port=port, log_level="warning")


def _check(label, ok, detail=""):
    mark = "OK " if ok is True else ("-- " if ok is None else "!! ")
    _out(f"  [{mark}] {label}{(': ' + detail) if detail else ''}")
    return ok is not False


def cmd_doctor(a):
    _out(f"Chatterbox Studio {__version__} — install check\n")
    ok = True
    v = sys.version_info
    ok &= _check("Python", (3, 10) <= v[:2] < (3, 13),
                 f"{platform.python_version()} (needs 3.10–3.12; 3.11 recommended)")
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "no CUDA GPU — will run on CPU (slow)"
        _check("PyTorch", True, f"{torch.__version__}")
        _check("GPU", True if cuda else None, dev)
        if cuda:
            free, total = torch.cuda.mem_get_info()
            _check("GPU memory", total > 5e9 or None,
                   f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
    except ImportError:
        ok &= _check("PyTorch", False, "not installed — run install.bat")
    for mod, label in (("chatterbox", "chatterbox-tts"), ("PyQt6", "PyQt6 (app)"),
                       ("fastapi", "FastAPI (API)"), ("mcp", "MCP SDK"),
                       ("librosa", "librosa (speed control)")):
        try:
            m = importlib.import_module(mod)
            ok &= _check(label, True, getattr(m, "__version__", "installed"))
        except Exception as e:
            ok &= _check(label, False, f"{type(e).__name__}: {e}")
    from .engine import capabilities
    caps = capabilities()
    if caps["standard"]:
        _check("Turbo model", caps["turbo"] or None,
               "available" if caps["turbo"] else "needs a newer chatterbox-tts")
        _check("Nano model", caps["nano"] or None, "available" if caps["nano"]
               else "not in the PyPI release; install chatterbox from GitHub to use it")
        _check("Multilingual", caps["multilingual"] or None,
               "V3" if caps["multilingual_v3"] else ("V2" if caps["multilingual"] else "missing"))
    try:
        import sounddevice as sd
        inp = sd.query_devices(kind="input")
        outp = sd.query_devices(kind="output")
        _check("Microphone", True, inp["name"])
        _check("Speakers", True, outp["name"])
    except Exception as e:
        ok &= _check("Audio devices", False, str(e))
    _check("ffmpeg (m4a/aac imports)", True if shutil.which("ffmpeg") else None,
           shutil.which("ffmpeg") or "not found; WAV/FLAC/OGG/MP3 still work")
    try:
        probe = config.DATA_DIR / ".write-test"
        probe.write_text("x")
        probe.unlink()
        _check("Data folder", True, str(config.DATA_DIR))
    except Exception as e:
        ok &= _check("Data folder", False, str(e))
    with socket.socket() as s:
        free = s.connect_ex((config.API_HOST, config.API_PORT)) != 0
    _check(f"Port {config.API_PORT}", True if free else None,
           "free" if free else "in use (the API may already be running)")
    _out("\nAll good." if ok else "\nSome checks failed — see above.")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="chatterbox_studio",
                                 description="Clone your voice and use it anywhere.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the app")
    sp = sub.add_parser("serve", help="run the HTTP API")
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.add_argument("--token", help="require this Bearer token")
    sp.add_argument("--preload", action="store_true", help="load the model at startup")
    sub.add_parser("mcp", help="MCP stdio server (needs the API running)")
    sub.add_parser("voices", help="list voices")
    sub.add_parser("doctor", help="check the install")
    cp = sub.add_parser("clone", help="create a voice from an audio file")
    cp.add_argument("audio")
    cp.add_argument("--name", required=True)
    cp.add_argument("--consent", action="store_true")
    s = sub.add_parser("say", help="speak text to a file")
    s.add_argument("text", nargs="?", help='text, or "-" to read stdin')
    s.add_argument("--file", "-f", help="read text from a .txt/.md file")
    s.add_argument("--voice", "-v", help="voice name or id")
    s.add_argument("--output", "-o", help="output file (extension picks the format)")
    s.add_argument("--format", choices=[f for f in config.FORMATS if f != "pcm"])
    s.add_argument("--preset", choices=list(config.PRESETS))
    s.add_argument("--language", "-l", choices=list(config.LANGUAGES))
    s.add_argument("--model", "-m", choices=list(config.MODELS))
    s.add_argument("--exaggeration", type=float)
    s.add_argument("--cfg-weight", dest="cfg_weight", type=float)
    s.add_argument("--temperature", type=float)
    s.add_argument("--speed", type=float)
    s.add_argument("--seed", type=int, help="repeatable output")
    s.add_argument("--srt", action="store_true", help="also write subtitles")
    s.add_argument("--play", action="store_true")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    cmd = a.cmd or "gui"
    if cmd == "gui":
        from .gui import run_gui
        run_gui(a.device)
    elif cmd == "mcp":
        from .mcp_server import main as run_mcp
        run_mcp()
    elif cmd == "doctor":
        sys.exit(cmd_doctor(a))
    else:
        {"say": cmd_say, "voices": cmd_voices, "clone": cmd_clone, "serve": cmd_serve}[cmd](a)
