# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Paths, environment settings, presets and model catalogue.

Everything user-tunable comes from environment variables so the GUI, the API, the
CLI and the MCP server all agree without a config file:

    CBX_DATA        data folder (voices, output, jobs)    default ~/.chatterbox_studio
    CBX_HOST        API bind address                      default 127.0.0.1
    CBX_PORT        API port                              default 8765
    CBX_TOKEN       Bearer token; required for any non-loopback bind
    CBX_URL         where the MCP server finds the API    default http://HOST:PORT
    CBX_MAX_CHARS   max characters per /tts request        default 20000
    CBX_ALLOW_INSECURE=1   allow a LAN bind without a token (not recommended)
    CBX_ALLOWED_ORIGINS    web origins allowed to call the API from a browser
    CBX_MCP_OUTPUT_DIRS    extra folders the MCP server may save audio into
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CBX_DATA", Path.home() / ".chatterbox_studio"))
VOICES_DIR = DATA_DIR / "voices"
OUT_DIR = DATA_DIR / "output"
JOBS_DIR = DATA_DIR / "jobs"

API_HOST = os.environ.get("CBX_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("CBX_PORT", "8765"))
API_TOKEN = os.environ.get("CBX_TOKEN", "")
ALLOW_INSECURE = os.environ.get("CBX_ALLOW_INSECURE", "") == "1"
ALLOWED_ORIGINS = os.environ.get("CBX_ALLOWED_ORIGINS", "")   # comma-separated web origins
# Extra folders (os.pathsep-separated) the MCP server may write audio into, besides output/
MCP_OUTPUT_DIRS = os.environ.get("CBX_MCP_OUTPUT_DIRS", "")
MAX_CHARS = int(os.environ.get("CBX_MAX_CHARS", "20000"))
MAX_JOB_CHARS = 500_000
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def ensure_dirs():
    for d in (VOICES_DIR, OUT_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# --- models ---------------------------------------------------------------------
# "auto" = Standard for English, Multilingual for anything else.
MODELS = {
    "auto": "Auto (best for the language)",
    "standard": "Standard — English, emotion + pacing controls",
    "turbo": "Turbo — English, fast, supports [laugh] [chuckle] [cough] tags",
    "nano": "Nano — English, smallest, runs on CPU",
    "multilingual": "Multilingual — 23 languages",
}
ENGLISH_ONLY = {"standard", "turbo", "nano"}

# --- synthesis settings -----------------------------------------------------------
DEFAULT_SETTINGS = {
    "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8,
    "language": "en", "model": "auto", "speed": 1.0,
    "sentence_pause": 0.25, "paragraph_pause": 0.7,
}
SETTING_RANGES = {
    "exaggeration": (0.25, 2.0), "cfg_weight": (0.0, 1.0), "temperature": (0.05, 5.0),
    "speed": (0.5, 2.0), "sentence_pause": (0.0, 3.0), "paragraph_pause": (0.0, 5.0),
}

# Plain-language presets. Values follow Resemble's own tuning advice: defaults suit most
# text; expressive speech wants lower cfg_weight and higher exaggeration; cfg_weight 0
# reduces accent bleed when the reference voice speaks a different language.
PRESETS = {
    "natural":    {"label": "Natural",       "exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    "narrator":   {"label": "Narrator",      "exaggeration": 0.4, "cfg_weight": 0.5, "temperature": 0.65},
    "calm":       {"label": "Calm",          "exaggeration": 0.3, "cfg_weight": 0.6, "temperature": 0.6},
    "expressive": {"label": "Expressive",    "exaggeration": 0.8, "cfg_weight": 0.3, "temperature": 0.8},
    "dramatic":   {"label": "Dramatic",      "exaggeration": 1.2, "cfg_weight": 0.3, "temperature": 0.9},
    "accent":     {"label": "Other-language voice (no accent bleed)",
                   "exaggeration": 0.5, "cfg_weight": 0.0, "temperature": 0.8},
}

LANGUAGES = {
    "en": "English", "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

FORMATS = ("wav", "flac", "mp3", "ogg", "opus", "pcm")

# An original passage for recording a reference voice. Varied sounds, natural rhythm,
# about 15 seconds when read at a relaxed pace.
READING_SCRIPT = (
    "Every morning I walk past the old bakery on the corner, where the smell of fresh "
    "bread drifts into the street. Sometimes I stop and buy a warm loaf, and sometimes "
    "I just keep going, thinking about the day ahead. It's a small thing, but it always "
    "puts me in a good mood."
)
