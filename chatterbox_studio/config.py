import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CBX_DATA", Path.home() / ".chatterbox_studio"))
VOICES_DIR = DATA_DIR / "voices"
OUT_DIR = DATA_DIR / "output"
API_HOST = os.environ.get("CBX_HOST", "127.0.0.1")   # localhost only by default
API_PORT = int(os.environ.get("CBX_PORT", "8765"))
API_TOKEN = os.environ.get("CBX_TOKEN", "")          # if set, Bearer auth is required

for _d in (VOICES_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_SETTINGS = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "language": "en"}
SETTING_RANGES = {"exaggeration": (0.25, 2.0), "cfg_weight": (0.0, 1.0), "temperature": (0.05, 5.0)}
LANGUAGES = {
    "en": "English", "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
    "sv": "Swedish", "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}
