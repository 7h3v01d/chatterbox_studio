"""Voice library. Every stored voice carries an explicit consent attestation."""
import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .config import VOICES_DIR

MIN_SECONDS = 3.0


class VoiceError(ValueError):
    pass


class VoiceNotFound(KeyError):
    pass


@dataclass
class Voice:
    id: str
    name: str
    path: str
    created: float
    duration: float
    consent: bool
    settings: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


def validate_settings(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        if k in config.SETTING_RANGES:
            lo, hi = config.SETTING_RANGES[k]
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise VoiceError(f"'{k}' must be a number.")
            if not lo <= v <= hi:
                raise VoiceError(f"'{k}' must be between {lo} and {hi}.")
            out[k] = v
        elif k == "language":
            if v not in config.LANGUAGES:
                raise VoiceError(f"Unsupported language '{v}'.")
            out[k] = v
        else:
            raise VoiceError(f"Unknown setting '{k}'.")
    return out


def effective_settings(voice, overrides: dict | None = None) -> dict:
    """Global defaults <- voice defaults <- explicit per-request overrides."""
    merged = dict(config.DEFAULT_SETTINGS)
    if voice is not None:
        merged.update(voice.settings)
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return merged


class VoiceStore:
    def __init__(self, root: Path = VOICES_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Voice]:
        voices = []
        for meta in sorted(self.root.glob("*/meta.json")):
            try:
                voices.append(Voice(**json.loads(meta.read_text())))
            except Exception:
                continue  # a corrupt entry is skipped, never half-loaded
        return sorted(voices, key=lambda v: v.created)

    def get(self, voice_id: str) -> Voice:
        for v in self.list():
            if v.id == voice_id:
                return v
        raise VoiceNotFound(voice_id)

    def add_audio(self, name: str, audio: np.ndarray, sr: int, consent: bool) -> Voice:
        if not consent:
            raise VoiceError("Consent is required: only clone your own voice, "
                             "or one you have explicit permission to clone.")
        name = (name or "").strip()
        if not name:
            raise VoiceError("Voice name is required.")
        audio = np.asarray(audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / float(sr)
        if duration < MIN_SECONDS:
            raise VoiceError(f"Sample too short ({duration:.1f}s). Need at least {MIN_SECONDS:.0f}s; ~10s is ideal.")
        peak = float(np.max(np.abs(audio))) or 1.0
        if peak < 0.01:
            raise VoiceError("Sample is silent or nearly silent. Check the microphone.")
        audio = audio / peak * 0.95

        vid = f"{re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-') or 'voice'}-{uuid.uuid4().hex[:6]}"
        d = self.root / vid
        d.mkdir(parents=True)
        wav_path = d / "reference.wav"
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")
        voice = Voice(vid, name, str(wav_path), time.time(), duration, True)
        (d / "meta.json").write_text(json.dumps(asdict(voice), indent=2))
        return voice

    def add_file(self, name: str, file_path: str, consent: bool) -> Voice:
        try:
            audio, sr = sf.read(file_path, dtype="float32")
        except Exception as e:
            raise VoiceError(f"Could not read audio file: {e}") from e
        return self.add_audio(name, audio, sr, consent)

    def update_settings(self, voice_id: str, settings: dict) -> Voice:
        v = self.get(voice_id)
        v.settings = {**v.settings, **validate_settings(settings)}
        (Path(v.path).parent / "meta.json").write_text(json.dumps(asdict(v), indent=2))
        return v

    def delete(self, voice_id: str):
        import shutil
        v = self.get(voice_id)
        shutil.rmtree(Path(v.path).parent)
