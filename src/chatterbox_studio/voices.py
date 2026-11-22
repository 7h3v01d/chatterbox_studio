# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Voice library. Every stored voice carries an explicit consent attestation.

Layout (unchanged from 0.4, so existing voices keep working):
    <CBX_DATA>/voices/<voice-id>/reference.wav
    <CBX_DATA>/voices/<voice-id>/meta.json
"""
import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from . import audio, config

MIN_SECONDS = 3.0
MAX_SECONDS = 120.0
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


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
    quality: dict = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Voice":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def public(self) -> dict:
        d = asdict(self)
        d.pop("path")
        return d


def validate_settings(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        if v is None:
            continue
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
        elif k == "model":
            if v not in config.MODELS:
                raise VoiceError(f"Unknown model '{v}'.")
            out[k] = v
        else:
            raise VoiceError(f"Unknown setting '{k}'.")
    return out


def effective_settings(voice: Optional[Voice], overrides: Optional[dict] = None) -> dict:
    """Global defaults <- voice defaults <- explicit per-request overrides."""
    merged = dict(config.DEFAULT_SETTINGS)
    if voice is not None:
        merged.update(voice.settings)
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return merged


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"


class VoiceStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or config.VOICES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- reading -------------------------------------------------------------
    def list(self) -> list[Voice]:
        voices = []
        for meta in self.root.glob("*/meta.json"):
            try:
                v = Voice.from_dict(json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                continue                    # a corrupt entry is skipped, never half-loaded
            # the folder is authoritative for where the audio lives (survives moving CBX_DATA)
            v.path = str(meta.parent / "reference.wav")
            if Path(v.path).is_file():
                voices.append(v)
        return sorted(voices, key=lambda v: v.created)

    def get(self, voice_id: str) -> Voice:
        if not voice_id or not _ID_RE.match(voice_id):
            raise VoiceNotFound(voice_id)
        for v in self.list():
            if v.id == voice_id:
                return v
        raise VoiceNotFound(voice_id)

    def resolve(self, ref: str) -> Voice:
        """Find a voice by id, or by name (case-insensitive) when the name is unique."""
        try:
            return self.get(ref)
        except VoiceNotFound:
            pass
        matches = [v for v in self.list() if v.name.strip().lower() == (ref or "").strip().lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise VoiceError(f"More than one voice is named '{ref}'. Use its id instead.")
        raise VoiceNotFound(ref)

    # ---- writing -------------------------------------------------------------
    def add_audio(self, name: str, samples, sr: int, consent: bool,
                  trim: bool = True) -> Voice:
        if not consent:
            raise VoiceError("Consent is required: only clone your own voice, "
                             "or one you have explicit permission to clone.")
        name = (name or "").strip()
        if not name:
            raise VoiceError("Give the voice a name.")
        if len(name) > 80:
            raise VoiceError("Voice name is too long (80 characters max).")
        a = audio.to_mono(np.asarray(samples, dtype=np.float32))
        if trim:
            a = audio.trim_silence(a, sr, threshold_db=-40.0, pad=0.15)
        duration = len(a) / float(sr)
        if duration < MIN_SECONDS:
            raise VoiceError(f"Sample too short ({duration:.1f}s of sound). Need at least "
                             f"{MIN_SECONDS:.0f}s; about 10s is ideal.")
        peak = float(np.max(np.abs(a)))
        if peak < 0.01:
            raise VoiceError("The sample is silent or nearly silent. Check the microphone.")
        if duration > MAX_SECONDS:
            a = a[: int(MAX_SECONDS * sr)]
            duration = MAX_SECONDS
        report = audio.analyze_reference(a, sr)
        a = audio.normalize_peak(a, -1.0)

        vid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        d = self.root / vid
        d.mkdir(parents=True)
        wav_path = d / "reference.wav"
        sf.write(str(wav_path), a, sr, subtype="PCM_16")
        voice = Voice(vid, name, str(wav_path), time.time(), round(duration, 2), True,
                      quality=asdict(report))
        self._save(voice)
        return voice

    def add_bytes(self, name: str, data: bytes, consent: bool) -> Voice:
        try:
            a, sr = audio.read_any(data)
        except audio.AudioError as e:
            raise VoiceError(str(e)) from e
        return self.add_audio(name, a, sr, consent)

    def add_file(self, name: str, file_path: str, consent: bool) -> Voice:
        try:
            a, sr = audio.read_any(str(file_path))
        except audio.AudioError as e:
            raise VoiceError(str(e)) from e
        return self.add_audio(name or Path(file_path).stem, a, sr, consent)

    def update_settings(self, voice_id: str, settings: dict) -> Voice:
        v = self.get(voice_id)
        v.settings = {**v.settings, **validate_settings(settings)}
        self._save(v)
        return v

    def rename(self, voice_id: str, name: str) -> Voice:
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise VoiceError("Name must be 1–80 characters.")
        v = self.get(voice_id)
        v.name = name
        self._save(v)
        return v

    def delete(self, voice_id: str):
        v = self.get(voice_id)
        shutil.rmtree(Path(v.path).parent)

    def _save(self, v: Voice):
        meta = Path(v.path).parent / "meta.json"
        tmp = meta.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(v), indent=2), encoding="utf-8")
        tmp.replace(meta)                      # atomic: never a half-written meta.json
