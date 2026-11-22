# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Audio utilities: numpy float32 mono throughout, no torch dependency.

encode()            -> wav / flac / mp3 / ogg / opus / pcm bytes
read_any()          -> decode an upload (soundfile, falling back to librosa/ffmpeg)
trim_silence()      -> strip leading/trailing silence
fade()              -> short fades so joins never click
change_speed()      -> pitch-preserving time stretch (librosa)
normalize_peak()    -> peak normalisation
analyze_reference() -> plain-language quality check for a voice sample
srt()               -> subtitles from timed cues
"""
import io
import struct
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

MIME = {"wav": "audio/wav", "flac": "audio/flac", "mp3": "audio/mpeg",
        "ogg": "audio/ogg", "opus": "audio/ogg", "pcm": "audio/L16"}


class AudioError(ValueError):
    pass


def to_mono(audio) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        # (frames, channels) from soundfile; (1, N) from torch-style tensors
        a = a.mean(axis=1) if a.shape[0] > a.shape[1] else a.mean(axis=0)
    return np.ascontiguousarray(a.reshape(-1))


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, int(round(seconds * sr))), dtype=np.float32)


def _frame_rms(a: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(a) // n * n
    if usable == 0:
        return np.array([float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0])
    frames = a[:usable].reshape(-1, n)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def trim_silence(a: np.ndarray, sr: int, threshold_db: float = -45.0,
                 pad: float = 0.05) -> np.ndarray:
    """Remove leading/trailing audio quieter than threshold (relative to peak)."""
    a = to_mono(a)
    if not len(a):
        return a
    peak = float(np.max(np.abs(a)))
    if peak <= 0:
        return a[:0]
    thr = peak * 10 ** (threshold_db / 20.0)
    loud = np.flatnonzero(np.abs(a) > thr)
    if not len(loud):
        return a[:0]
    p = int(pad * sr)
    return a[max(0, loud[0] - p): min(len(a), loud[-1] + 1 + p)]


def fade(a: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    a = np.array(a, dtype=np.float32, copy=True)
    n = min(len(a) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        a[:n] *= ramp
        a[-n:] *= ramp[::-1]
    return a


def normalize_peak(a: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak <= 0:
        return a
    return (a * (10 ** (target_db / 20.0) / peak)).astype(np.float32)


def change_speed(a: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Pitch-preserving tempo change. speed 1.25 = 25 % faster."""
    if abs(speed - 1.0) < 1e-3 or not len(a):
        return a
    try:
        import librosa
    except ImportError as e:
        raise AudioError("Speed control needs librosa (installed with chatterbox-tts): "
                         "pip install librosa") from e
    return librosa.effects.time_stretch(a.astype(np.float32), rate=float(speed)).astype(np.float32)


def encode(a: np.ndarray, sr: int, fmt: str = "wav") -> bytes:
    fmt = (fmt or "wav").lower()
    a = np.clip(to_mono(a), -1.0, 1.0)
    if fmt == "pcm":
        return (a * 32767).astype("<i2").tobytes()
    spec = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "mp3": ("MP3", "MPEG_LAYER_III"),
            "ogg": ("OGG", "VORBIS"), "opus": ("OGG", "OPUS")}
    if fmt not in spec:
        raise AudioError(f"Unsupported format '{fmt}'. Use one of: wav, flac, mp3, ogg, opus, pcm.")
    container, subtype = spec[fmt]
    if fmt == "opus" and sr not in (8000, 12000, 16000, 24000, 48000):
        a, sr = resample(a, sr, 48000), 48000
    buf = io.BytesIO()
    try:
        sf.write(buf, a, sr, format=container, subtype=subtype)
    except Exception as e:
        raise AudioError(f"This system's libsndfile can't write {fmt}: {e}. "
                         "Use wav or flac, or upgrade soundfile.") from e
    return buf.getvalue()


def resample(a: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or not len(a):
        return a
    n = int(round(len(a) * target / sr))
    x_old = np.linspace(0.0, 1.0, len(a), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, a).astype(np.float32)


def stream_wav_header(sr: int) -> bytes:
    """WAV header with unknown length (0xFFFFFFFF), which streaming players accept."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFFF))


def pcm16(a: np.ndarray) -> bytes:
    return (np.clip(to_mono(a), -1.0, 1.0) * 32767).astype("<i2").tobytes()


def read_any(source) -> tuple[np.ndarray, int]:
    """Decode a path, bytes or file-like. Tries libsndfile first (wav/flac/ogg/mp3), then
    librosa, which can use ffmpeg for m4a/aac/webm. Works on bytes in memory, so there is
    no temp file (0.4's held-open NamedTemporaryFile cannot be re-opened on Windows)."""
    data = source
    if isinstance(source, (bytes, bytearray)):
        data = io.BytesIO(source)
    try:
        a, sr = sf.read(data, dtype="float32", always_2d=False)
        return to_mono(a), int(sr)
    except Exception as first:
        try:
            import librosa
            if hasattr(data, "seek"):
                data.seek(0)
            a, sr = librosa.load(data, sr=None, mono=True)
            return a.astype(np.float32), int(sr)
        except Exception:
            raise AudioError(f"Could not read audio: {first}. Try WAV, FLAC, OGG or MP3.") from first


# ---- reference quality ---------------------------------------------------------
@dataclass
class QualityReport:
    duration: float
    peak_db: float
    clipped_pct: float
    level_db: float          # loudness of speech frames
    noise_db: float          # estimated background noise
    snr_db: float
    speech_pct: float
    verdict: str             # "great" | "ok" | "poor"
    tips: list = field(default_factory=list)

    def summary(self) -> str:
        head = {"great": "Great sample", "ok": "Usable sample", "poor": "Weak sample"}[self.verdict]
        return (f"{head}: {self.duration:.1f}s, signal-to-noise {self.snr_db:.0f} dB, "
                f"{self.speech_pct:.0f}% speech")


def analyze_reference(a: np.ndarray, sr: int) -> QualityReport:
    a = to_mono(a)
    dur = len(a) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    clipped = float(np.mean(np.abs(a) >= 0.999)) * 100 if len(a) else 0.0
    rms = _frame_rms(a, sr)
    noise = float(np.percentile(rms, 10)) if len(rms) else 0.0
    loud = float(np.percentile(rms, 90)) if len(rms) else 0.0
    gate = max(noise * 3.0, loud * 0.1, 1e-4)
    speech = rms[rms > gate]
    level = float(np.sqrt(np.mean(speech ** 2))) if len(speech) else 0.0
    snr = db(level) - db(noise) if level else 0.0
    speech_pct = len(speech) / max(1, len(rms)) * 100

    tips, score = [], 0
    if dur < 5:
        tips.append("Too short. Record at least 8–10 seconds of continuous speech.")
        score += 2
    elif dur < 8:
        tips.append("A little short. About 10 seconds clones best.")
        score += 1
    elif dur > 40:
        tips.append("Longer than needed. The model only listens to the first ~10 seconds, "
                    "so make sure those are your best.")
    if clipped > 0.1:
        tips.append("Audio is clipping (distorted). Move back from the mic or lower input gain.")
        score += 2 if clipped > 1 else 1
    if peak < 0.05:
        tips.append("Very quiet. Move closer to the mic or raise the input level.")
        score += 1
    if snr < 15:
        tips.append("Lots of background noise. Find a quieter room, turn off fans, and "
                    "keep the mic away from your computer.")
        score += 2
    elif snr < 25:
        tips.append("Some background noise. A quieter room will sound cleaner.")
        score += 1
    if speech_pct < 50 and dur >= 5:
        tips.append("Lots of silence. Speak continuously; pauses are trimmed automatically.")
        score += 1
    verdict = "great" if score == 0 else ("ok" if score <= 2 else "poor")
    if not tips:
        tips.append("Clean and clear. This should clone well.")
    return QualityReport(round(dur, 2), round(db(peak), 1), round(clipped, 2), round(db(level), 1),
                         round(db(noise), 1), round(snr, 1), round(speech_pct, 1), verdict, tips)


# ---- subtitles ------------------------------------------------------------------
def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def srt(cues) -> str:
    """cues: iterable of (start_seconds, end_seconds, text)."""
    blocks = []
    for i, (start, end, text) in enumerate(cues, 1):
        blocks.append(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text.strip()}\n")
    return "\n".join(blocks)
