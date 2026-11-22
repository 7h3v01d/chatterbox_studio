# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Chatterbox model wrapper.

* Four model families: standard (English, CFG + exaggeration), turbo and nano (English,
  fast, paralinguistic tags), multilingual (23 languages). "auto" picks standard for
  English and multilingual otherwise. One model is resident at a time.
* Text goes through text.plan(): chunks never exceed the limit, pauses are real silence.
* Each chunk is edge-trimmed, optionally time-stretched, and faded, so joins are clean
  and the rhythm is set by the pause settings instead of whatever silence the model left.
* render() yields pieces as they are produced (streaming); synthesize() returns one
  array plus subtitle cues.
* Only keyword arguments the installed model's generate() accepts are passed, so the
  wrapper works across chatterbox-tts releases.
* Fails loudly: every failure is an EngineError with a readable message.
"""
import copy
import gc
import inspect
import random
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np

from . import audio, config, text as textmod


class EngineError(RuntimeError):
    pass


class Cancelled(EngineError):
    pass


@dataclass
class Piece:
    audio: np.ndarray
    sr: int
    kind: str            # "speak" | "pause"
    text: str = ""


@dataclass
class Result:
    audio: np.ndarray
    sr: int
    cues: list = field(default_factory=list)     # (start, end, text)

    @property
    def seconds(self) -> float:
        return len(self.audio) / float(self.sr) if self.sr else 0.0


def resolve_kind(model: str, language: str) -> str:
    model = (model or "auto").lower()
    language = (language or "en").lower()
    if model not in config.MODELS:
        raise EngineError(f"Unknown model '{model}'. Choose: {', '.join(config.MODELS)}.")
    if model == "auto":
        return "standard" if language == "en" else "multilingual"
    if model in config.ENGLISH_ONLY and language != "en":
        raise EngineError(f"The {model} model is English-only. Pick Multilingual or Auto "
                          f"for {config.LANGUAGES.get(language, language)}.")
    return model


def default_factory(kind: str, device: str):
    """Load a real Chatterbox model. Imported lazily so tests and the CLI's doctor
    command work without the package installed."""
    try:
        if kind == "standard":
            from chatterbox.tts import ChatterboxTTS
            return ChatterboxTTS.from_pretrained(device=device)
        if kind in ("turbo", "nano"):
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            if kind == "nano":
                return ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
            return ChatterboxTurboTTS.from_pretrained(device=device)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        try:
            return ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        except TypeError:   # older release without the V3 checkpoint switch
            return ChatterboxMultilingualTTS.from_pretrained(device=device)
    except ImportError as e:
        raise EngineError(f"The {kind} model isn't in your installed chatterbox-tts. "
                          "Upgrade it:  pip install -U chatterbox-tts") from e
    except TypeError as e:
        raise EngineError(f"Your chatterbox-tts doesn't support the {kind} model ({e}). "
                          "Upgrade it:  pip install -U chatterbox-tts") from e


def _accepted(fn, kwargs: dict) -> dict:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _to_numpy(wav) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().float().numpy()
    return audio.to_mono(wav)


class TTSEngine:
    def __init__(self, device: Optional[str] = None,
                 model_factory: Callable[[str, str], object] = default_factory):
        self._device = device
        self._factory = model_factory
        self._model = None
        self._kind: Optional[str] = None
        self._default_conds = None
        self._voice_loaded: Optional[tuple] = None
        self._lock = threading.Lock()

    # ---- state ---------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def kind(self) -> Optional[str]:
        return self._kind

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def status(self) -> dict:
        return {"model_loaded": self.loaded, "model": self._kind, "device": self.device,
                "busy": self.busy}

    # ---- model management ----------------------------------------------------
    def load(self, model: str = "auto", language: str = "en"):
        with self._lock:
            self._ensure(resolve_kind(model, language))

    def unload(self):
        with self._lock:
            self._unload()

    def _unload(self):
        self._model = None
        self._kind = None
        self._default_conds = None
        self._voice_loaded = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _ensure(self, kind: str):
        if self._model is not None and self._kind == kind:
            return
        self._unload()
        try:
            self._model = self._factory(kind, self.device)
        except EngineError:
            raise
        except Exception as e:
            raise EngineError(f"Failed to load the {kind} model on {self.device}: {e}") from e
        self._kind = kind
        conds = getattr(self._model, "conds", None)
        self._default_conds = copy.deepcopy(conds) if conds is not None else None

    def _apply_voice(self, voice_path: Optional[str], exaggeration: float):
        key = (self._kind, voice_path)
        if key == self._voice_loaded:
            return
        if voice_path:
            fn = self._model.prepare_conditionals
            fn(voice_path, **_accepted(fn, {"exaggeration": exaggeration}))
        else:
            if self._default_conds is None:
                raise EngineError("This model has no built-in voice. Pick one of your voices.")
            self._model.conds = copy.deepcopy(self._default_conds)
        self._voice_loaded = key

    # ---- synthesis -----------------------------------------------------------
    def render(self, text: str, voice_path: Optional[str] = None,
               settings: Optional[dict] = None, *, chunk_limit: int = 250,
               seed: Optional[int] = None, cancel: Optional[threading.Event] = None,
               on_progress: Optional[Callable[[int, int], None]] = None,
               markdown: bool = True) -> Iterator[Piece]:
        """Yield Pieces in order. Holds the engine lock until exhausted or closed, so
        abandon it by calling .close() on the generator."""
        s = {**config.DEFAULT_SETTINGS, **(settings or {})}
        if not text or not text.strip():
            raise EngineError("There's no text to speak.")
        language = (s["language"] or "en").lower()
        if language not in config.LANGUAGES:
            raise EngineError(f"Unsupported language '{language}'.")
        kind = resolve_kind(s["model"], language)
        segments = textmod.plan(text, chunk_limit, s["sentence_pause"], s["paragraph_pause"],
                                markdown=markdown)
        total = textmod.speak_count(segments)
        if not total:
            raise EngineError("There's nothing speakable in that text.")

        with self._lock:
            self._ensure(kind)
            if seed is not None:
                self._seed(int(seed))
            sr = int(self._model.sr)
            done = 0
            if on_progress:
                on_progress(0, total)
            try:
                self._apply_voice(voice_path, s["exaggeration"])
                gen = self._model.generate
                for seg in segments:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("Cancelled.")
                    if seg.kind == "pause":
                        yield Piece(audio.silence(seg.seconds, sr), sr, "pause")
                        continue
                    kw = {"exaggeration": s["exaggeration"], "cfg_weight": s["cfg_weight"],
                          "temperature": s["temperature"]}
                    if kind == "multilingual":
                        kw["language_id"] = language
                    wav = _to_numpy(gen(seg.text, **_accepted(gen, kw)))
                    wav = audio.trim_silence(wav, sr, threshold_db=-50.0, pad=0.04)
                    wav = audio.change_speed(wav, sr, float(s["speed"]))
                    wav = audio.fade(wav, sr)
                    done += 1
                    if on_progress:
                        on_progress(done, total)
                    yield Piece(wav, sr, "speak", seg.text)
            except EngineError:
                raise
            except audio.AudioError as e:
                raise EngineError(str(e)) from e
            except Exception as e:
                raise EngineError(f"Synthesis failed: {e}") from e

    def synthesize(self, text: str, voice_path: Optional[str] = None,
                   settings: Optional[dict] = None, *, normalize: bool = True,
                   **kw) -> Result:
        parts, cues, t, sr = [], [], 0.0, None
        it = self.render(text, voice_path, settings, **kw)
        try:
            for p in it:
                sr = p.sr
                dur = len(p.audio) / float(p.sr)
                if p.kind == "speak":
                    cues.append((t, t + dur, p.text))
                parts.append(p.audio)
                t += dur
        finally:
            it.close()
        if not parts or sr is None:
            raise EngineError("No audio was produced.")
        out = np.concatenate(parts).astype(np.float32)
        if normalize:
            out = audio.normalize_peak(out, -1.0)
        return Result(out, sr, cues)

    @staticmethod
    def _seed(seed: int):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
