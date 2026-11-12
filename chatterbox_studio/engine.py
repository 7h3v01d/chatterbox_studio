"""Chatterbox TTS wrapper.

- English uses ChatterboxTTS; any other language uses ChatterboxMultilingualTTS.
  Only one model is resident at a time (switching frees the other).
- synthesize_stream() yields audio sentence-group by sentence-group, so playback can
  begin before the full text is done. Chatterbox itself isn't token-streaming.
- Fails loudly: every failure is an EngineError with a readable message.
"""
import copy
import gc
import re
import threading
from typing import Iterator, Optional, Tuple

from . import config


class EngineError(RuntimeError):
    pass


def _chunks(text: str, limit: int = 300):
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    cur = ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            yield cur
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        yield cur


class TTSEngine:
    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._model = None
        self._kind: Optional[str] = None       # "en" | "mtl"
        self._default_conds = None
        self._voice_loaded: Optional[str] = None
        self._lock = threading.Lock()

    # ---- state
    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def languages(self) -> dict:
        return dict(config.LANGUAGES)

    # ---- model management
    def load(self, language: str = "en"):
        with self._lock:
            self._ensure("en" if language == "en" else "mtl")

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
            if kind == "en":
                from chatterbox.tts import ChatterboxTTS as Model
            else:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Model
        except ImportError as e:
            raise EngineError(
                "This model isn't available in the installed chatterbox-tts. "
                "Upgrade it:  pip install -U chatterbox-tts") from e
        try:
            self._model = Model.from_pretrained(self.device)
        except Exception as e:
            raise EngineError(f"Failed to load Chatterbox ({kind}) on {self.device}: {e}") from e
        self._kind = kind
        conds = getattr(self._model, "conds", None)
        self._default_conds = copy.deepcopy(conds) if conds is not None else None

    def _apply_voice(self, voice_path: Optional[str], exaggeration: float):
        if voice_path == self._voice_loaded:
            return
        if voice_path:
            self._model.prepare_conditionals(voice_path, exaggeration=exaggeration)
        else:
            if self._default_conds is None:
                raise EngineError("This model has no built-in voice. Select a cloned voice.")
            self._model.conds = copy.deepcopy(self._default_conds)
        self._voice_loaded = voice_path

    # ---- synthesis
    def synthesize_stream(self, text: str, voice_path: Optional[str] = None,
                          exaggeration: float = 0.5, cfg_weight: float = 0.5,
                          temperature: float = 0.8, language: str = "en",
                          chunk_limit: int = 200) -> Iterator[Tuple["object", int]]:
        """Yield (tensor[1, N], sample_rate) per chunk. Holds the engine lock until exhausted
        or closed, so abandon it by closing the generator."""
        if not text or not text.strip():
            raise EngineError("Text is empty.")
        language = (language or "en").lower()
        if language not in config.LANGUAGES:
            raise EngineError(f"Unsupported language '{language}'.")
        kind = "en" if language == "en" else "mtl"
        with self._lock:
            self._ensure(kind)
            try:
                self._apply_voice(voice_path, exaggeration)
                for piece in _chunks(text, chunk_limit):
                    kw = dict(exaggeration=exaggeration, cfg_weight=cfg_weight,
                              temperature=temperature)
                    if kind == "mtl":
                        kw["language_id"] = language
                    yield self._model.generate(piece, **kw), self._model.sr
            except EngineError:
                raise
            except Exception as e:
                raise EngineError(f"Synthesis failed: {e}") from e

    def synthesize(self, text: str, voice_path: Optional[str] = None,
                   exaggeration: float = 0.5, cfg_weight: float = 0.5,
                   temperature: float = 0.8, language: str = "en"):
        import torch
        parts, sr = [], None
        for wav, sr in self.synthesize_stream(text, voice_path, exaggeration, cfg_weight,
                                              temperature, language, chunk_limit=300):
            parts.append(wav)
        if not parts:
            raise EngineError("No audio was produced.")
        return torch.cat(parts, dim=-1), sr
