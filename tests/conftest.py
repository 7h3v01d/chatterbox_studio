# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Test fixtures. A FakeModel stands in for Chatterbox so the suite runs anywhere,
with no GPU and no model download."""
import os
import sys
import tempfile
from pathlib import Path

_DATA = tempfile.mkdtemp(prefix="cbx-test-")
os.environ["CBX_DATA"] = _DATA
os.environ.pop("CBX_TOKEN", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

SR = 24000


class FakeModel:
    """Mimics the Chatterbox surface: .sr, .conds, prepare_conditionals(), generate().
    Output length is proportional to text length and is padded with 0.3 s of silence on
    both sides, so trimming and pause handling are observable."""

    def __init__(self, kind, has_default=True, fail_on=None):
        self.kind = kind
        self.sr = SR
        self.conds = {"voice": "default"} if has_default else None
        self.calls = []
        self.prepared = []
        self.fail_on = fail_on

    def prepare_conditionals(self, path, exaggeration=0.5):
        self.prepared.append(path)
        self.conds = {"voice": path}

    def generate(self, text, exaggeration=0.5, cfg_weight=0.5, temperature=0.8, **kw):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("boom")
        self.calls.append({"text": text, "exaggeration": exaggeration,
                           "cfg_weight": cfg_weight, "temperature": temperature, **kw})
        n = int(SR * 0.01 * len(text))
        t = np.arange(n) / SR
        tone = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        pad = np.zeros(int(SR * 0.3), dtype=np.float32)
        return np.concatenate([pad, tone, pad])[None, :]


class TurboFake(FakeModel):
    """Mirrors chatterbox-tts 0.1.7 ChatterboxTurboTTS.generate(): accepts exaggeration and
    cfg_weight (defaults 0.0) but ignores them, warning when they are > 0. No language_id."""
    warnings = 0

    def generate(self, text, repetition_penalty=1.2, min_p=0.0, top_p=0.95,
                 audio_prompt_path=None, exaggeration=0.0, cfg_weight=0.0,
                 temperature=0.8, top_k=1000, norm_loudness=True):
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            TurboFake.warnings += 1
        return super().generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight,
                                temperature=temperature)


def pytest_sessionfinish(session, exitstatus):
    """CBX_STRICT=1 (set by test.bat): a skipped test fails the run, so a machine missing
    PyQt6 or the MCP SDK can't report green while silently skipping those suites."""
    if os.environ.get("CBX_STRICT") == "1":
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr and tr.stats.get("skipped"):
            tr.write_line(f"CBX_STRICT: {len(tr.stats['skipped'])} skipped — install all "
                          "dependencies (pip install -e .[dev]) before releasing.", red=True)
            session.exitstatus = 1


@pytest.fixture
def factory():
    made = []

    def make(kind, device):
        m = TurboFake(kind) if kind in ("turbo", "nano") else FakeModel(
            kind, has_default=(kind != "multilingual"))
        made.append(m)
        return m
    make.made = made
    return make


@pytest.fixture
def engine(factory):
    from chatterbox_studio.engine import TTSEngine
    return TTSEngine("cpu", model_factory=factory)


@pytest.fixture
def store(tmp_path):
    from chatterbox_studio.voices import VoiceStore
    return VoiceStore(tmp_path / "voices")


def speech_like(seconds=10.0, sr=SR, noise=0.002, amp=0.4):
    """Syllable-modulated tone with a little noise: passes the quality check."""
    t = np.arange(int(seconds * sr)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)
    rng = np.random.default_rng(0)
    return (amp * env * np.sin(2 * np.pi * 180 * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def voice(store):
    return store.add_audio("Leon", speech_like(), SR, consent=True)
