# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)

import pytest
import soundfile as sf

from chatterbox_studio import cli, engine as engmod

from conftest import SR, FakeModel, speech_like


@pytest.fixture(autouse=True)
def fake_models(monkeypatch):
    monkeypatch.setattr(engmod, "default_factory", lambda k, d: FakeModel(k))
    # TTSEngine binds the factory as a default argument at import time
    orig = engmod.TTSEngine.__init__

    def init(self, device=None, model_factory=None):
        orig(self, device or "cpu", lambda k, d: FakeModel(k))
    monkeypatch.setattr(engmod.TTSEngine, "__init__", init)


def test_clone_then_say_with_srt(tmp_path, capsys):
    wav = tmp_path / "me.wav"
    sf.write(str(wav), speech_like(), SR)
    with pytest.raises(SystemExit, match="--consent"):
        cli.main(["clone", str(wav), "--name", "CliVoice"])
    cli.main(["clone", str(wav), "--name", "CliVoice", "--consent"])
    md = tmp_path / "script.md"
    md.write_text("# Intro\n\nHello there. [pause 1s] Second line.", encoding="utf-8")
    out = tmp_path / "out.mp3"
    cli.main(["say", "--file", str(md), "--voice", "clivoice", "-o", str(out), "--srt",
              "--preset", "narrator"])
    assert out.read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3") or out.stat().st_size > 100
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "Intro" in srt and "Second line." in srt
    printed = capsys.readouterr().out
    assert "Saved" in printed


def test_unknown_voice_lists_available(capsys):
    with pytest.raises(SystemExit, match="not found"):
        cli.main(["say", "hi", "--voice", "nobody-here"])


def test_parser_help_runs():
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
