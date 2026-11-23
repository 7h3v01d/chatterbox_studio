# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
import io
import json
import threading

import numpy as np
import pytest
import soundfile as sf

from chatterbox_studio import audio, text as T
from chatterbox_studio.engine import Cancelled, EngineError, resolve_kind
from chatterbox_studio.voices import VoiceError, VoiceNotFound, effective_settings

from conftest import SR, speech_like


# ---- text planning --------------------------------------------------------------
class TestChunking:
    # REGRESSION (0.4): chunks ignored the limit whenever one sentence was longer than it.
    def test_unpunctuated_text_respects_limit(self):
        chunks = T.chunk(("word " * 400).strip(), 200)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_long_sentence_splits_at_clauses_first(self):
        s = "First one. " + "long clause, " * 60 + "end."
        chunks = T.chunk(s, 200)
        assert all(len(c) <= 200 for c in chunks)
        assert all(c.endswith((",", ".")) for c in chunks)   # split on punctuation, not mid-clause

    def test_giant_token_is_still_bounded(self):
        assert all(len(c) <= 50 for c in T.chunk("x" * 500, 50))

    def test_nothing_lost(self):
        src = "Hello there. " + "alpha beta, gamma delta; " * 40 + "Done!"
        assert " ".join(T.chunk(src, 120)).split() == src.split()

    def test_short_sentences_are_packed(self):
        assert T.chunk("A. B. C.", 250) == ["A. B. C."]


class TestPlan:
    def test_pause_markup(self):
        p = T.plan("Hi. [pause 2s] There. [pause 500ms] Again [pause] end", 250, 0.2, 0.7)
        kinds = [(s.kind, s.text or s.seconds) for s in p]
        assert kinds == [("speak", "Hi."), ("pause", 2.0), ("speak", "There."),
                         ("pause", 0.5), ("speak", "Again"), ("pause", T.DEFAULT_PAUSE),
                         ("speak", "end")]

    def test_paragraph_pause_and_no_leading_or_trailing_silence(self):
        p = T.plan("[pause 3s]One.\n\nTwo. [pause]", 250, 0.2, 0.9)
        assert p[0].kind == "speak" and p[-1].kind == "speak"
        assert p[1] == T.Segment("pause", seconds=0.9)

    def test_sentence_pause_between_chunks(self):
        p = T.plan("a " * 200, 100, 0.3, 0.7)
        assert [s.kind for s in p][:3] == ["speak", "pause", "speak"]
        assert p[1].seconds == 0.3

    def test_markdown_is_cleaned(self):
        md = "# Title\n\n- **Bold** item with [a link](http://x.y)\n\n```py\ncode()\n```\nEnd"
        spoken = " ".join(s.text for s in T.plan(md) if s.kind == "speak")
        assert spoken == "Title Bold item with a link End"

    def test_turbo_tags_survive(self):
        assert "[laugh]" in T.plan("That's funny [laugh] really.")[0].text

    def test_pause_capped(self):
        assert T.plan("a [pause 999s] b")[1].seconds == T.MAX_PAUSE


# ---- audio ------------------------------------------------------------------------
class TestAudio:
    def test_trim(self):
        a = np.concatenate([np.zeros(SR), 0.5 * np.ones(SR // 2), np.zeros(SR)]).astype(np.float32)
        out = audio.trim_silence(a, SR, pad=0.0)
        assert abs(len(out) - SR // 2) < 10

    @pytest.mark.parametrize("fmt", ["wav", "flac", "mp3", "ogg"])
    def test_encode_roundtrip(self, fmt):
        a = speech_like(1.0)
        data = audio.encode(a, SR, fmt)
        back, sr = audio.read_any(data)
        assert sr == SR and abs(len(back) - len(a)) < SR * 0.1

    def test_pcm(self):
        assert len(audio.encode(np.zeros(100, np.float32), SR, "pcm")) == 200

    def test_bad_format(self):
        with pytest.raises(audio.AudioError):
            audio.encode(np.zeros(10, np.float32), SR, "aiff")

    def test_speed_changes_length(self):
        pytest.importorskip("librosa")
        a = speech_like(2.0)
        assert len(audio.change_speed(a, SR, 2.0)) == pytest.approx(len(a) / 2, rel=0.05)

    def test_quality_good_vs_bad(self):
        good = audio.analyze_reference(speech_like(10), SR)
        assert good.verdict == "great", good.tips
        noisy = audio.analyze_reference(speech_like(3, noise=0.3, amp=0.3), SR)
        assert noisy.verdict == "poor" and len(noisy.tips) >= 2
        clipped = audio.analyze_reference(np.clip(speech_like(10, amp=3.0), -1, 1), SR)
        assert any("clipping" in t for t in clipped.tips)

    def test_srt(self):
        s = audio.srt([(0.0, 1.5, "Hello"), (61.25, 3725.0, "World")])
        assert "00:00:00,000 --> 00:00:01,500\nHello" in s
        assert "00:01:01,250 --> 01:02:05,000\nWorld" in s


# ---- engine -----------------------------------------------------------------------
class TestEngine:
    def test_resolve_kind(self):
        assert resolve_kind("auto", "en") == "standard"
        assert resolve_kind("auto", "fr") == "multilingual"
        with pytest.raises(EngineError, match="English-only"):
            resolve_kind("turbo", "fr")

    def test_synthesize_with_cues_and_pauses(self, engine):
        r = engine.synthesize("One sentence here. [pause 1s] Another one.", None,
                              {"sentence_pause": 0.0})
        assert [c[2] for c in r.cues] == ["One sentence here.", "Another one."]
        gap = r.cues[1][0] - r.cues[0][1]
        assert gap == pytest.approx(1.0, abs=0.01)
        assert np.max(np.abs(r.audio)) == pytest.approx(10 ** (-1 / 20), rel=1e-3)

    def test_model_silence_is_trimmed(self, engine):
        r = engine.synthesize("abcdefghij", None)          # fake pads 0.3 s each side
        assert r.seconds < 0.1 + 0.3                        # 0.1 s tone + small pads

    def test_chunks_never_exceed_limit_at_model(self, engine, factory):
        engine.synthesize("word " * 300, None, chunk_limit=150)
        assert all(len(c["text"]) <= 150 for c in factory.made[0].calls)

    def test_multilingual_passes_language_and_needs_voice(self, engine, factory, voice):
        with pytest.raises(EngineError, match="no built-in voice"):
            engine.synthesize("Bonjour.", None, {"language": "fr"})
        engine.synthesize("Bonjour.", voice.path, {"language": "fr"})
        assert factory.made[-1].calls[-1]["language_id"] == "fr"

    def test_turbo_gets_only_supported_kwargs(self, engine, factory):
        engine.synthesize("Hello [laugh] there.", None, {"model": "turbo"})
        call = factory.made[0].calls[0]
        assert "language_id" not in call

    def test_voice_cache_and_model_switch(self, engine, factory, voice):
        engine.synthesize("a.", voice.path)
        engine.synthesize("b.", voice.path)
        assert factory.made[0].prepared == [voice.path]   # prepared once
        engine.synthesize("c.", voice.path, {"model": "turbo"})
        assert engine.kind == "turbo" and factory.made[1].prepared == [voice.path]

    def test_cancel(self, engine):
        ev = threading.Event()
        seen = []

        def prog(d, t):
            seen.append(d)
            if d == 2:
                ev.set()
        with pytest.raises(Cancelled):
            engine.synthesize("One. " * 20, None, chunk_limit=5, cancel=ev, on_progress=prog)
        assert max(seen) == 2

    def test_lock_released_after_close(self, engine):
        it = engine.render("One. Two. Three.", None, chunk_limit=5)
        next(it)
        assert engine.busy
        it.close()
        assert not engine.busy

    def test_failure_is_readable(self, factory):
        from chatterbox_studio.engine import TTSEngine
        from conftest import FakeModel
        e = TTSEngine("cpu", model_factory=lambda k, d: FakeModel(k, fail_on="bad"))
        with pytest.raises(EngineError, match="Synthesis failed: boom"):
            e.synthesize("good. bad.", None, chunk_limit=5)
        assert not e.busy

    def test_empty_text(self, engine):
        with pytest.raises(EngineError):
            engine.synthesize("   ", None)
        with pytest.raises(EngineError, match="nothing speakable"):
            engine.synthesize("[pause 2s]", None)


# ---- voices -----------------------------------------------------------------------
class TestVoices:
    def test_consent_required(self, store):
        with pytest.raises(VoiceError, match="Consent"):
            store.add_audio("x", speech_like(), SR, consent=False)

    def test_silence_trimmed_and_quality_saved(self, store):
        a = np.concatenate([np.zeros(SR * 3, np.float32), speech_like(6)])
        v = store.add_audio("Trim", a, SR, True)
        assert v.duration < 6.5
        assert v.quality["verdict"] in ("great", "ok")

    def test_too_short_after_trim(self, store):
        a = np.concatenate([np.zeros(SR * 10, np.float32), speech_like(1)])
        with pytest.raises(VoiceError, match="too short"):
            store.add_audio("Short", a, SR, True)

    # REGRESSION (0.4): uploads went through a held-open NamedTemporaryFile, which Windows
    # refuses to reopen. add_bytes decodes in memory.
    def test_add_bytes_no_tempfile(self, store, monkeypatch):
        import tempfile
        monkeypatch.setattr(tempfile, "NamedTemporaryFile",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("temp file used")))
        buf = io.BytesIO()
        sf.write(buf, speech_like(), SR, format="WAV")
        assert store.add_bytes("Bytes", buf.getvalue(), True).name == "Bytes"

    def test_resolve_by_name(self, store, voice):
        assert store.resolve("leon").id == voice.id
        assert store.resolve(voice.id).id == voice.id
        with pytest.raises(VoiceNotFound):
            store.resolve("nobody")

    def test_bad_ids_rejected(self, store):
        for bad in ("../etc", "", "A B"):
            with pytest.raises(VoiceNotFound):
                store.get(bad)

    def test_settings_validation_and_merge(self, store, voice):
        store.update_settings(voice.id, {"exaggeration": 0.9, "model": "turbo"})
        with pytest.raises(VoiceError):
            store.update_settings(voice.id, {"speed": 9})
        v = store.get(voice.id)
        eff = effective_settings(v, {"exaggeration": 0.3, "temperature": None})
        assert eff["exaggeration"] == 0.3 and eff["model"] == "turbo" and eff["temperature"] == 0.8

    def test_reads_04_meta(self, store):
        d = store.root / "old-abc123"
        d.mkdir()
        sf.write(str(d / "reference.wav"), speech_like(), SR)
        (d / "meta.json").write_text(json.dumps({"id": "old-abc123", "name": "Old",
                                                 "path": "C:/elsewhere/reference.wav",
                                                 "created": 1, "duration": 10, "consent": True,
                                                 "settings": {}}))
        v = store.get("old-abc123")
        assert v.path == str(d / "reference.wav")      # folder wins over stale path

    def test_rename_and_delete(self, store, voice):
        assert store.rename(voice.id, "New").name == "New"
        store.delete(voice.id)
        assert store.list() == []


# ---- jobs -------------------------------------------------------------------------
class TestJobs:
    def test_job_with_srt(self, engine, tmp_path):
        from chatterbox_studio.jobs import JobManager
        jm = JobManager(engine, tmp_path)
        j = jm.wait(jm.submit("First line. Second line.", "", None, {}, "flac", srt=True,
                              ).id)
        assert j.status == "done", j.error
        assert j.done == j.total == 1
        assert "First line." in open(j.paths["srt"], encoding="utf-8").read()
        assert sf.info(j.paths["audio"]).format == "FLAC"

    def test_job_failure_reported(self, engine, tmp_path):
        from chatterbox_studio.jobs import JobManager
        jm = JobManager(engine, tmp_path)
        j = jm.wait(jm.submit("Bonjour.", "", None, {"language": "fr"}).id)
        assert j.status == "failed" and "built-in voice" in j.error

    def test_history(self, engine, tmp_path):
        from chatterbox_studio.jobs import history, save_result
        r = engine.synthesize("Hello history.", None)
        save_result(r, "wav", text="Hello history.", out_dir=tmp_path)
        h = history(tmp_path)
        assert h[0]["text"] == "Hello history." and h[0]["seconds"] > 0
