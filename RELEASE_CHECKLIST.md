# Release checklist

The automated suite runs against a fake model. These checks need the real thing: Windows,
Python 3.11, real chatterbox-tts weights, a GPU, a mic and speakers. Tick each one and note
the result before tagging a release.

Record: date · Windows version · Python · torch + CUDA · GPU · chatterbox-tts version

## 1. Install
- [ ] Fresh folder: `install.bat` completes; `doctor` shows all OK (Nano may show "--" on PyPI)
- [ ] `test.bat` passes with **0 skipped** (strict mode)

## 2. App (run.bat)
- [ ] New voice → record ~12 s → quality report shows → save → appears in the list
- [ ] Drag a WAV onto the window → consent prompt → voice imported
- [ ] Speak with "Start playing straight away" on: audio starts before generation finishes
- [ ] Speak with it off: plays after generation; Replay and Save as… (mp3) work
- [ ] Stop mid-sentence stops audio within ~0.1 s; partial take appears in History
- [ ] Styles: Natural vs Dramatic sound clearly different
- [ ] Emotion slider 0.3 → 1.7 on the same voice, same text: audibly different (CBX-03 real-model check)
- [ ] Turbo: `That's hilarious [laugh] anyway.` produces a laugh; console shows **no** "CFG… ignored" warning
- [ ] Multilingual: French text with your voice; Other-language preset reduces accent bleed
- [ ] Same seed twice → takes sound the same (note whether they are bit-identical)
- [ ] Long-form: paste ~3,000 words, narrate to MP3 with subtitles; progress/ETA move; Cancel works
- [ ] Open the .srt in a video editor: cues line up with the audio
- [ ] History: play, reuse text, delete

## 3. Server and integrations
- [ ] Connect → Start server; `http://127.0.0.1:8765/docs` loads and "Try it out" works
- [ ] curl snippet from Connect produces hello.mp3
- [ ] OpenAI SDK snippet produces hello.mp3
- [ ] Claude Desktop: paste the MCP config, restart; "list my voices" and "say hello in my voice" work
- [ ] MCP: ask it to save to `C:\Windows\test.wav` → refused with a clear message
- [ ] Browser: open any website's dev console and run
      `fetch("http://127.0.0.1:8765/voices")` → blocked (403 / CORS error)
- [ ] "My home network" without a token → refused; with a token → reachable from another PC

## 4. Freeze
- [ ] CHANGELOG updated · version bumped in `__init__.py` and `pyproject.toml` · zip built
