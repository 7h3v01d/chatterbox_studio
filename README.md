# Chatterbox Studio

**Clone your voice in about 15 seconds, then use it anywhere**: in a desktop app, from the command line, over HTTP, through any OpenAI-compatible speech client, or from Claude Desktop and other AI tools via MCP.

Built on [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI. Everything runs on your own computer. Nothing is uploaded.

---

## Install (Windows)

1. Install **Python 3.11** from python.org. Tick **“Add python.exe to PATH.”**
2. Double-click **`install.bat`**. It creates a private environment, installs everything, and if you have an NVIDIA graphics card, installs the GPU version of PyTorch automatically. It ends with a health check.
3. Double-click **`run.bat`**.

The first time you press **Speak**, the voice model downloads (a few GB, once).

> Mac / Linux: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e .` then `python -m chatterbox_studio`.

Something not working? Run `.venv\Scripts\python -m chatterbox_studio doctor` for a plain-English check of Python, GPU, microphone, speakers and ports.

## Using the app

**Make a voice**: click **＋ New voice**, read the short passage aloud, and stop when the bar fills (about 10–15 s). You get an instant quality check (noise, clipping, length) with tips if anything needs fixing. Name it, confirm consent, and save. You can also drag an audio file onto the window.

**Speak**: pick a voice, type, and press **Speak** (or Ctrl+Enter). Choose a **Style**:

| Style | Good for |
|---|---|
| Natural | everyday speech (default) |
| Narrator | steady voice-over and audiobooks |
| Calm | relaxed, even delivery |
| Expressive / Dramatic | characters, ads, energy |
| Other-language voice | when the voice is in a different language from the text (stops accent bleed) |

**Fine-tune** gives you Emotion, Pacing, Variation, Speed, Pause length and a **Seed**. The same seed with the same settings should reproduce the same take *on the same machine with the same versions*; GPU kernels, drivers and model updates can still change the result, so treat it as "very close", not guaranteed bit-identical. **Save as voice default** remembers your settings for that voice, everywhere (app, CLI, API, MCP).

**In your text:**
- `[pause 1s]`, `[pause 500ms]` or `[pause]` add silence
- a blank line starts a new paragraph, with a slightly longer pause
- Markdown is cleaned up automatically (headings, bullets, bold, links)
- with the **Turbo** model: `[laugh]`, `[chuckle]`, `[cough]`

**Models**

| Model | Use it for |
|---|---|
| Auto | picks Standard for English, Multilingual otherwise |
| Standard | English with the full Emotion/Pacing controls |
| Turbo | fastest English, with reaction tags |
| Nano | smallest English model, usable on CPU. **Not in the PyPI release (0.1.7)**; needs chatterbox installed from GitHub (`doctor` tells you) |
| Multilingual | 23 languages. Uses V3 when your chatterbox-tts has it, V2 otherwise. Its built-in voice speaks with an English accent, so use one of your voices |

**Long-form**: paste or open a `.txt`/`.md` file (articles, scripts, chapters). It narrates in the background with progress and an ETA, can be cancelled, and can write an **.srt subtitle file** timed to the audio, ready for video editors.

**History**: every take is kept. Play it again, reuse its text and settings, or find the file.

**Connect**: start the local server and copy a ready-made setup for Claude Desktop, Python or curl.

## Command line

```bat
python -m chatterbox_studio say "Hello there" --voice Leon -o hello.mp3
python -m chatterbox_studio say --file chapter1.md --preset narrator --srt -o chapter1.mp3
type notes.txt | python -m chatterbox_studio say - --voice Leon --play
python -m chatterbox_studio clone my_voice.wav --name Leon --consent
python -m chatterbox_studio voices
python -m chatterbox_studio serve --preload
python -m chatterbox_studio doctor
```

Voices can be given by **name** or id. Output format follows the file extension (wav, flac, mp3, ogg, opus).

## HTTP API

Start it from the app (**Connect → Start server**) or with `serve.bat`. Interactive docs are at `http://127.0.0.1:8765/docs`.

```
GET    /health                         status (no auth)
GET    /voices   /presets  /models  /languages
POST   /voices                         multipart: name, consent=true, file
GET    /voices/{id}   PATCH /voices/{id} {"name"}   DELETE /voices/{id}
GET    /voices/{id}/audio              reference clip
PUT    /voices/{id}/settings           save per-voice defaults
POST   /tts                            {"text","voice_id","preset","model","language","speed","seed","format",...}
POST   /tts/stream                     chunked WAV that starts playing straight away
POST   /jobs                           long-form in the background {"text",...,"format","srt":true}
GET    /jobs  /jobs/{id}  /jobs/{id}/audio  /jobs/{id}/srt     DELETE /jobs/{id} cancels
POST   /v1/audio/speech                OpenAI-compatible     GET /v1/models
```

Unset options fall back to the voice's saved defaults, then to global defaults. `/tts` accepts up to 20,000 characters (`CBX_MAX_CHARS`); use `/jobs` for anything longer (up to 500,000).

**OpenAI-compatible**: point any OpenAI client at `http://127.0.0.1:8765/v1`. `voice` takes your voice's name or id. Stock names such as `alloy` or `nova` fall back to the built-in voice, so existing apps work unchanged. Supports `mp3` (default), `opus`, `flac`, `wav`, `pcm`, and `speed`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
        model="chatterbox", voice="Leon", input="Hello from my own voice!") as r:
    r.stream_to_file("hello.mp3")
```

## MCP (Claude Desktop and other AI tools)

In the app: **Connect → Claude Desktop (MCP) config** copies the exact config for your machine. Paste it into `claude_desktop_config.json` and restart Claude Desktop. The server must be running.

Tools: `status`, `list_voices`, `list_presets`, `speak` (switches to a background job automatically for long text), `speak_long` (audio + subtitles), `set_voice_defaults`, `clone_voice` (consent required). Deleting voices is deliberately **not** exposed to AI tools.

Audio from MCP is saved in Chatterbox Studio's `output` folder. An AI tool can pick a file name but cannot write anywhere else, cannot overwrite an existing file, and the extension must match the audio format. To let it save into other folders, list them in `CBX_MCP_OUTPUT_DIRS` (separated by `;` on Windows).

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `CBX_DATA` | `%USERPROFILE%\.chatterbox_studio` | voices, output, jobs |
| `CBX_HOST` / `CBX_PORT` | `127.0.0.1` / `8765` | API address |
| `CBX_TOKEN` | – | require `Authorization: Bearer <token>` |
| `CBX_URL` | `http://HOST:PORT` | where the MCP server finds the API |
| `CBX_MAX_CHARS` | `20000` | per `/tts` request |
| `CBX_ALLOW_INSECURE` | – | `1` allows a network bind without a token (not recommended) |
| `CBX_ALLOWED_ORIGINS` | – | comma-separated web origins allowed to call the API from a browser page |
| `CBX_MCP_OUTPUT_DIRS` | – | extra folders MCP tools may save audio into |

## Safety and privacy

- **Consent is required** for every voice. Only clone your own voice, or one you have explicit permission to use.
- The API listens on **this computer only** by default. It **refuses to listen on your network without a token**.
- **Web pages can't use it.** Browser requests from other sites are rejected (an `Origin` check), and without a token only `localhost`/`127.0.0.1` host names are accepted, which blocks DNS-rebinding tricks. Scripts, curl, the OpenAI SDK and MCP are unaffected.
- **Bounded work.** At most 20 long-form jobs (2 million characters) can wait at once; beyond that the API answers `429 Too Many Requests`.
- Chatterbox embeds an inaudible **PerTh watermark** in all generated audio.
- Resemble AI marked Chatterbox as retired research in August 2026. The model is still MIT-licensed and works, but expect no upstream fixes.

## Development

```bat
test.bat            REM strict: any skipped test fails the run
python -m pytest -q REM non-strict
```

The suite uses a fake model, so it runs in seconds without a GPU or a model download. That proves the app's logic, not speech quality: see `RELEASE_CHECKLIST.md` for the real-model and Windows checks to do before tagging a release.

Layout: `src/chatterbox_studio/` contains `text.py` (planning and chunking), `audio.py` (encode, trim, speed, quality, subtitles), `engine.py` (models), `voices.py`, `jobs.py`, `api.py`, `mcp_server.py`, `cli.py`, `gui.py` and `recorder.py`.

---

Apache-2.0 © 2026 Leon Priest ([7h3v01d](https://github.com/7h3v01d))
