# Changelog

## 0.5.0 — 2026-11-23

### Fixed
- **Long sentences were sent to the model whole.** The chunker ignored its size limit whenever one sentence (or an unpunctuated paragraph) was longer than it. A 2,000-character run-on became a single 1,999-character request, which Chatterbox truncates or garbles. Chunks are now always within the limit: they split at clause punctuation first, then at spaces. Regression tests are revert-proven.
- **Voice upload via the API failed on Windows.** `POST /voices` wrote to a `NamedTemporaryFile` and re-opened it by name while it was still open, which Windows refuses. Uploads are now decoded in memory. Regression test is revert-proven.
- **MCP server crashed on startup with current `mcp`.** mcp 2.x renamed `FastMCP` to `MCPServer`, so `pip install mcp` broke `python -m chatterbox_studio mcp`. Both 1.x and 2.x now work. Regression test is revert-proven.
- Sentences ran into each other: chunks were concatenated with whatever silence the model left. Each chunk is now edge-trimmed and faded, with consistent, adjustable pauses.
- Version numbers disagreed (0.1.0 in the package, 0.2.0 in the API). There is now a single `__version__`.
- Voice folders store their path in `meta.json`; moving `CBX_DATA` broke them. The folder location is now authoritative. 0.4 voices load unchanged.
- `meta.json` writes are atomic.

### Added
- **Guided voice recording**: reading script, mic picker, level meter with clipping warning, length target, auto-stop at 30 s, automatic silence trimming, and an instant quality report (signal-to-noise, clipping, length) with plain-English tips.
- **Models**: Turbo (fast, with `[laugh]`/`[chuckle]`/`[cough]` tags), Nano (CPU-capable) and Multilingual V3, plus "Auto". Only arguments the installed model accepts are passed, so the app works across chatterbox-tts releases.
- **Style presets** (Natural, Narrator, Calm, Expressive, Dramatic, Other-language voice) and plain-language fine-tune controls, plus **speed** (pitch-preserving), **pause length** and a **seed** for repeatable takes.
- **Text markup**: `[pause 1s]`, paragraph pauses, and automatic Markdown clean-up.
- **Long-form narration** as background jobs with progress, ETA and cancel, plus **.srt subtitles** timed to the audio.
- **History** of every take: replay, reuse text and settings, delete.
- **Output formats**: wav, flac, mp3, ogg, opus, pcm.
- **API**: `/jobs`, voice rename and reference download, `/presets`, `/models`, voices by name, 50 MB upload cap, a clear 413 that points long text at `/jobs`.
- **OpenAI compatibility**: mp3 default, opus/flac/pcm, `speed`, `/v1/models`, and stock voice names that fall back to the built-in voice.
- **MCP**: `status`, `list_presets`, `speak_long` (audio + subtitles), presets/model/speed on `speak`, automatic job hand-off for long text, and a readable error when the app isn't running.
- **CLI**: `say` (text, file or stdin; `--srt`, `--play`, `--preset`), `clone`, `voices`, `doctor`. Running with no arguments opens the app.
- **Connect tab**: start/stop the server, choose this computer or the home network, generate a token, and copy ready-made Claude Desktop / Python / curl setups.
- **Installer**: `install.bat` (venv, dependencies, automatic CUDA PyTorch, health check), `run.bat`, `serve.bat`, `test.bat`.
- 69 tests with a fake model (core, API, CLI, MCP and an offscreen GUI smoke test).

### Changed
- **Deny-first networking**: the API refuses a non-loopback bind without a token (override with `CBX_ALLOW_INSECURE=1`). Deleting voices is not exposed over MCP.
- The UI was rebuilt in the house design system (Obsidian/Teal, JetBrains Mono, flat controls). Workers are QObjects on QThreads with `_worker_refs`, and results return through a GUI-thread bridge.
- All playback goes through sounddevice (one audio stack). QtMultimedia is no longer used.
- `src/` layout, `pyproject.toml`, SPDX/Apache-2.0 headers.
