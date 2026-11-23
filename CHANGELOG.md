# Changelog

## 0.5.2 — 2026-11-23 · re-review correction release

Fixes from the re-review of 0.5.1. Each finding was reproduced first: the new tests in
`tests/test_review_052.py` fail on 0.5.1 (12 fail, 4 controls pass) and all pass on 0.5.2.

- **CBX-07 (high) — "never overwrite" wasn't atomic.** The existence check and
  `os.replace` were separate steps, so a file created in between was replaced. New files
  are now published in one no-replace step: `os.rename` on Windows (never replaces),
  `os.link` on POSIX, and exclusive create on filesystems without hard links.
  Interleaved-writer test included.
- **CBX-08 — a take's files weren't committed as a unit.** If the subtitles or sidecar
  failed, the audio stayed behind. `save_result` is now all-or-nothing: everything is
  encoded first, replaced files are moved aside, and on failure the files this call created
  are removed and the originals restored. Pre-existing files are never deleted. MCP
  `speak_long` does the same for its audio + subtitle pair.
- **CBX-09 — long MCP requests lost settings.** Over 2,000 characters, `speak()`
  dropped `exaggeration` and `cfg_weight` when handing off to a job. Both tools now build
  options in one shared function, and `temperature` is accepted by both. A test compares
  the short and long requests field by field.
- **CBX-10 — origin check trusted all of localhost.** `http://localhost:9999` (any local
  app) was accepted. Browser requests are now allowed only from the API's own exact origin
  (its `/docs` page) or origins listed exactly in `CBX_ALLOWED_ORIGINS`. Other ports,
  `https`, other loopback names and `null` are refused.
- **MCP write race (from the re-review's note).** Validation and file creation are separate
  steps; reproduced that swapping an output sub-folder for a symlink in between redirected
  the write outside. The MCP writer now re-checks where the file actually landed right
  after the exclusive create, before writing data, and removes it if it escaped.
  (A symlink planted at the file name itself was already refused by the exclusive create;
  now tested.)
- Two 0.5.1 tests were updated for changed mechanisms (publish now uses link/rename; exact
  origins). Their intent is unchanged, and they still pass against 0.5.1.

Known limitations (accepted, documented):
- The no-hard-link fallback never replaces a file, but a reader could briefly see a file
  that is still being filled.
- Restoring originals after a failed overwrite relies on `os.replace` succeeding; if the
  disk fails mid-rollback, the original stays as a hidden `.bak` file beside it.
- The Windows `os.rename` publish path is exercised only on Windows (release checklist).

## 0.5.1 — 2026-11-23 · review correction release

Fixes from the external adversarial review of 0.5.0. Every finding was reproduced against
the 0.5.0 code before fixing, and every fix has a regression test that fails on 0.5.0
(`tests/test_review_051.py`: 16 fail on 0.5.0, 5 controls pass on both, all 21 pass on 0.5.1).

### Security
- **CBX-01 (high) — MCP could write anywhere.** `speak`/`speak_long` accepted any `save_to`
  path and overwrote existing files. MCP output is now confined to the output folder, plus
  folders the user lists in `CBX_MCP_OUTPUT_DIRS`. Paths are resolved through symlinks before
  the check, existing files are never overwritten (exclusive create), and the extension must
  match the audio format. Bad paths are refused *before* any synthesis.
- **CBX-06 (high, new) — web pages could drive the API.** The review raised browser-origin
  and DNS-rebinding risk as untested. Verification showed it was real: with no token (the
  default), any web page could plant a voice via a plain HTML form POST, and a DNS-rebinding
  page could list and delete voices. Now browser requests from non-loopback origins get 403
  (`CBX_ALLOWED_ORIGINS` to opt in), and without a token only loopback `Host` headers are
  accepted. Scripts, curl, the OpenAI SDK and MCP send no `Origin` and are unaffected.
- **CBX-02 (high) — unbounded job queue.** Reproduced: 150 jobs (15 M characters) admitted.
  Admission is now capped at 20 pending jobs and 2 M pending characters. `POST /jobs`
  returns 429 with `Retry-After`; the app and MCP show the message. Finished jobs release
  their text from memory (the history sidecar keeps it on disk).

### Correctness
- **CBX-04 — cancel vs. finalise race.** Reproduced: a cancel arriving after the last chunk
  still produced a "done" job with files. There is now a defined commit boundary: a cancel
  before it writes nothing; after it, cancel is a no-op. All outputs are written to a temp
  file and published with `os.replace`, sidecar last, so history never lists partial takes.
- **CBX-05 — takes could overwrite each other.** Reproduced: same leading words within one
  second gave the same filename. Generated names now carry a random suffix and never
  overwrite. Explicit destinations (`-o`, Save as…) still overwrite, as the user chose them.
- **Turbo log spam (new).** Found reading chatterbox-tts 0.1.7: Turbo's `generate()` accepts
  `exaggeration`/`cfg_weight` but ignores them and logs a warning on every chunk when they
  are > 0. They are now sent as 0 for Turbo/Nano.
- **Nano error was wrong (new).** chatterbox-tts 0.1.7 (latest on PyPI) has no Nano model.
  0.5.0 told users to "upgrade", which can't help. It now detects support by signature and
  explains the GitHub install. `doctor` reports Turbo/Nano/Multilingual V2-vs-V3 accurately.
- Multilingual V3 is now chosen by signature, not by catching `TypeError`, which could mask
  real errors and load twice.

### Not changed, with reasons
- **CBX-03 — conditioning cache ignores exaggeration: not a defect.** Checked against the
  chatterbox-tts 0.1.7 source: Standard and Multilingual `generate()` rebuild the emotion
  conditioning whenever the `exaggeration` they are given differs, and Turbo disables it.
  The cache only skips re-running the speaker encoder. A test pins the invariant this
  relies on (exaggeration reaches every `generate()` call), and a real-model listening
  check is on the release checklist.
- **59 vs 69 tests.** Not missing tests: without PyQt6 and the MCP SDK, the GUI (8) and MCP
  (2) modules skip as a whole (59 + 2 skips). To stop that looking green, `test.bat` now
  sets `CBX_STRICT=1`, which fails the run if anything is skipped.

### Docs
- Seed reproducibility qualified (same machine and versions; not guaranteed bit-identical).
- Nano availability, Multilingual V2/V3 and its built-in voice described accurately.
- New `RELEASE_CHECKLIST.md`: Windows / real-model / MCP / browser smoke checks.

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
