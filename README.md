# Chatterbox Studio

Record your voice → clone it with Resemble AI's Chatterbox → play it back → use it from code or LLMs.

## Setup (Windows, Python 3.11)
    python -m venv .venv && .venv\Scripts\activate
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121   # GPU build
    pip install -r requirements.txt

## Run
    python -m chatterbox_studio gui      # desktop app (can also host the API)
    python -m chatterbox_studio serve    # headless API on 127.0.0.1:8765
    python -m chatterbox_studio mcp      # MCP stdio server (talks to the API above)

## API
    GET  /health
    GET  /voices
    POST /voices          multipart: name, consent=true, file
    DELETE /voices/{id}
    PUT  /voices/{id}/settings   {"exaggeration","cfg_weight","temperature","language"}  -> saved as that voice's defaults
    GET  /languages
    POST /tts             {"text","voice_id","language"?,"exaggeration"?,"cfg_weight"?,"temperature"?}  -> audio/wav
    POST /tts/stream      same body -> chunked WAV (plays as sentence-groups finish)
    POST /v1/audio/speech OpenAI-compatible (voice = voice_id, wav only)

Env: CBX_DATA, CBX_HOST, CBX_PORT, CBX_TOKEN (Bearer auth), CBX_URL (MCP -> API).

## MCP config (Claude Desktop etc.)
    {"mcpServers": {"chatterbox": {"command": "C:\\path\\.venv\\Scripts\\python.exe",
                                   "args": ["-m", "chatterbox_studio", "mcp"]}}}

## Defaults, languages, streaming
- Unset request options fall back to the voice's saved defaults, then to global defaults.
- English uses ChatterboxTTS; any other language switches to ChatterboxMultilingualTTS
  (needs a recent `pip install -U chatterbox-tts`). Only one model is kept in memory, so
  switching languages reloads. The multilingual model has no built-in voice: pick a cloned one.
- Streaming is per sentence-group, not per token. If generation is slower than realtime
  you will hear gaps between chunks; turn streaming off for gap-free playback.

## Notes
- Voices require a consent attestation; the API refuses any voice without it.
- Chatterbox embeds an inaudible Perth watermark in all output.
- ~10 s of clean, single-speaker audio clones best.
