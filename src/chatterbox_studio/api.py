# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Local REST API.

    GET    /health                      no auth; model/device/busy status
    GET    /languages  /models  /presets
    GET    /voices                      POST /voices (multipart: name, consent, file)
    GET    /voices/{id}                 PATCH /voices/{id} {"name"}     DELETE /voices/{id}
    GET    /voices/{id}/audio           the reference clip
    PUT    /voices/{id}/settings        save per-voice defaults
    POST   /tts                         -> audio in the requested format
    POST   /tts/stream                  -> chunked WAV, plays as it generates
    POST   /jobs                        long-form, background; GET /jobs, GET /jobs/{id}
    GET    /jobs/{id}/audio  /jobs/{id}/srt     DELETE /jobs/{id} (cancel)
    POST   /v1/audio/speech             OpenAI-compatible     GET /v1/models

Voices may be referenced by id or by (unique) name. Unknown OpenAI stock voice names
("alloy", "nova", ...) map to the default voice so off-the-shelf clients just work.
"""
import hmac
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse, Response,
                               StreamingResponse)
from pydantic import BaseModel, Field

from . import __version__, audio, config
from .engine import EngineError, TTSEngine, resolve_kind
from .jobs import JobManager, QueueFull
from .voices import VoiceError, VoiceNotFound, VoiceStore, effective_settings

OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova",
                 "sage", "shimmer", "verse"}


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _hostname(value: str) -> str:
    """'example.com:8765' -> 'example.com'; '[::1]:8765' -> '[::1]'."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        return value[: value.find("]") + 1] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def origin_allowed(origin: str, host_header: str, extra: set) -> bool:
    """Exact-origin policy (CBX-10): a browser request is allowed only from this API's own
    origin (its /docs page), i.e. http://<the Host it was addressed to>, or from an origin
    listed exactly in CBX_ALLOWED_ORIGINS. Other ports and schemes on localhost are other
    sites and are refused; so is the opaque origin "null"."""
    origin = (origin or "").strip().lower().rstrip("/")
    if origin in extra:
        return True
    host = (host_header or "").strip().lower()
    return bool(host) and origin == f"http://{host}"


def check_bind(host: str, token: str, allow_insecure: bool = False):
    """Deny-first: refuse to expose the API beyond this machine without a token."""
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not token and not allow_insecure:
        raise RuntimeError(
            f"Refusing to listen on {host} without a token. Anyone on your network could "
            "use your cloned voices. Set CBX_TOKEN=<a long random string> (or, at your own "
            "risk, CBX_ALLOW_INSECURE=1).")


class SynthOptions(BaseModel):
    voice_id: str = ""
    language: Optional[str] = None
    model: Optional[str] = None
    exaggeration: Optional[float] = Field(None, ge=0.25, le=2.0)
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(None, ge=0.05, le=5.0)
    speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    sentence_pause: Optional[float] = Field(None, ge=0.0, le=3.0)
    paragraph_pause: Optional[float] = Field(None, ge=0.0, le=5.0)
    preset: Optional[str] = None
    seed: Optional[int] = None
    markdown: bool = True

    def overrides(self) -> dict:
        base = {}
        if self.preset:
            if self.preset not in config.PRESETS:
                raise HTTPException(400, f"Unknown preset '{self.preset}'. "
                                         f"Choose: {', '.join(config.PRESETS)}.")
            base = {k: v for k, v in config.PRESETS[self.preset].items() if k != "label"}
        explicit = {k: getattr(self, k) for k in
                    ("language", "model", "exaggeration", "cfg_weight", "temperature",
                     "speed", "sentence_pause", "paragraph_pause")}
        return {**base, **{k: v for k, v in explicit.items() if v is not None}}


class TTSRequest(SynthOptions):
    text: str = Field(min_length=1)
    format: str = "wav"


class JobRequest(SynthOptions):
    text: str = Field(min_length=1, max_length=config.MAX_JOB_CHARS)
    format: str = "wav"
    srt: bool = False


class SettingsIn(BaseModel):
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    language: Optional[str] = None
    model: Optional[str] = None
    speed: Optional[float] = None
    sentence_pause: Optional[float] = None
    paragraph_pause: Optional[float] = None


class RenameIn(BaseModel):
    name: str


class OpenAISpeech(BaseModel):
    model: str = "chatterbox"
    input: str = Field(min_length=1)
    voice: str = ""
    response_format: str = "mp3"       # OpenAI's default
    speed: Optional[float] = Field(None, ge=0.25, le=4.0)
    instructions: Optional[str] = None  # accepted and ignored


def create_app(engine: TTSEngine, store: VoiceStore,
               jobs: Optional[JobManager] = None) -> FastAPI:
    app = FastAPI(title="Chatterbox Studio", version=__version__)
    jobs = jobs or JobManager(engine)

    def auth(authorization: Optional[str] = Header(None)):
        if not config.API_TOKEN:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token.encode(), config.API_TOKEN.encode()):
            raise HTTPException(401, "Invalid or missing bearer token.")

    A = [Depends(auth)]

    @app.middleware("http")
    async def browser_guard(request, call_next):
        """Block web pages from driving the API (CBX-06).

        * Any request with an Origin header must come from this API's own origin (or one
          listed exactly in CBX_ALLOWED_ORIGINS). Browsers always send Origin on cross-site POSTs, while
          scripts, curl, the OpenAI SDK and the MCP server send none. This stops a web page
          planting a voice with a plain HTML form (no CORS preflight needed).
        * Without a token, the Host header must be loopback. A DNS-rebinding page arrives
          as Host: attacker.example, so it is rejected even though the socket is local.
        """
        origin = request.headers.get("origin")
        extra = {o.strip().lower().rstrip("/") for o in config.ALLOWED_ORIGINS.split(",")
                 if o.strip()}
        if origin is not None and not origin_allowed(origin, request.headers.get("host", ""),
                                                     extra):
            return JSONResponse({"detail": "Cross-origin browser requests are not allowed."},
                                status_code=403)
        if not config.API_TOKEN and not config.ALLOW_INSECURE and _hostname(request.headers.get("host", "")) not in LOOPBACK_HOSTS:
            return JSONResponse({"detail": "Host not allowed. Use 127.0.0.1 or localhost, "
                                           "or set a token for network access."}, status_code=403)
        return await call_next(request)

    def find_voice(ref: str, openai: bool = False):
        if not ref:
            return None
        try:
            v = store.resolve(ref)
        except VoiceNotFound:
            if openai and ref.lower() in OPENAI_VOICES:
                return None
            raise HTTPException(404, f"Unknown voice '{ref}'. GET /voices lists them.")
        except VoiceError as e:
            raise HTTPException(400, str(e))
        if not v.consent:
            raise HTTPException(403, "Voice has no consent attestation.")
        return v

    def check_text(t: str, limit: int = config.MAX_CHARS):
        if len(t) > limit:
            raise HTTPException(413, f"Text is {len(t)} characters; the limit here is {limit}. "
                                     "Use POST /jobs for long-form text.")

    def check_format(fmt: str):
        if fmt not in config.FORMATS:
            raise HTTPException(400, f"Unsupported format '{fmt}'. Use: {', '.join(config.FORMATS)}.")

    def synth_response(text, voice, overrides, fmt, seed=None, markdown=True) -> Response:
        check_format(fmt)
        s = effective_settings(voice, overrides)
        try:
            res = engine.synthesize(text, voice.path if voice else None, s,
                                    seed=seed, markdown=markdown)
            data = audio.encode(res.audio, res.sr, fmt)
        except (EngineError, audio.AudioError) as e:
            raise HTTPException(500 if isinstance(e, EngineError) else 400, str(e))
        headers = {"X-Audio-Seconds": f"{res.seconds:.2f}", "X-Sample-Rate": str(res.sr)}
        return Response(data, media_type=audio.MIME[fmt], headers=headers)

    # ---- info ------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, **engine.status()}

    @app.get("/languages", dependencies=A)
    def languages():
        return dict(config.LANGUAGES)

    @app.get("/models", dependencies=A)
    def models():
        return dict(config.MODELS)

    @app.get("/presets", dependencies=A)
    def presets():
        return config.PRESETS

    # ---- voices ----------------------------------------------------------------
    @app.get("/voices", dependencies=A)
    def list_voices():
        return [v.public() for v in store.list()]

    @app.post("/voices", dependencies=A)
    async def add_voice(name: str = Form(...), consent: bool = Form(...),
                        file: UploadFile = File(...)):
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Audio file is larger than 50 MB.")
        try:
            return store.add_bytes(name, data, consent).public()
        except VoiceError as e:
            raise HTTPException(400, str(e))

    @app.get("/voices/{voice_id}", dependencies=A)
    def get_voice(voice_id: str):
        try:
            return store.get(voice_id).public()
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")

    @app.get("/voices/{voice_id}/audio", dependencies=A)
    def voice_audio(voice_id: str):
        try:
            return FileResponse(store.get(voice_id).path, media_type="audio/wav")
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")

    @app.patch("/voices/{voice_id}", dependencies=A)
    def rename_voice(voice_id: str, body: RenameIn):
        try:
            return store.rename(voice_id, body.name).public()
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")
        except VoiceError as e:
            raise HTTPException(400, str(e))

    @app.put("/voices/{voice_id}/settings", dependencies=A)
    def set_defaults(voice_id: str, body: SettingsIn):
        try:
            return store.update_settings(voice_id, body.model_dump(exclude_none=True)).public()
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")
        except VoiceError as e:
            raise HTTPException(400, str(e))

    @app.delete("/voices/{voice_id}", dependencies=A)
    def delete_voice(voice_id: str):
        try:
            store.delete(voice_id)
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")
        return {"deleted": voice_id}

    # ---- speech ----------------------------------------------------------------
    @app.post("/tts", dependencies=A)
    def tts(req: TTSRequest):
        check_text(req.text)
        return synth_response(req.text, find_voice(req.voice_id), req.overrides(),
                              req.format.lower(), req.seed, req.markdown)

    @app.post("/tts/stream", dependencies=A)
    def tts_stream(req: TTSRequest):
        """Chunked WAV. The first piece is generated before the response starts, so setup
        errors still return a proper HTTP error; a failure mid-stream aborts the
        connection. Closing the connection stops generation."""
        check_text(req.text)
        voice = find_voice(req.voice_id)
        s = effective_settings(voice, req.overrides())
        it = engine.render(req.text, voice.path if voice else None, s, seed=req.seed,
                           chunk_limit=200, markdown=req.markdown)
        try:
            first = next(it)
        except StopIteration:
            raise HTTPException(500, "No audio was produced.")
        except EngineError as e:
            raise HTTPException(500, str(e))

        def body():
            try:
                yield audio.stream_wav_header(first.sr)
                yield audio.pcm16(first.audio)
                for p in it:
                    yield audio.pcm16(p.audio)
            finally:
                it.close()        # releases the engine lock if the client disconnects

        return StreamingResponse(body(), media_type="audio/wav")

    # ---- jobs ------------------------------------------------------------------
    @app.post("/jobs", dependencies=A)
    def create_job(req: JobRequest):
        fmt = req.format.lower()
        if fmt == "pcm":
            raise HTTPException(400, "Jobs produce files; use wav, flac, mp3, ogg or opus.")
        check_format(fmt)
        voice = find_voice(req.voice_id)
        s = effective_settings(voice, req.overrides())
        try:
            resolve_kind(s["model"], s["language"])          # fail fast, not in the queue
        except EngineError as e:
            raise HTTPException(400, str(e))
        try:
            job = jobs.submit(req.text, voice.id if voice else "", voice.path if voice else None,
                              s, fmt, req.srt, req.seed)
        except QueueFull as e:
            raise HTTPException(429, str(e), headers={"Retry-After": "30"})
        except ValueError as e:
            raise HTTPException(400, str(e))
        return job.public()

    @app.get("/jobs", dependencies=A)
    def list_jobs():
        return [j.public() for j in jobs.list()]

    def _job(job_id):
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job.")

    @app.get("/jobs/{job_id}", dependencies=A)
    def get_job(job_id: str):
        return _job(job_id).public()

    @app.get("/jobs/{job_id}/audio", dependencies=A)
    def job_audio(job_id: str):
        j = _job(job_id)
        if j.status != "done":
            raise HTTPException(409, f"Job is {j.status}.")
        p = Path(j.paths["audio"])
        return FileResponse(p, media_type=audio.MIME[j.fmt], filename=p.name)

    @app.get("/jobs/{job_id}/srt", dependencies=A)
    def job_srt(job_id: str):
        j = _job(job_id)
        if j.status != "done" or "srt" not in j.paths:
            raise HTTPException(409, "No subtitles for this job (submit with srt=true).")
        return PlainTextResponse(Path(j.paths["srt"]).read_text(encoding="utf-8"),
                                 media_type="application/x-subrip")

    @app.delete("/jobs/{job_id}", dependencies=A)
    def cancel_job(job_id: str):
        _job(job_id)
        return jobs.cancel(job_id).public()

    # ---- OpenAI compatible -----------------------------------------------------
    @app.get("/v1/models", dependencies=A)
    def openai_models():
        return {"object": "list", "data": [{"id": "chatterbox", "object": "model",
                                            "owned_by": "local"}]}

    @app.post("/v1/audio/speech", dependencies=A)
    def openai_speech(req: OpenAISpeech):
        check_text(req.input)
        fmt = req.response_format.lower()
        if fmt not in config.FORMATS:
            raise HTTPException(400, "response_format must be mp3, opus, flac, wav or pcm.")
        ov = {}
        if req.speed is not None:
            ov["speed"] = max(0.5, min(2.0, req.speed))
        return synth_response(req.input, find_voice(req.voice, openai=True), ov, fmt)

    return app
