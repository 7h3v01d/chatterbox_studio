"""Local REST API. Includes streaming and an OpenAI-compatible /v1/audio/speech endpoint."""
import hmac
import io
import struct
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from . import config
from .engine import EngineError, TTSEngine
from .voices import VoiceError, VoiceNotFound, VoiceStore, effective_settings


def wav_bytes(wav, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze().cpu().numpy(), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pcm16(wav) -> bytes:
    arr = np.clip(wav.squeeze().cpu().numpy(), -1.0, 1.0)
    return (arr * 32767).astype("<i2").tobytes()


def stream_header(sr: int) -> bytes:
    """WAV header with unknown length (0xFFFFFFFF), as streaming clients expect."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", 0xFFFFFFFF))


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice_id: str = ""
    # None => use the voice's saved defaults (then global defaults)
    exaggeration: Optional[float] = Field(None, ge=0.25, le=2.0)
    cfg_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(None, ge=0.05, le=5.0)
    language: Optional[str] = None


class SettingsIn(BaseModel):
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None
    temperature: Optional[float] = None
    language: Optional[str] = None


class OpenAISpeech(BaseModel):
    model: str = "chatterbox"
    input: str = Field(min_length=1, max_length=5000)
    voice: str = ""
    response_format: str = "wav"


def create_app(engine: TTSEngine, store: VoiceStore) -> FastAPI:
    app = FastAPI(title="Chatterbox Studio", version="0.2.0")

    def auth(authorization: Optional[str] = Header(None)):
        if not config.API_TOKEN:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, config.API_TOKEN):
            raise HTTPException(401, "Invalid or missing bearer token.")

    def prepare(voice_id: str, overrides: dict):
        """Returns (voice_path | None, effective settings). Fails closed on consent."""
        voice = None
        if voice_id:
            try:
                voice = store.get(voice_id)
            except VoiceNotFound:
                raise HTTPException(404, f"Unknown voice '{voice_id}'.")
            if not voice.consent:
                raise HTTPException(403, "Voice has no consent attestation.")
        return (voice.path if voice else None), effective_settings(voice, overrides)

    def overrides_of(req: TTSRequest) -> dict:
        return {"exaggeration": req.exaggeration, "cfg_weight": req.cfg_weight,
                "temperature": req.temperature, "language": req.language}

    def run(text, voice_id, overrides) -> Response:
        path, s = prepare(voice_id, overrides)
        try:
            wav, sr = engine.synthesize(text, path, s["exaggeration"], s["cfg_weight"],
                                        s["temperature"], s["language"])
        except EngineError as e:
            raise HTTPException(500, str(e))
        return Response(wav_bytes(wav, sr), media_type="audio/wav")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": engine.loaded, "device": engine.device}

    @app.get("/languages", dependencies=[Depends(auth)])
    def languages():
        return engine.languages()

    @app.get("/voices", dependencies=[Depends(auth)])
    def list_voices():
        return [v.public() for v in store.list()]

    @app.post("/voices", dependencies=[Depends(auth)])
    async def add_voice(name: str = Form(...), consent: bool = Form(...),
                        file: UploadFile = File(...)):
        with tempfile.NamedTemporaryFile(suffix="_" + (file.filename or "up.wav")) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            try:
                return store.add_file(name, tmp.name, consent).public()
            except VoiceError as e:
                raise HTTPException(400, str(e))

    @app.put("/voices/{voice_id}/settings", dependencies=[Depends(auth)])
    def set_defaults(voice_id: str, body: SettingsIn):
        try:
            return store.update_settings(voice_id, body.model_dump(exclude_none=True)).public()
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")
        except VoiceError as e:
            raise HTTPException(400, str(e))

    @app.delete("/voices/{voice_id}", dependencies=[Depends(auth)])
    def delete_voice(voice_id: str):
        try:
            store.delete(voice_id)
        except VoiceNotFound:
            raise HTTPException(404, "Unknown voice.")
        return {"deleted": voice_id}

    @app.post("/tts", dependencies=[Depends(auth)])
    def tts(req: TTSRequest):
        return run(req.text, req.voice_id, overrides_of(req))

    @app.post("/tts/stream", dependencies=[Depends(auth)])
    def tts_stream(req: TTSRequest):
        """Chunked WAV: first sentence-group arrives as soon as it is ready.
        The first chunk is generated before the response starts, so setup errors still
        return a proper HTTP error; a failure mid-stream aborts the connection."""
        path, s = prepare(req.voice_id, overrides_of(req))
        it = engine.synthesize_stream(req.text, path, s["exaggeration"], s["cfg_weight"],
                                      s["temperature"], s["language"])
        try:
            first = next(it)
        except StopIteration:
            raise HTTPException(500, "No audio was produced.")
        except EngineError as e:
            raise HTTPException(500, str(e))

        def body():
            try:
                wav, sr = first
                yield stream_header(sr)
                yield pcm16(wav)
                for wav, _ in it:
                    yield pcm16(wav)
            finally:
                it.close()  # releases the engine lock if the client disconnects

        return StreamingResponse(body(), media_type="audio/wav")

    @app.post("/v1/audio/speech", dependencies=[Depends(auth)])
    def openai_speech(req: OpenAISpeech):
        if req.response_format != "wav":
            raise HTTPException(400, "Only response_format='wav' is supported.")
        return run(req.input, req.voice, {})

    return app
