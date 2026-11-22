# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Saving generations (audio + subtitles + a JSON sidecar for history) and a small
background job queue for long-form text such as articles, scripts and audiobooks.

One worker thread runs jobs in order; the engine lock already serialises synthesis,
so a single worker keeps memory flat and progress honest.
"""
import json
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from . import audio, config
from .engine import Cancelled, EngineError, Result, TTSEngine


# ---- outputs & history ---------------------------------------------------------
def _stem(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)[:6]
    return ("_".join(words)[:40] or "speech").lower()


def save_result(result: Result, fmt: str = "wav", dest: Optional[Path] = None, *,
                text: str = "", voice: str = "", settings: Optional[dict] = None,
                srt: bool = False, out_dir: Optional[Path] = None) -> dict:
    """Write audio (+ optional .srt) and a .json sidecar. Returns the paths written."""
    out_dir = Path(out_dir or config.OUT_DIR)
    if dest is None:
        dest = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{_stem(text)}.{fmt}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(audio.encode(result.audio, result.sr, fmt))
    paths = {"audio": str(dest)}
    if srt:
        sp = dest.with_suffix(".srt")
        sp.write_text(audio.srt(result.cues), encoding="utf-8")
        paths["srt"] = str(sp)
    side = dest.with_suffix(dest.suffix + ".json")
    side.write_text(json.dumps({
        "created": time.time(), "text": text, "voice": voice, "settings": settings or {},
        "seconds": round(result.seconds, 2), "format": fmt, **paths}, indent=2), encoding="utf-8")
    paths["meta"] = str(side)
    return paths


def history(out_dir: Optional[Path] = None, limit: int = 200) -> list[dict]:
    out_dir = Path(out_dir or config.OUT_DIR)
    items = []
    for side in out_dir.glob("*.json"):
        try:
            d = json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(d.get("audio", "")).is_file():
            d["meta"] = str(side)
            items.append(d)
    items.sort(key=lambda d: d.get("created", 0), reverse=True)
    return items[:limit]


# ---- jobs ----------------------------------------------------------------------
@dataclass
class Job:
    id: str
    text: str
    voice_id: str
    voice_path: Optional[str]
    settings: dict
    fmt: str = "wav"
    srt: bool = False
    seed: Optional[int] = None
    status: str = "queued"            # queued | running | done | failed | cancelled
    done: int = 0
    total: int = 0
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    seconds: float = 0.0
    error: str = ""
    paths: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        hidden = ("cancel", "voice_path", "text", "paths")
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in hidden}
        d["settings"] = dict(self.settings)
        d["preview"] = self.text[:120]
        d["chars"] = len(self.text)
        d["progress"] = round(self.done / self.total, 3) if self.total else 0.0
        d["has_audio"] = "audio" in self.paths
        d["has_srt"] = "srt" in self.paths
        return d


class JobManager:
    MAX_KEEP = 100

    def __init__(self, engine: TTSEngine, out_dir: Optional[Path] = None):
        self.engine = engine
        self.out_dir = Path(out_dir or config.JOBS_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="cbx-jobs", daemon=True)
        self._worker.start()

    def submit(self, text: str, voice_id: str, voice_path: Optional[str], settings: dict,
               fmt: str = "wav", srt: bool = False, seed: Optional[int] = None) -> Job:
        if fmt not in config.FORMATS or fmt == "pcm":
            raise ValueError(f"Unsupported job format '{fmt}'.")
        job = Job(uuid.uuid4().hex[:12], text, voice_id, voice_path, dict(settings),
                  fmt, srt, seed)
        with self._lock:
            self._jobs[job.id] = job
            self._prune()
        self._q.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = time.time()
        return job

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j.status in ("done", "failed", "cancelled"):
                return j
            time.sleep(0.02)
        return self.get(job_id)

    def _prune(self):
        finished = [j for j in self._jobs.values() if j.status in ("done", "failed", "cancelled")]
        excess = len(self._jobs) - self.MAX_KEEP
        for j in sorted(finished, key=lambda j: j.created)[:max(0, excess)]:
            self._jobs.pop(j.id, None)

    def _run(self):
        while True:
            jid = self._q.get()
            try:
                job = self.get(jid)
            except KeyError:
                continue
            if job.cancel.is_set():
                continue
            job.status, job.started = "running", time.time()

            def progress(done, total, j=job):
                j.done, j.total = done, total
            try:
                res = self.engine.synthesize(job.text, job.voice_path, job.settings,
                                             seed=job.seed, cancel=job.cancel,
                                             on_progress=progress)
                job.paths = save_result(res, job.fmt, self.out_dir / f"{job.id}.{job.fmt}",
                                        text=job.text, voice=job.voice_id,
                                        settings=job.settings, srt=job.srt)
                job.seconds = round(res.seconds, 2)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except (EngineError, audio.AudioError) as e:
                job.status, job.error = "failed", str(e)
            except Exception as e:                      # never kill the worker
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished = time.time()
