# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Saving generations (audio + subtitles + a JSON sidecar for history) and a small
background job queue for long-form text such as articles, scripts and audiobooks.

One worker thread runs jobs in order; the engine lock already serialises synthesis.

Governance (0.5.1):
  * Admission is bounded: at most MAX_PENDING jobs and MAX_PENDING_CHARS of text may be
    queued or running. Beyond that, submit() raises QueueFull (the API returns 429).
  * Cancellation has a defined commit boundary. A job can be cancelled while queued or
    rendering. Once synthesis finishes, the job atomically moves to "committing" under its
    lock; a cancel that arrives after that point is a no-op and the job completes. A cancel
    that wins the race means nothing is written.
  * Files are written to a temporary name and published with os.replace, audio and
    subtitles first and the history sidecar last, so history never lists half-written takes.
  * Default output names carry a random suffix, so takes never overwrite each other.
  * Job text is released from memory once the job ends (the sidecar keeps it on disk).
"""
import json
import os
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

MAX_PENDING = 20
MAX_PENDING_CHARS = 2_000_000
FINAL = ("done", "failed", "cancelled")


class QueueFull(RuntimeError):
    pass


# ---- outputs & history ---------------------------------------------------------
def _stem(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)[:6]
    return ("_".join(words)[:40] or "speech").lower()


def default_name(text: str, fmt: str) -> str:
    """Collision-resistant: timestamp + readable stem + random suffix."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}_{_stem(text)}_{uuid.uuid4().hex[:8]}.{fmt}"


def atomic_write(path: Path, data: bytes, overwrite: bool = True):
    """Write to a temp file in the same folder, then publish with os.replace."""
    path = Path(path)
    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"{path} already exists; not overwriting.")
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with open(tmp, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_result(result: Result, fmt: str = "wav", dest: Optional[Path] = None, *,
                text: str = "", voice: str = "", settings: Optional[dict] = None,
                srt: bool = False, out_dir: Optional[Path] = None,
                overwrite: Optional[bool] = None) -> dict:
    """Write audio (+ optional .srt) and a .json sidecar. Returns the paths written.

    Overwrite policy: an explicit `dest` overwrites by default (the user chose that
    path); generated names never overwrite."""
    out_dir = Path(out_dir or config.OUT_DIR)
    if dest is None:
        dest = out_dir / default_name(text, fmt)
        overwrite = False if overwrite is None else overwrite
    overwrite = True if overwrite is None else overwrite
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = audio.encode(result.audio, result.sr, fmt)          # encode before touching disk
    paths = {"audio": str(dest)}
    atomic_write(dest, data, overwrite)
    if srt:
        sp = dest.with_suffix(".srt")
        atomic_write(sp, audio.srt(result.cues).encode("utf-8"), overwrite)
        paths["srt"] = str(sp)
    side = dest.with_suffix(dest.suffix + ".json")
    meta = {"created": time.time(), "text": text, "voice": voice, "settings": settings or {},
            "seconds": round(result.seconds, 2), "format": fmt, **paths}
    atomic_write(side, json.dumps(meta, indent=2).encode("utf-8"), True)   # commit marker
    paths["meta"] = str(side)
    return paths


def history(*dirs: Path, limit: int = 200) -> list[dict]:
    """Takes from the given folders (default: output and jobs), newest first."""
    dirs = dirs or (config.OUT_DIR, config.JOBS_DIR)
    items = []
    for d in dirs:
        for side in Path(d).glob("*.json"):
            try:
                item = json.loads(side.read_text(encoding="utf-8"))
            except Exception:
                continue
            if Path(item.get("audio", "")).is_file():
                item["meta"] = str(side)
                items.append(item)
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
    status: str = "queued"     # queued | running | committing | done | failed | cancelled
    done: int = 0
    total: int = 0
    chars: int = 0
    preview: str = ""
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    seconds: float = 0.0
    error: str = ""
    paths: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict:
        hidden = ("cancel", "lock", "voice_path", "text", "paths")
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in hidden}
        d["settings"] = dict(self.settings)
        d["progress"] = round(self.done / self.total, 3) if self.total else 0.0
        d["has_audio"] = "audio" in self.paths
        d["has_srt"] = "srt" in self.paths
        return d


class JobManager:
    MAX_KEEP = 100

    def __init__(self, engine: TTSEngine, out_dir: Optional[Path] = None,
                 max_pending: int = MAX_PENDING, max_pending_chars: int = MAX_PENDING_CHARS):
        self.engine = engine
        self.out_dir = Path(out_dir or config.JOBS_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_pending = max_pending
        self.max_pending_chars = max_pending_chars
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="cbx-jobs", daemon=True)
        self._worker.start()

    # ---- admission -----------------------------------------------------------
    def _pending(self) -> tuple[int, int]:
        live = [j for j in self._jobs.values() if j.status not in FINAL]
        return len(live), sum(j.chars for j in live)

    def submit(self, text: str, voice_id: str, voice_path: Optional[str], settings: dict,
               fmt: str = "wav", srt: bool = False, seed: Optional[int] = None) -> Job:
        if fmt not in config.FORMATS or fmt == "pcm":
            raise ValueError(f"Unsupported job format '{fmt}'.")
        if len(text) > config.MAX_JOB_CHARS:
            raise ValueError(f"Text is over {config.MAX_JOB_CHARS:,} characters.")
        job = Job(uuid.uuid4().hex[:12], text, voice_id, voice_path, dict(settings),
                  fmt, srt, seed, chars=len(text), preview=text[:120])
        with self._lock:
            n, chars = self._pending()
            if n >= self.max_pending:
                raise QueueFull(f"The queue is full ({n} jobs waiting). Try again when some finish.")
            if chars + job.chars > self.max_pending_chars:
                raise QueueFull(f"Too much text queued ({chars:,} characters waiting). "
                                "Try again when some jobs finish.")
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
        """Cancel a queued or rendering job. No effect once the job is committing or
        finished (see the module docstring for the commit boundary)."""
        job = self.get(job_id)
        with job.lock:
            if job.status in ("queued", "running"):
                job.cancel.set()
                if job.status == "queued":
                    self._finish(job, "cancelled")
        return job

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j.status in FINAL:
                return j
            time.sleep(0.02)
        return self.get(job_id)

    def _prune(self):
        finished = [j for j in self._jobs.values() if j.status in FINAL]
        excess = len(self._jobs) - self.MAX_KEEP
        for j in sorted(finished, key=lambda j: j.created)[:max(0, excess)]:
            self._jobs.pop(j.id, None)

    @staticmethod
    def _finish(job: Job, status: str, error: str = ""):
        job.status, job.error, job.finished = status, error, time.time()
        job.text = ""                      # release memory; the sidecar keeps the text

    # ---- worker --------------------------------------------------------------
    def _run(self):
        while True:
            jid = self._q.get()
            try:
                job = self.get(jid)
            except KeyError:
                continue
            with job.lock:
                if job.status != "queued":
                    continue
                job.status, job.started = "running", time.time()

            def progress(done, total, j=job):
                j.done, j.total = done, total
            try:
                res = self.engine.synthesize(job.text, job.voice_path, job.settings,
                                             seed=job.seed, cancel=job.cancel,
                                             on_progress=progress)
                with job.lock:                                   # commit boundary
                    if job.cancel.is_set():
                        raise Cancelled("Cancelled.")
                    job.status = "committing"
                job.paths = save_result(res, job.fmt, self.out_dir / f"{job.id}.{job.fmt}",
                                        text=job.text, voice=job.voice_id,
                                        settings=job.settings, srt=job.srt, overwrite=False)
                job.seconds = round(res.seconds, 2)
                self._finish(job, "done")
            except Cancelled:
                self._finish(job, "cancelled")
            except (EngineError, audio.AudioError) as e:
                self._finish(job, "failed", str(e))
            except Exception as e:                      # never kill the worker
                self._finish(job, "failed", f"{type(e).__name__}: {e}")
