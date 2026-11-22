# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Leon Priest (GitHub: 7h3v01d)
"""Microphone capture with a live level meter. sounddevice is imported lazily so the
rest of the app works on machines without PortAudio."""
import threading
from typing import Optional

import numpy as np


class RecorderError(RuntimeError):
    pass


def list_inputs() -> list[tuple[int, str]]:
    """(device index, name) for every input device; [] if audio is unavailable."""
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d.get("max_input_channels", 0) > 0]
    except Exception:
        return []


class Recorder:
    def __init__(self):
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self.level = 0.0
        self.sample_rate = 44100

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self, device: Optional[int] = None):
        if self._stream is not None:
            raise RecorderError("Already recording.")
        try:
            import sounddevice as sd
            info = sd.query_devices(device, kind="input")
            self.sample_rate = int(info["default_samplerate"])
            with self._lock:
                self._frames = []
            self._stream = sd.InputStream(device=device, samplerate=self.sample_rate,
                                          channels=1, dtype="float32", callback=self._cb)
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise RecorderError(f"Could not open the microphone: {e}") from e

    def _cb(self, indata, frames, time_info, status):
        with self._lock:
            self._frames.append(indata.copy())
        self.level = float(np.abs(indata).max())

    def stop(self) -> np.ndarray:
        if not self._stream:
            raise RecorderError("Not recording.")
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            self.level = 0.0
        with self._lock:
            if not self._frames:
                raise RecorderError("No audio was captured.")
            return np.concatenate(self._frames, axis=0)[:, 0]

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(f) for f in self._frames) / float(self.sample_rate)
