import threading
import numpy as np
import sounddevice as sd


class RecorderError(RuntimeError):
    pass


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

    def start(self):
        try:
            info = sd.query_devices(kind="input")
            self.sample_rate = int(info["default_samplerate"])
            self._frames = []
            self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                          dtype="float32", callback=self._cb)
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise RecorderError(f"Could not open microphone: {e}") from e

    def _cb(self, indata, frames, time_info, status):
        with self._lock:
            self._frames.append(indata.copy())
        self.level = float(np.abs(indata).max())

    def stop(self) -> np.ndarray:
        if not self._stream:
            raise RecorderError("Not recording.")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.level = 0.0
        with self._lock:
            if not self._frames:
                raise RecorderError("No audio captured.")
            return np.concatenate(self._frames, axis=0)[:, 0]

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(f) for f in self._frames) / float(self.sample_rate)
