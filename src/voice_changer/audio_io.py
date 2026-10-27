"""Device lookup, mic capture and the playback queue (Virtual Mic / Monitoring)."""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from .phrases import FRAME_SAMPLES, SAMPLE_RATE

# WASAPI gives the lowest shared-mode latency; auto_convert lets it resample for us.
HOSTAPI_PREFERENCE = ["Windows WASAPI", "Windows DirectSound", "MME"]


def _hostapi_name(index: int) -> str:
    return sd.query_hostapis(index)["name"]


def find_device(name: str, kind: str) -> tuple[int | None, object | None]:
    """Return (device index, extra_settings) for a name substring; None = system default."""
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    if not name:
        default = sd.default.device[0 if kind == "input" else 1]
        return (default if default is not None and default >= 0 else None), None

    matches = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d[channels_key] > 0 and name.lower() in d["name"].lower()
    ]
    if not matches:
        available = sorted({d["name"] for d in sd.query_devices() if d[channels_key] > 0})
        raise RuntimeError(
            f"No {kind} device matching {name!r}. Available:\n  " + "\n  ".join(available)
        )

    def rank(item: tuple[int, dict]) -> int:
        api = _hostapi_name(item[1]["hostapi"])
        return HOSTAPI_PREFERENCE.index(api) if api in HOSTAPI_PREFERENCE else 99

    index, dev = min(matches, key=rank)
    extra = None
    if _hostapi_name(dev["hostapi"]) == "Windows WASAPI":
        extra = sd.WasapiSettings(auto_convert=True)
    return index, extra


def device_label(index: int | None) -> str:
    if index is None:
        return "system default"
    d = sd.query_devices(index)
    return f"{d['name']} [{_hostapi_name(d['hostapi'])}]"


class MicCapture:
    """Pushes FRAME_SAMPLES-long float32 mono frames at SAMPLE_RATE into `frames`."""

    def __init__(self, device_name: str):
        self.device, extra = find_device(device_name, "input")
        self.frames: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            latency="low",
            extra_settings=extra,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:
        self.frames.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class PcmPlayer:
    """Plays queued int16 mono PCM in order; outputs silence when empty."""

    def __init__(self, device_name: str, sample_rate: int, kind_label: str):
        self.device, extra = find_device(device_name, "output")
        self.label = kind_label
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = sd.RawOutputStream(
            device=self.device,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            latency="low",
            extra_settings=extra,
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status) -> None:
        n = len(outdata)
        with self._lock:
            take = min(n, len(self._buf))
            outdata[:take] = self._buf[:take]
            del self._buf[:take]
        if take < n:
            outdata[take:] = b"\x00" * (n - take)

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    @property
    def busy(self) -> bool:
        with self._lock:
            return len(self._buf) > 0

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
