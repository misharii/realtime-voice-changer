"""Silero VAD (v5) via onnxruntime, without pulling in torch."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .phrases import FRAME_SAMPLES, SAMPLE_RATE

MODEL_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
CONTEXT_SAMPLES = 64  # v5 expects the previous frame's last 64 samples prepended


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Silero VAD model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class SileroVad:
    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(ensure_model()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one FRAME_SAMPLES-long float32 frame."""
        assert frame.shape == (FRAME_SAMPLES,), frame.shape
        x = np.concatenate([self._context, frame[np.newaxis, :]], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0][0])
