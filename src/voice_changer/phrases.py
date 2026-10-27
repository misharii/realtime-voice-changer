"""Cuts a stream of mic frames into Phrases using per-frame speech probabilities."""

from __future__ import annotations

import io
import time
import wave
from collections import deque
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms; the frame size Silero VAD expects at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


@dataclass
class Phrase:
    audio: np.ndarray  # float32 mono, SAMPLE_RATE
    ended_at: float = field(default_factory=time.perf_counter)

    @property
    def duration_ms(self) -> int:
        return len(self.audio) * 1000 // SAMPLE_RATE


def phrase_to_wav(phrase: Phrase) -> bytes:
    pcm = (np.clip(phrase.audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


class PhraseSegmenter:
    """Stateful: feed it (frame, speech_prob) pairs, get completed Phrases back."""

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 400,
        min_phrase_ms: int = 250,
        preroll_ms: int = 200,
        max_phrase_ms: int = 15_000,
    ):
        self.threshold = threshold
        # Hysteresis: once speaking, only a clearly lower prob counts as silence.
        self.release = max(threshold - 0.15, 0.05)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_phrase_ms // FRAME_MS)
        self.max_frames = max(1, max_phrase_ms // FRAME_MS)
        self.preroll: deque[np.ndarray] = deque(maxlen=max(0, preroll_ms // FRAME_MS))
        self._frames: list[np.ndarray] = []
        self._speech_frames = 0
        self._silent_run = 0
        self.speaking = False

    def reset(self) -> None:
        self.preroll.clear()
        self._frames = []
        self._speech_frames = 0
        self._silent_run = 0
        self.speaking = False

    def feed(self, frame: np.ndarray, prob: float) -> Phrase | None:
        if not self.speaking:
            if prob >= self.threshold:
                self.speaking = True
                self._frames = [*self.preroll, frame]
                self._speech_frames = 1
                self._silent_run = 0
                self.preroll.clear()
            else:
                self.preroll.append(frame)
            return None

        self._frames.append(frame)
        if prob >= self.release:
            self._speech_frames += 1
            self._silent_run = 0
        else:
            self._silent_run += 1

        if self._silent_run >= self.silence_frames or len(self._frames) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> Phrase | None:
        frames, speech = self._frames, self._speech_frames
        # Trim most of the trailing silence; keep ~2 frames so words don't end abruptly.
        trailing = max(0, self._silent_run - 2)
        if trailing:
            frames = frames[:-trailing]
        self.reset()
        if speech < self.min_speech_frames:
            return None
        return Phrase(np.concatenate(frames).astype(np.float32, copy=False))
