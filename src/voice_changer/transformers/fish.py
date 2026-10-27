"""Cascade: transcribe the Phrase (STT_PROVIDER), then stream Fish TTS in the Target Voice."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import httpx

from ..config import Settings, TargetVoice
from ..phrases import Phrase
from ..stt import build_stt
from . import TransformError, VoiceTransformer

log = logging.getLogger(__name__)


class FishTransformer(VoiceTransformer):
    def __init__(self, settings: Settings):
        env = settings.env
        api_key = env.get("FISH_API_KEY", "").strip()
        if not api_key:
            raise ValueError("FISH_API_KEY is missing from .env")
        self.stt = build_stt(settings)
        self.model = env.get("FISH_TTS_MODEL", "s2.1-pro-free").strip()
        self.latency = env.get("FISH_TTS_LATENCY", "balanced").strip()
        self.output_sample_rate = int(env.get("FISH_TTS_SAMPLE_RATE", "44100"))
        t = settings.transform_timeout_s
        self._client = httpx.Client(
            base_url=env.get("FISH_BASE_URL", "https://api.fish.audio").strip(),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(t, connect=t),
        )

    def warm_up(self) -> None:
        self.stt.warm_up()
        try:
            self._client.get("/")  # any response leaves a TLS connection in the pool
        except httpx.HTTPError:
            pass

    def speak(self, text: str, voice: TargetVoice) -> Iterator[bytes]:
        body = {
            "text": text,
            "format": "pcm",
            "sample_rate": self.output_sample_rate,
            "latency": self.latency,
            "normalize": True,
        }
        if voice.voice_id:
            body["reference_id"] = voice.voice_id
        try:
            with self._client.stream(
                "POST", "/v1/tts", json=body, headers={"model": self.model}
            ) as r:
                if r.status_code != 200:
                    r.read()
                    raise TransformError(f"TTS HTTP {r.status_code}: {r.text[:200]}")
                leftover = b""
                for chunk in r.iter_bytes():
                    chunk = leftover + chunk
                    cut = len(chunk) - (len(chunk) % 2)  # keep int16 samples whole
                    leftover = chunk[cut:]
                    if cut:
                        yield chunk[:cut]
        except httpx.HTTPError as e:
            raise TransformError(f"TTS request failed: {e!r}") from e

    def transform(self, phrase: Phrase, voice: TargetVoice) -> Iterator[bytes]:
        t0 = time.perf_counter()
        text = self.stt.transcribe(phrase)
        stt_ms = (time.perf_counter() - t0) * 1000
        if not text:
            log.info("(no words heard, STT %d ms)", stt_ms)
            return
        log.info("heard (STT %d ms): %s", stt_ms, text)
        yield from self.speak(text, voice)

    def close(self) -> None:
        self.stt.close()
        self._client.close()
