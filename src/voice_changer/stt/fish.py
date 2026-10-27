"""Fish Audio ASR (needs paid Fish API credit)."""

from __future__ import annotations

import httpx

from ..config import Settings
from ..phrases import Phrase, phrase_to_wav
from ..transformers import TransformError
from . import SpeechToText


class FishAsr(SpeechToText):
    def __init__(self, settings: Settings):
        env = settings.env
        api_key = env.get("FISH_API_KEY", "").strip()
        if not api_key:
            raise ValueError("FISH_API_KEY is missing from .env")
        self.language = env.get("FISH_ASR_LANGUAGE", "").strip() or None
        t = settings.transform_timeout_s
        self._client = httpx.Client(
            base_url=env.get("FISH_BASE_URL", "https://api.fish.audio").strip(),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(t, connect=t),
        )

    def transcribe(self, phrase: Phrase) -> str:
        data = {"ignore_timestamps": "true"}
        if self.language:
            data["language"] = self.language
        try:
            r = self._client.post(
                "/v1/asr",
                files={"audio": ("phrase.wav", phrase_to_wav(phrase), "audio/wav")},
                data=data,
            )
        except httpx.HTTPError as e:
            raise TransformError(f"ASR request failed: {e!r}") from e
        if r.status_code != 200:
            raise TransformError(f"ASR HTTP {r.status_code}: {r.text[:200]}")
        return (r.json().get("text") or "").strip()
