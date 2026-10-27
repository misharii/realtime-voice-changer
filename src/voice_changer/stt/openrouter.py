"""OpenRouter /audio/transcriptions (any of its transcription models)."""

from __future__ import annotations

import base64

import httpx

from ..config import Settings
from ..phrases import Phrase, phrase_to_wav
from ..transformers import TransformError
from . import SpeechToText


class OpenRouterStt(SpeechToText):
    def __init__(self, settings: Settings):
        env = settings.env
        api_key = env.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from .env")
        self.model = env.get("OPENROUTER_STT_MODEL", "qwen/qwen3-asr-0.6b").strip()
        self.language = env.get("OPENROUTER_STT_LANGUAGE", "").strip() or None
        t = settings.transform_timeout_s
        self._client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "realtime-voice-changer"},
            timeout=httpx.Timeout(t, connect=t),
        )

    def transcribe(self, phrase: Phrase) -> str:
        body = {
            "model": self.model,
            "input_audio": {"data": base64.b64encode(phrase_to_wav(phrase)).decode(), "format": "wav"},
        }
        if self.language:
            body["language"] = self.language
        try:
            r = self._client.post("/audio/transcriptions", json=body)
        except httpx.HTTPError as e:
            raise TransformError(f"STT request failed: {e!r}") from e
        if r.status_code != 200:
            raise TransformError(f"STT HTTP {r.status_code}: {r.text[:200]}")
        return (r.json().get("text") or "").strip()
