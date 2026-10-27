"""Speech-to-text: Phrase audio → transcript. Used by cascade Voice Transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from ..config import Settings
from ..phrases import Phrase


class SpeechToText(ABC):
    _client: httpx.Client

    @abstractmethod
    def transcribe(self, phrase: Phrase) -> str:
        """Return the transcript ('' if no words). Raise TransformError to drop the Phrase."""

    def warm_up(self) -> None:
        try:
            self._client.get("/")  # any response leaves a TLS connection in the pool
        except httpx.HTTPError:
            pass

    def close(self) -> None:
        self._client.close()


def build_stt(settings: Settings) -> SpeechToText:
    provider = settings.env.get("STT_PROVIDER", "openrouter").strip().lower()
    if provider == "openrouter":
        from .openrouter import OpenRouterStt

        return OpenRouterStt(settings)
    if provider == "fish":
        from .fish import FishAsr

        return FishAsr(settings)
    raise ValueError(f"Unknown STT_PROVIDER {provider!r} (supported: openrouter, fish)")
