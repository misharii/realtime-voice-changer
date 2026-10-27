"""Voice Transformers: Phrase audio in → Target Voice audio out.

The rest of the app only knows this interface. A provider can transcribe and
re-synthesise (Fish) or convert speech directly (e.g. ElevenLabs speech-to-speech).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..config import Settings, TargetVoice
from ..phrases import Phrase


class TransformError(Exception):
    """The Phrase could not be transformed and should be dropped."""


class VoiceTransformer(ABC):
    #: Sample rate of the int16 mono PCM yielded by transform().
    output_sample_rate: int

    @abstractmethod
    def transform(self, phrase: Phrase, voice: TargetVoice) -> Iterator[bytes]:
        """Yield int16 little-endian mono PCM chunks as soon as they are available.

        Yielding nothing means there was nothing to say (e.g. no words heard).
        Raise TransformError to drop the Phrase.
        """

    def warm_up(self) -> None:
        """Optional: open connections ahead of the first Phrase."""

    def close(self) -> None:
        pass


def build_transformer(settings: Settings) -> VoiceTransformer:
    if settings.provider == "fish":
        from .fish import FishTransformer

        return FishTransformer(settings)
    raise ValueError(f"Unknown PROVIDER {settings.provider!r} (supported: fish)")
