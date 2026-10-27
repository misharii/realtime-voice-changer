import numpy as np
import pytest

from voice_changer.config import TargetVoice, parse_voices
from voice_changer.phrases import FRAME_MS, FRAME_SAMPLES, PhraseSegmenter


def frame(value: float = 0.0) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.float32)


def run(seg: PhraseSegmenter, probs: list[float]):
    return [p for p in (seg.feed(frame(i), prob) for i, prob in enumerate(probs)) if p is not None]


def seg(**kw) -> PhraseSegmenter:
    defaults = dict(threshold=0.5, silence_ms=4 * FRAME_MS, min_phrase_ms=3 * FRAME_MS, preroll_ms=2 * FRAME_MS)
    return PhraseSegmenter(**{**defaults, **kw})


def test_phrase_ends_after_silence_and_includes_preroll():
    phrases = run(seg(), [0.0] * 5 + [0.9] * 10 + [0.0] * 4)
    assert len(phrases) == 1
    # 2 preroll + 10 speech + 2 kept trailing frames
    assert len(phrases[0].audio) == 14 * FRAME_SAMPLES
    assert phrases[0].audio[0] == 3.0  # starts at the first preroll frame


def test_short_blip_is_ignored():
    assert run(seg(), [0.9, 0.9] + [0.0] * 6) == []


def test_brief_dip_does_not_split_phrase():
    phrases = run(seg(), [0.9] * 5 + [0.0] * 2 + [0.9] * 5 + [0.0] * 4)
    assert len(phrases) == 1


def test_max_length_forces_cut():
    phrases = run(seg(max_phrase_ms=8 * FRAME_MS), [0.9] * 20)
    assert len(phrases) == 2


def test_parse_voices():
    assert parse_voices(" a:123 , b:456,") == [TargetVoice("a", "123"), TargetVoice("b", "456")]
    with pytest.raises(ValueError):
        parse_voices("noid")
