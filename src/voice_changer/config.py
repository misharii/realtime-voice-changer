"""Settings loaded from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class TargetVoice:
    name: str
    voice_id: str


@dataclass(frozen=True)
class Settings:
    provider: str
    voices: list[TargetVoice]
    input_device: str
    output_device: str
    monitoring: bool
    monitor_device: str
    vad_threshold: float
    vad_silence_ms: int
    vad_min_phrase_ms: int
    vad_preroll_ms: int
    vad_max_phrase_ms: int
    transform_timeout_s: float
    env: dict[str, str]  # raw values, for provider-specific keys


def parse_voices(raw: str) -> list[TargetVoice]:
    voices = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, voice_id = entry.partition(":")
        if not sep or not name.strip() or not voice_id.strip():
            raise ValueError(f"VOICES entry must be name:id, got {entry!r}")
        voices.append(TargetVoice(name.strip(), voice_id.strip()))
    return voices


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or Path.cwd() / ".env", override=True)
    env = dict(os.environ)

    def get(key: str, default: str = "") -> str:
        return env.get(key, default).strip()

    voices = parse_voices(get("VOICES"))
    if not voices:
        raise ValueError("VOICES is empty; add at least one name:id pair to .env")

    return Settings(
        provider=get("PROVIDER", "fish").lower(),
        voices=voices,
        input_device=get("INPUT_DEVICE"),
        output_device=get("OUTPUT_DEVICE", "CABLE Input"),
        monitoring=_bool(get("MONITORING", "false")),
        monitor_device=get("MONITOR_DEVICE"),
        vad_threshold=float(get("VAD_THRESHOLD", "0.5")),
        vad_silence_ms=int(get("VAD_SILENCE_MS", "400")),
        vad_min_phrase_ms=int(get("VAD_MIN_PHRASE_MS", "250")),
        vad_preroll_ms=int(get("VAD_PREROLL_MS", "200")),
        vad_max_phrase_ms=int(get("VAD_MAX_PHRASE_MS", "15000")),
        transform_timeout_s=float(get("TRANSFORM_TIMEOUT_S", "3")),
        env=env,
    )
