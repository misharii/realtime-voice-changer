# Realtime Voice Changer

Mic → Phrase detection (Silero VAD) → Voice Transformer (STT → Fish TTS) → VB-Cable virtual mic.
STT is OpenRouter (`qwen/qwen3-asr-0.6b`) by default; `STT_PROVIDER=fish` switches to Fish ASR.
Terms: [GLOSSARY.md](GLOSSARY.md). Design: [docs/adr](docs/adr).

## Setup

1. Install VB-Cable: run `VBCABLE_Setup_x64.exe` **as Administrator**, then reboot.
2. Set `OPENROUTER_API_KEY` (or add Fish **API** credit and set `STT_PROVIDER=fish`).
3. Put your Target Voices in `.env` → `VOICES=name:reference_id,...`
4. `uv sync`

## Run

```
uv run voice-changer                 # start
uv run voice-changer --list-devices  # find device names for .env
```

In Discord/Teams/etc. select **CABLE Output (VB-Audio Virtual Cable)** as the microphone.

Hotkeys: `Ctrl+Alt+1..9` switch Target Voice, `Ctrl+Alt+M` mute, `Ctrl+C` quit.

## Adding a provider (e.g. ElevenLabs)

Implement `VoiceTransformer` in `src/voice_changer/transformers/<name>.py`
(Phrase in → int16 PCM chunks out), register it in `build_transformer`, set `PROVIDER=<name>`.
