# Cascade Voice Transformer (STT → TTS) behind an audio-in/audio-out seam

Fish Audio offers no real-time speech-to-speech conversion, only ASR and TTS, so the Fish Voice Transformer transcribes each Phrase and re-synthesises it in the Target Voice. This costs roughly 1–1.2 s per Phrase and discards the Speaker's prosody, timing and non-speech sounds. We accepted that in order to use Fish voices, and rejected local RVC-style conversion (~200 ms, keeps intonation) because it needs a GPU, needs per-voice trained models and can't use Fish voices.

The provider seam is placed at the Voice Transformer level (Phrase audio in → Target Voice audio out), not at the TTS level. A true speech-to-speech provider, such as ElevenLabs, can then replace the whole cascade without the capture, VAD or output stages knowing.

Within the cascade, transcription is its own swappable step. Fish ASR needs paid Fish API credit (TTS on `s2.1-pro-free` does not), so the default is OpenRouter `qwen/qwen3-asr-0.6b`. It was the fastest and cheapest model in testing that transcribed Arabic correctly; Deepgram nova-3 returned empty Arabic transcripts when auto-detecting the language.

Failed or timed-out transformations drop the Phrase; the Source Voice is never passed through to the Virtual Mic.
