# Glossary

**Speaker**: The person talking into the physical microphone.

**Source Voice**: The Speaker's own voice as captured from the input device.

**Target Voice**: The voice the Speaker wants to be heard as. Identified by a provider-specific voice id (for Fish Audio, a `reference_id`). Selectable at runtime from a list of saved Target Voices.
_Avoid_: "selected audio", "voice model" (ambiguous with ML models)

**Phrase**: A continuous stretch of speech from the Speaker, bounded by silence. The unit that gets transformed.

**Voice Transformer**: Anything that turns a Phrase in the Source Voice into audio in the Target Voice. May work by transcribing and re-synthesising, or by converting speech directly.

**Virtual Mic**: The audio device other apps (Discord, Teams, games) select as their microphone; it carries only transformed audio.

**Monitoring**: Optionally playing the transformed audio back to the Speaker's own headphones as well.

**Mute**: A state in which no Phrases are captured or sent.
