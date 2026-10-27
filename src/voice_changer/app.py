"""Entry point: mic → Phrases → Voice Transformer → Virtual Mic (+ optional Monitoring)."""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time

import sounddevice as sd
from pynput import keyboard

from .audio_io import MicCapture, PcmPlayer, device_label
from .config import Settings, TargetVoice, load_settings
from .phrases import Phrase, PhraseSegmenter
from .transformers import TransformError, VoiceTransformer, build_transformer
from .vad import SileroVad

log = logging.getLogger("voice_changer")


class VoiceChanger:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.voices = settings.voices
        self.voice: TargetVoice = self.voices[0]
        self.muted = False
        self._stop = threading.Event()
        self._phrases: queue.Queue[Phrase] = queue.Queue()

        self.transformer: VoiceTransformer = build_transformer(settings)
        self.vad = SileroVad()
        self.segmenter = PhraseSegmenter(
            threshold=settings.vad_threshold,
            silence_ms=settings.vad_silence_ms,
            min_phrase_ms=settings.vad_min_phrase_ms,
            preroll_ms=settings.vad_preroll_ms,
            max_phrase_ms=settings.vad_max_phrase_ms,
        )
        self.mic = MicCapture(settings.input_device)
        rate = self.transformer.output_sample_rate
        self.players = [PcmPlayer(settings.output_device, rate, "Virtual Mic")]
        if settings.monitoring:
            self.players.append(PcmPlayer(settings.monitor_device, rate, "Monitoring"))

    # --- controls ---------------------------------------------------------

    def select_voice(self, n: int) -> None:
        if 0 <= n < len(self.voices):
            self.voice = self.voices[n]
            log.info("Target Voice → %d: %s", n + 1, self.voice.name)

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        log.info("MUTED" if self.muted else "Unmuted")

    def _hotkeys(self) -> keyboard.GlobalHotKeys:
        bindings = {"<ctrl>+<alt>+m": self.toggle_mute}
        for i in range(min(9, len(self.voices))):
            bindings[f"<ctrl>+<alt>+{i + 1}"] = lambda i=i: self.select_voice(i)
        return keyboard.GlobalHotKeys(bindings)

    # --- workers ----------------------------------------------------------

    def _listen(self) -> None:
        silent_frames = 0
        while not self._stop.is_set():
            try:
                frame = self.mic.frames.get(timeout=0.2)
            except queue.Empty:
                continue
            # Exact zeros (not just quiet) means Windows is muting the mic.
            silent_frames = silent_frames + 1 if not frame.any() else 0
            if silent_frames == 150:  # ~5 s
                log.warning("Mic is sending pure silence: is it muted in Windows Sound settings?")
            if self.muted:
                if self.segmenter.speaking or self.segmenter.preroll:
                    self.segmenter.reset()
                    self.vad.reset()
                continue
            phrase = self.segmenter.feed(frame, self.vad(frame))
            if phrase is not None:
                self._phrases.put(phrase)

    def _transform(self) -> None:
        while not self._stop.is_set():
            try:
                phrase = self._phrases.get(timeout=0.2)
            except queue.Empty:
                continue
            voice = self.voice
            started = time.perf_counter()
            first_audio: float | None = None
            try:
                for pcm in self.transformer.transform(phrase, voice):
                    if first_audio is None:
                        first_audio = time.perf_counter()
                    for p in self.players:
                        p.feed(pcm)
            except TransformError as e:
                log.warning("dropped %d ms Phrase: %s", phrase.duration_ms, e)
                continue
            if first_audio is not None:
                log.info(
                    "%s | %d ms Phrase | first audio %d ms after cut (+%d ms silence wait) | queue wait %d ms",
                    voice.name,
                    phrase.duration_ms,
                    (first_audio - phrase.ended_at) * 1000,
                    self.settings.vad_silence_ms,
                    (started - phrase.ended_at) * 1000,
                )

    # --- lifecycle --------------------------------------------------------

    def run(self) -> None:
        log.info("Input:       %s", device_label(self.mic.device))
        for p in self.players:
            log.info("%-12s %s", p.label + ":", device_label(p.device))
        log.info("Voices:      %s", ", ".join(f"{i + 1}={v.name}" for i, v in enumerate(self.voices)))
        log.info("Target Voice → 1: %s", self.voice.name)
        log.info("Hotkeys: Ctrl+Alt+1..9 switch voice, Ctrl+Alt+M mute, Ctrl+C quit")

        self.transformer.warm_up()
        for p in self.players:
            p.start()
        self.mic.start()
        threads = [
            threading.Thread(target=self._listen, daemon=True, name="listen"),
            threading.Thread(target=self._transform, daemon=True, name="transform"),
        ]
        for t in threads:
            t.start()
        hotkeys = self._hotkeys()
        hotkeys.start()
        log.info("Listening...")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop.set()
            hotkeys.stop()
            self.mic.close()
            for p in self.players:
                p.close()
            self.transformer.close()


def list_devices() -> None:
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        kinds = "/".join(k for k, c in (("in", d["max_input_channels"]), ("out", d["max_output_channels"])) if c)
        print(f"{i:3d}  {kinds:6s}  {api:20s}  {d['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time voice changer → VB-Cable")
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.list_devices:
        list_devices()
        return
    VoiceChanger(load_settings()).run()


if __name__ == "__main__":
    main()
