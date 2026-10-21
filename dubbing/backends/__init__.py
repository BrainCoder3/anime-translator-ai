"""Engine protocols used by the pipeline.

Tested: FasterWhisperASR, OllamaTranslator, EdgeTTS, SingleSpeaker. Not tested: PyannoteDiarizer.
Voice cloning lives in the full prototype (src/), outside this package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ASR(Protocol):
    def transcribe(self, wav_path: str | Path, language: str) -> list[tuple[int, int, str]]:
        """Return (start_ms, end_ms, text) utterances."""


class Translator(Protocol):
    def translate(self, text: str, source_lang: str, target_lang: str, *, previous: str = "",
                  next_: str = "") -> str: ...

    def ask(self, prompt: str) -> str:
        """Raw LLM answer in JSON mode."""


class TTS(Protocol):
    engine_name: str

    def synthesize(self, text: str, out_wav: str | Path, *, rate_percent: int = 0,
                   voice: str | None = None) -> Path: ...


class Diarizer(Protocol):
    def diarize(self, wav_path: str | Path) -> tuple[list[dict], dict[str, list[float]]]:
        """Return (turns [{start, end, speaker}] in seconds, {speaker: embedding})."""
