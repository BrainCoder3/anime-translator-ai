"""Speech recognition with faster-whisper (CPU, int8 by default)."""
from __future__ import annotations

from pathlib import Path

from ..media import duration_ms


class FasterWhisperASR:
    """With precise_timestamps, each utterance spans its first to last word.

    Segment-level timestamps came out in blocks of 5 s on a 15 s test clip, far off the real speech.
    Word timestamps cost about twice the time and bring the error under about 0.4 s.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8",
                 beam_size: int = 5, precise_timestamps: bool = True) -> None:
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self.beam_size, self.precise_timestamps = beam_size, precise_timestamps
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, wav_path: str | Path, language: str) -> list[tuple[int, int, str]]:
        """Return (start_ms, end_ms, text). Ends are clamped to the file duration; empty or <50 ms spans are dropped."""
        total = duration_ms(wav_path)
        segments, _ = self._load().transcribe(str(wav_path), language=language, beam_size=self.beam_size,
                                              word_timestamps=self.precise_timestamps)
        out: list[tuple[int, int, str]] = []
        for s in segments:
            text = s.text.strip()
            words = getattr(s, "words", None) if self.precise_timestamps else None
            first, last = (words[0].start, words[-1].end) if words else (s.start, s.end)
            start, end = int(round(first * 1000)), min(int(round(last * 1000)), total)
            if text and end - start >= 50:
                out.append((start, end, text))
        return out
