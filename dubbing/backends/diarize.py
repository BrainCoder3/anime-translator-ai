"""Speaker diarization adapters."""
from __future__ import annotations

import math
import os
import wave
from pathlib import Path


class SingleSpeaker:
    """Default: every segment is SPEAKER_00."""

    def diarize(self, wav_path: str | Path) -> tuple[list[dict], dict[str, list[float]]]:
        return [], {}


class PyannoteDiarizer:
    """pyannote turns and per-speaker embeddings, ported from the original pipeline.

    Not tested here (pyannote, torch and a token were unavailable). The token comes from HF_TOKEN.
    """

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1",
                 min_speakers: int = 1, max_speakers: int = 8, num_speakers: int | None = None) -> None:
        self.model, self.min_speakers, self.max_speakers, self.num_speakers = model, min_speakers, max_speakers, num_speakers

    @staticmethod
    def _load_wav(path: str | Path):
        import numpy as np
        import torch
        with wave.open(str(path), "rb") as w:
            sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        audio = audio.reshape(-1, ch).T if ch > 1 else audio[np.newaxis, :]
        return {"waveform": torch.from_numpy(audio), "sample_rate": sr}

    def diarize(self, wav_path: str | Path) -> tuple[list[dict], dict[str, list[float]]]:
        from pyannote.audio import Pipeline
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set (export your Hugging Face token first)")
        pipe = Pipeline.from_pretrained(self.model, token=token)
        audio = self._load_wav(wav_path)
        kwargs = ({"num_speakers": self.num_speakers} if self.num_speakers is not None
                  else {"min_speakers": self.min_speakers, "max_speakers": self.max_speakers})
        out = pipe(audio, **kwargs)

        embeddings: dict[str, list[float]] = {}
        for label, raw in zip(out.speaker_diarization.labels(), out.speaker_embeddings):
            vec = [float(x) for x in raw]
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0 and all(math.isfinite(x) for x in vec):
                embeddings[str(label)] = [x / norm for x in vec]
        turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(s)}
                 for t, s in out.exclusive_speaker_diarization]
        return turns, embeddings
