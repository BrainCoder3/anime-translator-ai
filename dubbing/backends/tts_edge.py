"""Speech synthesis with edge-tts (online voices) and a speaking-rate control."""
from __future__ import annotations

import asyncio
import concurrent.futures
import tempfile
import time
from pathlib import Path

from ..media import FFMPEG, run_command


def rate_string(rate_percent: int) -> str:
    """25 -> '+25%', -10 -> '-10%'."""
    return f"{int(rate_percent):+d}%"


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:   # a loop is already running
        return pool.submit(asyncio.run, coro).result()


class EdgeTTS:
    engine_name = "edge-tts"

    def __init__(self, voice: str = "fr-FR-DeniseNeural", retries: int = 2, sample_rate: int = 24000) -> None:
        self.voice, self.retries, self.sample_rate = voice, retries, sample_rate

    def synthesize(self, text: str, out_wav: str | Path, *, rate_percent: int = 0,
                   voice: str | None = None) -> Path:
        import edge_tts
        if not text.strip():
            raise ValueError("empty text")
        out_wav = Path(out_wav)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            with tempfile.TemporaryDirectory() as tmp:
                mp3 = Path(tmp) / "voice.mp3"
                try:
                    comm = edge_tts.Communicate(text, voice or self.voice, rate=rate_string(rate_percent))
                    _run_async(comm.save(str(mp3)))
                    if not mp3.exists() or mp3.stat().st_size == 0:
                        raise RuntimeError("edge-tts produced no audio")
                    run_command([FFMPEG, "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1",
                                 "-ar", str(self.sample_rate), "-c:a", "pcm_s16le", str(out_wav)])
                    return out_wav
                except Exception as exc:
                    last = exc
            time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"edge-tts failed after {self.retries + 1} attempt(s): {last}")
