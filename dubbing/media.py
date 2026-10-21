"""Audio helpers: ffmpeg for decoding, tempo and muxing; numpy and the wave module for the rest."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {tail}")
    return proc


def to_wav(src: str | Path, dst: str | Path, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Decode any audio or video file to PCM16 WAV."""
    run_command([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", str(channels),
                 "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)])
    return Path(dst)


def duration_ms(path: str | Path) -> int:
    out = run_command([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                       "-of", "default=noprint_wrappers=1:nokey=1", str(path)]).stdout.strip()
    return int(round(float(out) * 1000))


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """PCM16 WAV to (sample rate, float32 mono in [-1, 1])."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return sr, data


def write_wav(path: str | Path, sample_rate: int, data: np.ndarray) -> Path:
    """Mono PCM16 WAV from float audio in [-1, 1] (values outside are clipped)."""
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return Path(path)


def wav_duration_ms(path: str | Path) -> int:
    with wave.open(str(path), "rb") as w:
        return int(round(w.getnframes() / w.getframerate() * 1000))


def trim_silence(data: np.ndarray, sample_rate: int, pad_ms: int = 50, rel_db: float = 18.0,
                 frame_ms: int = 10) -> np.ndarray:
    """Cut leading and trailing silence. A frame is silent if it is rel_db below the clip's mean level."""
    if data.size == 0 or not np.any(data):
        return data
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n = data.size // frame
    if n == 0:
        return data
    frames = data[: n * frame].reshape(n, frame)
    frame_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
    mean_db = 20 * np.log10(np.sqrt((data ** 2).mean()) + 1e-12)
    loud = np.flatnonzero(frame_db > mean_db - rel_db)
    if loud.size == 0:
        return data
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, loud[0] * frame - pad)
    end = min(data.size, (loud[-1] + 1) * frame + pad)
    return data[start:end]


def tempo(src: str | Path, dst: str | Path, factor: float) -> Path:
    """Change speed without changing pitch. ffmpeg atempo only accepts 0.5-2.0 per filter, so it is chained."""
    if factor <= 0:
        raise ValueError("factor must be > 0")
    filters: list[float] = []
    f = factor
    while f > 2.0:
        filters.append(2.0)
        f /= 2.0
    while f < 0.5:
        filters.append(0.5)
        f /= 0.5
    filters.append(f)
    chain = ",".join(f"atempo={x:.6f}" for x in filters)
    run_command([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-filter:a", chain,
                 "-c:a", "pcm_s16le", str(dst)])
    return Path(dst)


def mix_timeline(clips: Sequence[tuple[int, np.ndarray]], sample_rate: int, total_ms: int) -> np.ndarray:
    """Place (start_ms, clip) pairs on a silent canvas. It grows to fit; an overloaded mix is rescaled, not clipped."""
    total = int(sample_rate * total_ms / 1000)
    for start_ms, clip in clips:
        total = max(total, int(sample_rate * start_ms / 1000) + clip.size)
    canvas = np.zeros(total, dtype=np.float32)
    for start_ms, clip in clips:
        s = int(sample_rate * start_ms / 1000)
        canvas[s:s + clip.size] += clip
    peak = float(np.abs(canvas).max()) if canvas.size else 0.0
    return canvas / peak if peak > 1.0 else canvas


def srt_time(ms: int) -> str:
    h, rem = divmod(int(ms), 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def write_srt(entries: Sequence[tuple[int, int, str]], path: str | Path) -> Path:
    lines: list[str] = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines += [str(i), f"{srt_time(start)} --> {srt_time(end)}", text.strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def mux(video: str | Path, audio: str | Path, out: str | Path, srt: str | Path | None = None) -> Path:
    """Replace the audio track of a video, optionally adding subtitles."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio)]
    if srt:
        cmd += ["-i", str(srt)]
    cmd += ["-map", "0:v", "-map", "1:a"] + (["-map", "2:0", "-c:s", "mov_text"] if srt else [])
    cmd += ["-c:v", "copy", "-c:a", "aac", "-shortest", str(out)]
    run_command(cmd)
    return Path(out)
