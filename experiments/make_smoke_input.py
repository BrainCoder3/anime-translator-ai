"""Japanese test clip: three original sentences, one with a number, separated by silences.

    python experiments/make_smoke_input.py OUTPUT.wav
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dubbing import media  # noqa: E402

SENTENCES = ["こんにちは、今日はとてもいい天気ですね。",
             "明日の朝八時に、駅の前で会いましょう。",
             "ありがとうございます、本当に助かりました。"]
VOICE = "ja-JP-NanamiNeural"
GAP_MS = 700


async def _say(text: str, path: pathlib.Path) -> None:
    import edge_tts
    await edge_tts.Communicate(text, VOICE).save(str(path))


def build(out_wav: pathlib.Path) -> pathlib.Path:
    sr, parts = 16000, []
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(SENTENCES):
            mp3 = pathlib.Path(tmp) / f"{i}.mp3"
            asyncio.run(_say(text, mp3))
            wav = media.to_wav(mp3, pathlib.Path(tmp) / f"{i}.wav", sample_rate=sr)
            parts += [np.zeros(int(sr * GAP_MS / 1000), dtype=np.float32), media.read_wav(wav)[1]]
    parts.append(np.zeros(int(sr * GAP_MS / 1000), dtype=np.float32))
    return media.write_wav(out_wav, sr, np.concatenate(parts))


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_input_ja.wav")
    print("written", build(target), media.duration_ms(target), "ms")
