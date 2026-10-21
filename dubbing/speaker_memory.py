"""Character memory across episodes, built from speaker embeddings.

Speakers are compared with the characters not yet claimed in the episode (one-to-one). With best
cosine s and margin m (best minus runner-up, -1 for the runner-up if alone):
  s >= match and m >= margin -> MATCHED (centroid updated once per episode)
  s < new_character          -> NEW
  otherwise                  -> AMBIGUOUS (temporary voice, memory untouched)
No unclaimed character left -> AMBIGUOUS_DUPLICATE. An empty memory makes every speaker NEW.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .config import VOICE_POOL, SpeakerThresholds

Vector = Sequence[float]


class Status(str, Enum):
    NEW = "NEW"
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    AMBIGUOUS_DUPLICATE = "AMBIGUOUS_DUPLICATE"


@dataclass
class Assignment:
    speaker: str
    status: Status
    voice: str
    similarity: float
    margin: float | None = None
    character: str | None = None
    best_candidate: str | None = None


def normalize(vec: Vector) -> list[float] | None:
    """Unit-length copy, or None for empty, non-finite or zero vectors."""
    if not vec or any(not math.isfinite(x) for x in vec):
        return None
    norm = math.sqrt(sum(x * x for x in vec))
    return None if norm == 0 else [x / norm for x in vec]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine for vectors that are already unit length."""
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def decide(best: float, margin: float, th: SpeakerThresholds) -> Status:
    if best >= th.match and margin >= th.margin:
        return Status.MATCHED
    if best < th.new_character:
        return Status.NEW
    return Status.AMBIGUOUS


class CharacterMemory:
    def __init__(self, thresholds: SpeakerThresholds | None = None,
                 voice_pool: Sequence[str] = VOICE_POOL) -> None:
        self.th = thresholds or SpeakerThresholds()
        self.voice_pool = tuple(voice_pool)
        self.meta: dict[str, dict] = {}
        self.embeddings: dict[str, list[float]] = {}

    def _next_id(self) -> str:
        nums = []
        for cid in self.meta:
            try:
                nums.append(int(cid.split("_")[-1]))
            except ValueError:
                pass
        return f"CHARACTER_{(max(nums) + 1 if nums else 1):03d}"

    def _unused_voice(self, current: Mapping[str, str]) -> str:
        used = {m.get("voice") for m in self.meta.values() if m.get("voice")} | set(current.values())
        for voice in self.voice_pool:
            if voice not in used:
                return voice
        return self.voice_pool[len(current) % len(self.voice_pool)]

    def _create(self, speaker: str, emb: list[float], episode: str, current: dict[str, str]) -> tuple[str, str]:
        cid, voice = self._next_id(), self._unused_voice(current)
        self.meta[cid] = {"voice": voice, "enrolled_from": speaker, "embedding_count": 1,
                          "seen_episodes": [episode]}
        self.embeddings[cid] = emb
        return cid, voice

    def _update(self, cid: str, emb: list[float], episode: str) -> None:
        data = self.meta[cid]
        if episode in data["seen_episodes"]:     # re-running an episode must not count twice
            return
        n = int(data["embedding_count"])
        centroid = normalize([(old * n + new) / (n + 1) for old, new in zip(self.embeddings[cid], emb)])
        if centroid is not None:
            self.embeddings[cid] = centroid
        data["seen_episodes"].append(episode)
        data["embedding_count"] = n + 1

    def recognize(self, current: Mapping[str, Vector], episode: str) -> dict[str, Assignment]:
        """Assign each speaker of the episode to a character. Updates the memory on NEW and MATCHED."""
        started_empty = not self.embeddings      # state before this episode, not after the first NEW
        claimed: set[str] = set()
        voices: dict[str, str] = {}
        out: dict[str, Assignment] = {}

        for speaker, raw in current.items():
            emb = normalize(raw)
            if emb is None:
                raise ValueError(f"{speaker}: invalid embedding")

            if started_empty:
                cid, voice = self._create(speaker, emb, episode, voices)
                claimed.add(cid)
                out[speaker] = Assignment(speaker, Status.NEW, voice, 1.0, None, cid)
                voices[speaker] = voice
                continue

            scores = sorted(((cosine(emb, e), cid) for cid, e in self.embeddings.items()), reverse=True)
            available = [(s, cid) for s, cid in scores if cid not in claimed]

            if not available:
                voice = self._unused_voice(voices)
                out[speaker] = Assignment(speaker, Status.AMBIGUOUS_DUPLICATE, voice, scores[0][0],
                                          None, None, scores[0][1])
                voices[speaker] = voice
                continue

            best, best_id = available[0]
            second = available[1][0] if len(available) > 1 else -1.0
            margin = best - second
            status = decide(best, margin, self.th)

            if status is Status.MATCHED:
                voice = self.meta[best_id]["voice"]
                claimed.add(best_id)
                self._update(best_id, emb, episode)
                out[speaker] = Assignment(speaker, status, voice, best, margin, best_id)
            elif status is Status.NEW:
                cid, voice = self._create(speaker, emb, episode, voices)
                claimed.add(cid)
                out[speaker] = Assignment(speaker, status, voice, best, margin, cid)
            else:
                voice = self._unused_voice(voices)
                out[speaker] = Assignment(speaker, status, voice, best, margin, None, best_id)
            voices[speaker] = voice
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"meta": self.meta, "embeddings": self.embeddings,
                                          "thresholds": asdict(self.th)}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterMemory":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mem = cls(SpeakerThresholds(**data["thresholds"]))
        mem.meta, mem.embeddings = data["meta"], data["embeddings"]
        return mem
