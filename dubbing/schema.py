"""Segment data model."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Segment:
    id: int
    start_ms: int
    end_ms: int
    text_src: str
    speaker: str = "SPEAKER_00"
    text_fr: str = ""

    def __post_init__(self) -> None:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"segment {self.id}: end_ms must be greater than start_ms")

    @property
    def target_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(**{k: data[k] for k in ("id", "start_ms", "end_ms", "text_src") if k in data},
                   speaker=data.get("speaker", "SPEAKER_00"), text_fr=data.get("text_fr", ""))
