"""Per-segment audit: numbers only, no dialogue and no file paths. Same layout as the original audit."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__


@dataclass
class SegmentAudit:
    id: int
    character_id: str
    engine: str
    target_ms: int
    final_duration_ms: int
    overflow_ms: int
    rate_applied: int = 0
    adapted: bool = False
    attempts: list = field(default_factory=list)


def build_audit(records: list[SegmentAudit], config: dict, extra: dict | None = None) -> dict:
    over = [r.id for r in records if r.overflow_ms > 0]
    return {"version": __version__, "config": config, "overflow_segments": over,
            "segments": [asdict(r) for r in records], **(extra or {})}


def save_audit(audit: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")


def load_audit(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
