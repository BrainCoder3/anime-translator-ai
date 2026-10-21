"""Markdown table helper."""
from __future__ import annotations

from typing import Iterable, Mapping


def _fmt(value, floatfmt: str) -> str:
    if isinstance(value, float):
        return format(value, floatfmt)
    return str(value)


def to_markdown_table(rows: Iterable[Mapping], columns: list[str], floatfmt: str = ".3f") -> str:
    """Rows (dicts) as a GitHub markdown table."""
    rows = list(rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c, ""), floatfmt) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])
