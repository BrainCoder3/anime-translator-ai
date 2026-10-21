"""Matplotlib figures (headless, colour-blind-safe palette)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import stats  # noqa: E402

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def timing_scatter(segments: Sequence[Mapping], path, tol_ms: float = 200, title: str = "") -> Path:
    """Dubbed duration against source slot; points above the diagonal overrun."""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    engines = sorted({str(s.get("engine", "?")) for s in segments})
    top = max(max(s["target_ms"], s["final_duration_ms"]) for s in segments) * 1.08
    xs = [0, top]
    ax.plot(xs, xs, color="#444", lw=1, label="perfect fit")
    ax.fill_between(xs, [x - tol_ms for x in xs], [x + tol_ms for x in xs], color="#444", alpha=0.12,
                    label=f"±{tol_ms:.0f} ms")
    for color, eng in zip(PALETTE, engines):
        pts = [s for s in segments if str(s.get("engine", "?")) == eng]
        ax.scatter([s["target_ms"] for s in pts], [s["final_duration_ms"] for s in pts], s=42, color=color,
                   edgecolor="white", label=f"{eng} (n={len(pts)})", zorder=3)
        if len(pts) >= 5:
            f = stats.linear_fit([s["target_ms"] for s in pts], [s["final_duration_ms"] for s in pts])
            lo, hi = min(s["target_ms"] for s in pts), max(s["target_ms"] for s in pts)
            ax.plot([lo, hi], [f["intercept"] + f["slope"] * lo, f["intercept"] + f["slope"] * hi],
                    color=color, ls="--", lw=1.2, zorder=2)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("source slot (ms)")
    ax.set_ylabel("dubbed duration (ms)")
    ax.set_title(title or "Duration fit per segment")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return _save(fig, path)


def overrun_bars(segments: Sequence[Mapping], path, tol_ms: float = 0, title: str = "") -> Path:
    """Dubbed minus slot, per segment; positive means too long."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    engines = sorted({str(s.get("engine", "?")) for s in segments})
    colors = {e: PALETTE[i % len(PALETTE)] for i, e in enumerate(engines)}
    ids = [s.get("id", i + 1) for i, s in enumerate(segments)]
    err = [s["final_duration_ms"] - s["target_ms"] for s in segments]
    ax.bar(range(len(segments)), err, color=[colors[str(s.get("engine", "?"))] for s in segments], width=0.7)
    ax.axhline(0, color="#222", lw=0.8)
    if tol_ms:
        ax.axhspan(-tol_ms, tol_ms, color="#444", alpha=0.1, label=f"±{tol_ms:.0f} ms")
    ax.set_xticks(range(len(segments)))
    ax.set_xticklabels(ids)
    ax.set_xlabel("segment id")
    ax.set_ylabel("dubbed − slot (ms)")
    ax.set_title(title or "Duration error per segment")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[e]) for e in engines]
    ax.legend(handles, engines, frameon=False, fontsize=8, loc="best")
    return _save(fig, path)


def wer_vs_snr(curves: Mapping[str, Sequence[Mapping]], path, title: str = "", ylabel: str = "WER") -> Path:
    """One line per model. Each point: {"label", "x" (position), "value", "lo", "hi"}; 'clean' goes on the left."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels: list[str] = []
    for color, (name, pts) in zip(PALETTE, curves.items()):
        xs = [p["x"] for p in pts]
        ax.plot(xs, [p["value"] for p in pts], marker="o", color=color, label=name)
        ax.fill_between(xs, [p["lo"] for p in pts], [p["hi"] for p in pts], color=color, alpha=0.18)
        if not labels:
            labels = [p["label"] for p in pts]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("signal-to-noise ratio (dB)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title or f"{ylabel} versus noise level")
    ax.legend(frameon=False)
    return _save(fig, path)


def error_types_stack(rows: Sequence[Mapping], path, title: str = "") -> Path:
    """Stacked substitution / deletion / insertion rates per noise level. Rows: {"label", "S", "D", "I"} (rates)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [r["label"] for r in rows]
    bottom = [0.0] * len(rows)
    for color, key, name in zip(PALETTE, ("S", "D", "I"), ("substitutions", "deletions", "insertions")):
        vals = [r[key] for r in rows]
        ax.bar(labels, vals, bottom=bottom, color=color, label=name, width=0.65)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("signal-to-noise ratio (dB)")
    ax.set_ylabel("errors per reference word")
    ax.set_title(title or "Error types")
    ax.legend(frameon=False)
    return _save(fig, path)


def strategy_boxplots(data: Mapping[str, Mapping[str, Sequence[float]]], path, title: str = "",
                      ylabel: str = "overrun (ms)") -> Path:
    """data[group][strategy] -> list of values; one panel per group (e.g. compression ratio)."""
    groups = list(data)
    strategies = list(next(iter(data.values())))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 0.6, 4.4), sharey=True)
    axes = [axes] if len(groups) == 1 else list(axes)
    for ax, g in zip(axes, groups):
        bp = ax.boxplot([data[g][s] for s in strategies], patch_artist=True, widths=0.6,
                        medianprops={"color": "#222"})
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels(strategies, rotation=35, ha="right", fontsize=8)
        ax.set_title(g, fontsize=10)
    axes[0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    return _save(fig, path)


def grouped_bars(values: Mapping[str, Mapping[str, float]], path, title: str = "", ylabel: str = "",
                 errors: Mapping[str, Mapping[str, tuple[float, float]]] | None = None) -> Path:
    """values[group][series] -> number; optional (lo, hi) error bars."""
    groups = list(values)
    series = list(next(iter(values.values())))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 3.6, 4.0))
    top = max(max(row.values()) for row in values.values())
    for i, (color, s) in enumerate(zip(PALETTE, series)):
        xs = [j + i * width for j in range(len(groups))]
        vals = [values[g][s] for g in groups]
        yerr = None
        if errors:
            yerr = [[max(0.0, v - errors[g][s][0]) for g, v in zip(groups, vals)],
                    [max(0.0, errors[g][s][1] - v) for g, v in zip(groups, vals)]]
        ax.bar(xs, vals, width=width * 0.92, color=color, label=s, yerr=yerr, capsize=2, error_kw={"lw": 0.8})
        for x, v in zip(xs, vals):
            ax.text(x, v + top * 0.015, f"{v:.2f}" if top <= 5 else f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks([j + width * (len(series) - 1) / 2 for j in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.12)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, path)
