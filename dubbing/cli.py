"""Command line: ``python -m dubbing run ...`` and ``python -m dubbing timing ...``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .eval import timing
from .eval.report import to_markdown_table
from .pipeline import STRATEGIES


def _cmd_run(args: argparse.Namespace) -> int:
    from .backends.asr_whisper import FasterWhisperASR
    from .backends.translate_ollama import OllamaClient, OllamaTranslator
    from .backends.tts_edge import EdgeTTS
    from .pipeline import DubbingPipeline

    cfg = Config(source_lang=args.source_lang, target_lang=args.target_lang, whisper_model=args.whisper_model,
                 ollama_model=args.ollama_model, tts_voice=args.voice)
    pipe = DubbingPipeline(cfg, FasterWhisperASR(cfg.whisper_model),
                           OllamaTranslator(OllamaClient(cfg.ollama_url, cfg.ollama_model, cfg.ollama_timeout_s)),
                           EdgeTTS(cfg.tts_voice), strategy=args.strategy)
    result = pipe.run(args.input, args.output)
    stats = timing.overflow_stats(result.audit["segments"], args.tol)
    print("\n" + to_markdown_table([stats], ["n_segments", "n_overrun", "overrun_rate", "fit_rate",
                                             "mean_overrun_ms", "max_overrun_ms"]))
    print("\noutputs:", *(f"{k}={v}" for k, v in result.paths.items()), sep="\n  ")
    return 0


def _cmd_timing(args: argparse.Namespace) -> int:
    segments = json.loads(Path(args.audit).read_text(encoding="utf-8"))["segments"]
    rows = {"all": timing.overflow_stats(segments, args.tol)}
    if args.group_by:
        rows.update(timing.by_group(segments, args.group_by, args.tol))
    cols = ["group", "n_segments", "n_overrun", "overrun_rate", "fit_rate", "mean_overrun_ms",
            "max_overrun_ms", "median_ratio", "p95_ratio"]
    print(to_markdown_table([{"group": g, **r} for g, r in rows.items()], cols))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dubbing", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="dub a file (needs ffmpeg, Whisper, Ollama and internet for edge-tts)")
    run.add_argument("input")
    run.add_argument("-o", "--output", required=True)
    run.add_argument("--strategy", choices=STRATEGIES, default="adapt+speedup")
    run.add_argument("--source-lang", default="ja")
    run.add_argument("--target-lang", default="fr")
    run.add_argument("--whisper-model", default="small")
    run.add_argument("--ollama-model", default=Config().ollama_model)
    run.add_argument("--voice", default=Config().tts_voice)
    run.add_argument("--tol", type=float, default=Config().fit.tolerance_ms)
    run.set_defaults(func=_cmd_run)

    tm = sub.add_parser("timing", help="duration-fit metrics of an audit.json")
    tm.add_argument("audit")
    tm.add_argument("--tol", type=float, default=200.0)
    tm.add_argument("--group-by", default=None)
    tm.set_defaults(func=_cmd_timing)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
