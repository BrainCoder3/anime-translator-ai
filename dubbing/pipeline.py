"""Dubbing pipeline: transcription, speakers, translation, duration fitting, synthesis, mix.

Fitting strategies:
  none              synthesise the translation as is
  speedup           speed the voice up by at most max_natural_rate percent
  stretch_uncapped  speed up as much as needed, up to UNCAPPED_LIMIT percent
  adapt             rewrite the text shorter when more than max_natural_rate is needed
  adapt+speedup     adapt, then capped speed-up (rule of the original pipeline, default)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from . import adaptation, fit, media
from .audit import SegmentAudit, build_audit, save_audit
from .backends import ASR, TTS, Diarizer, Translator
from .backends.diarize import SingleSpeaker
from .config import Config
from .diarization import assign_speakers
from .schema import Segment
from .speaker_memory import CharacterMemory
from .textutil import Lexicon

STRATEGIES = ("none", "speedup", "stretch_uncapped", "adapt", "adapt+speedup")
UNCAPPED_LIMIT = 100   # percent; the TTS rate control is not meaningful far beyond this


@dataclass
class Voice:
    data: np.ndarray
    sample_rate: int
    ms: int


@dataclass
class Outcome:
    text: str
    adapted: bool
    rate_applied: int
    base_ms: int           # duration before any fitting
    final_ms: int
    target_ms: int
    attempts: list = field(default_factory=list)
    voice: Voice | None = None

    @property
    def overrun_ms(self) -> int:
        return int(fit.overrun_ms(self.final_ms, self.target_ms))


@dataclass
class DubResult:
    segments: list[Segment]
    audit: dict
    paths: dict[str, Path]


class DubbingPipeline:
    def __init__(self, cfg: Config, asr: ASR, translator: Translator, tts: TTS, *,
                 diarizer: Diarizer | None = None, memory: CharacterMemory | None = None,
                 lexicon: Lexicon | None = None, strategy: str = "adapt+speedup",
                 voices: Mapping[str, str] | None = None, log: Callable[[str], None] = print) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
        self.cfg, self.asr, self.translator, self.tts = cfg, asr, translator, tts
        self.diarizer = diarizer or SingleSpeaker()
        self.memory, self.lexicon, self.strategy = memory, lexicon, strategy
        self.voices = dict(voices or {})
        self.log = log
        self._cache: dict[tuple, Voice] = {}
        self._tts_dir: Path | None = None

    def use_workdir(self, work: str | Path) -> None:
        """Set the synthesis cache folder. Required before synth() or fit_text(); run() calls it."""
        self._tts_dir = Path(work)
        self._tts_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def synth(self, text: str, voice: str, rate: int = 0) -> Voice:
        """Synthesise, trim silence and measure. Cached per (text, voice, rate)."""
        key = (text, voice, rate)
        if key not in self._cache:
            if self._tts_dir is None:
                raise RuntimeError("work directory not set")
            name = hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:16]
            path = self.tts.synthesize(text, self._tts_dir / f"{name}.wav", rate_percent=rate, voice=voice)
            sr, data = media.read_wav(path)
            data = media.trim_silence(data, sr)
            self._cache[key] = Voice(data, sr, int(round(data.size / sr * 1000)))
        return self._cache[key]

    def fit_text(self, text: str, target_ms: int, voice: str, *, previous: str = "", next_: str = "",
                 strategy: str | None = None) -> Outcome:
        strategy = strategy or self.strategy
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        cfg = self.cfg.fit
        base = self.synth(text, voice, 0)
        decision = fit.decide(base.ms, target_ms, cfg)
        attempts: list = []
        adapted = False

        if "adapt" in strategy and decision.needs_adaptation:
            res = adaptation.adapt(text, target_ms, ask=self.translator.ask,
                                   measure_ms=lambda t: self.synth(t, voice, 0).ms, cfg=cfg,
                                   lexicon=self.lexicon, previous=previous, next_=next_)
            text, adapted, attempts = res.text, res.adapted, res.attempts
            decision = fit.decide(self.synth(text, voice, 0).ms, target_ms, cfg)

        rate = 0
        if decision.rate_needed > 0:
            if strategy in ("speedup", "adapt+speedup"):
                rate = decision.speedup_rate
            elif strategy == "stretch_uncapped":
                rate = min(decision.rate_needed, UNCAPPED_LIMIT)
        final = self.synth(text, voice, rate)
        return Outcome(text, adapted, rate, base.ms, final.ms, target_ms, attempts, final)

    def run(self, audio_path: str | Path, out_dir: str | Path, episode: str = "episode") -> DubResult:
        out_dir, work = Path(out_dir), Path(out_dir) / "work"
        self.use_workdir(work / "tts")
        cfg = self.cfg

        wav16 = media.to_wav(audio_path, work / "source_16k.wav")
        source_ms = media.duration_ms(wav16)
        self.log(f"[1/5] transcription ({cfg.source_lang})")
        segments = [Segment(i, s, e, t) for i, (s, e, t) in enumerate(self.asr.transcribe(wav16, cfg.source_lang), 1)]
        if not segments:
            raise RuntimeError("no speech detected in the input")

        self.log("[2/5] speakers")
        turns, embeddings = self.diarizer.diarize(wav16)
        characters: dict[str, str] = {}
        for seg, a in zip(segments, assign_speakers(segments, turns)):
            seg.speaker = a["speaker"] or "SPEAKER_00"
        if self.memory is not None and embeddings:
            for speaker, a in self.memory.recognize(embeddings, episode).items():
                self.voices.setdefault(speaker, a.voice)
                characters[speaker] = a.character or f"{a.status.value}:{speaker}"

        self.log("[3/5] translation")
        previous = ""
        for seg in segments:
            seg.text_fr = self.translator.translate(seg.text_src, cfg.source_lang, cfg.target_lang, previous=previous)
            previous = seg.text_fr

        self.log(f"[4/5] duration fitting (strategy: {self.strategy})")
        records: list[SegmentAudit] = []
        clips: list[tuple[int, np.ndarray]] = []
        sample_rate = None
        for i, seg in enumerate(segments):
            voice = self.voices.get(seg.speaker, cfg.tts_voice)
            prev_t = segments[i - 1].text_fr if i > 0 else ""
            next_t = segments[i + 1].text_fr if i + 1 < len(segments) else ""
            o = self.fit_text(seg.text_fr, seg.target_ms, voice, previous=prev_t, next_=next_t)
            seg.text_fr = o.text
            sample_rate = o.voice.sample_rate
            clips.append((seg.start_ms, o.voice.data))
            records.append(SegmentAudit(seg.id, characters.get(seg.speaker, seg.speaker), self.tts.engine_name,
                                        seg.target_ms, o.final_ms, o.overrun_ms, o.rate_applied, o.adapted, o.attempts))
            self.log(f"    #{seg.id}: slot {seg.target_ms} ms -> dub {o.final_ms} ms "
                     f"(+{o.rate_applied}%{', adapted' if o.adapted else ''}, overrun {o.overrun_ms} ms)")

        self.log("[5/5] mixing and writing outputs")
        mix = media.mix_timeline(clips, sample_rate, source_ms)
        paths = {
            "voice": media.write_wav(out_dir / "french_voice.wav", sample_rate, mix),
            "srt": media.write_srt([(s.start_ms, s.end_ms, s.text_fr) for s in segments], out_dir / "french.srt"),
            "segments": out_dir / "segments.json",
            "audit": out_dir / "audit.json",
        }
        paths["segments"].write_text(json.dumps([s.to_dict() for s in segments], ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        audit = build_audit(records, cfg.to_dict(), {"strategy": self.strategy, "engine": self.tts.engine_name})
        save_audit(audit, paths["audit"])
        return DubResult(segments, audit, paths)
