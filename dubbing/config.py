"""Parameters. Threshold defaults are the values hard-coded in the original v12.4 pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SpeakerThresholds:
    match: float = 0.84
    margin: float = 0.08
    new_character: float = 0.72


@dataclass(frozen=True)
class FitConfig:
    max_natural_rate: int = 25   # percent speed-up still sounding natural
    tolerance_ms: int = 50
    max_adapt_candidates: int = 2
    semantic_score_threshold: int = 90
    words_per_second: float = 3.4


VOICE_POOL: tuple[str, ...] = (
    "fr-FR-DeniseNeural", "fr-FR-EloiseNeural", "fr-FR-HenriNeural",
    "fr-FR-VivienneMultilingualNeural", "fr-FR-RemyMultilingualNeural",
    "fr-BE-CharlineNeural", "fr-BE-GerardNeural", "fr-CH-ArianeNeural", "fr-CH-FabriceNeural",
    "fr-CA-AntoineNeural", "fr-CA-JeanNeural", "fr-CA-SylvieNeural", "fr-CA-ThierryNeural",
)


@dataclass(frozen=True)
class Config:
    source_lang: str = "ja"
    target_lang: str = "fr"
    whisper_model: str = "small"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_timeout_s: int = 180
    tts_voice: str = "fr-FR-DeniseNeural"
    speakers: SpeakerThresholds = field(default_factory=SpeakerThresholds)
    fit: FitConfig = field(default_factory=FitConfig)

    def to_dict(self) -> dict:
        return asdict(self)
