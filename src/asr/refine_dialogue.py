
import json
import tempfile
import wave
from pathlib import Path

import torch
from faster_whisper import WhisperModel

from src.core.run_manager import (
    get_run_dir,
    reset_stage,
)

MODEL_SIZE = "large-v3"

SPEAKER_THRESHOLD = 0.65
COVERAGE_THRESHOLD = 0.30
LONG_SEGMENT_SECONDS = 12.0

WINDOW_GAP = 0.45
WINDOW_PADDING = 0.25
MAX_WINDOW_SECONDS = 4.5


MIN_RESULT_SECONDS = 0.35
MAX_JA_CHARS_PER_SECOND = 10.0
MIN_REPLACEMENT_COVERAGE = 0.45


def candidate_is_reliable(result, text):
    """
    Rejette les hallucinations typiques de Whisper :
    répétitions, texte impossible pour la durée,
    faible probabilité acoustique, etc.
    """

    from collections import Counter

    duration = max(
        float(result.end) - float(result.start),
        0.001,
    )

    if duration < MIN_RESULT_SECONDS:
        return False, "too_short"

    compact = "".join(
        c for c in text
        if not c.isspace()
        and c not in "。、！？!?.,…ー-"
    )

    # Beaucoup trop de caractères pour la durée audio.
    if (
        compact
        and len(compact) / duration
        > MAX_JA_CHARS_PER_SECOND
    ):
        return False, "impossible_text_speed"

    # Exemple : うううううううううう...
    if len(compact) >= 10:
        counts = Counter(compact)

        repetition_ratio = (
            max(counts.values())
            / len(compact)
        )

        if repetition_ratio >= 0.55:
            return False, "excessive_repetition"

    avg_logprob = getattr(
        result,
        "avg_logprob",
        None,
    )

    no_speech_prob = getattr(
        result,
        "no_speech_prob",
        None,
    )

    compression_ratio = getattr(
        result,
        "compression_ratio",
        None,
    )

    if (
        avg_logprob is not None
        and avg_logprob < -1.25
    ):
        return False, "low_logprob"

    if (
        no_speech_prob is not None
        and avg_logprob is not None
        and no_speech_prob > 0.65
        and avg_logprob < -0.50
    ):
        return False, "likely_no_speech"

    if (
        compression_ratio is not None
        and compression_ratio > 2.4
    ):
        return False, "high_compression"

    return True, None


def detected_speech_seconds(
    start,
    end,
    diarization,
):
    total = 0.0

    for item in diarization:
        total += overlap(
            start,
            end,
            float(item["start"]),
            float(item["end"]),
        )

    return total


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def speaker_stats(start, end, diarization):
    scores = {}

    for item in diarization:
        amount = overlap(
            start,
            end,
            float(item["start"]),
            float(item["end"]),
        )

        if amount <= 0:
            continue

        speaker = item["speaker"]

        scores[speaker] = (
            scores.get(speaker, 0.0)
            + amount
        )

    duration = max(
        end - start,
        0.001,
    )

    if not scores:
        return None, 0.0, 0.0

    speaker = max(
        scores,
        key=scores.get,
    )

    total = sum(scores.values())

    confidence = (
        scores[speaker] / total
        if total > 0
        else 0.0
    )

    coverage = min(
        total / duration,
        1.0,
    )

    return (
        speaker,
        confidence,
        coverage,
    )


def build_windows(
    segment,
    diarization,
    audio_duration,
):
    start = float(segment["start"])
    end = float(segment["end"])

    speech_parts = []

    for item in diarization:
        s = max(
            start,
            float(item["start"]),
        )

        e = min(
            end,
            float(item["end"]),
        )

        if e - s >= 0.08:
            speech_parts.append(
                [s, e]
            )

    # Si Pyannote n'a rien trouvé,
    # on garde la zone Whisper.
    if not speech_parts:
        return [[
            max(0.0, start - WINDOW_PADDING),
            min(
                audio_duration,
                end + WINDOW_PADDING,
            ),
        ]]

    speech_parts.sort(
        key=lambda x: x[0]
    )

    groups = []

    current_start = speech_parts[0][0]
    current_end = speech_parts[0][1]

    for s, e in speech_parts[1:]:

        gap = s - current_end
        proposed_duration = (
            e - current_start
        )

        if (
            gap <= WINDOW_GAP
            and proposed_duration
            <= MAX_WINDOW_SECONDS
        ):
            current_end = max(
                current_end,
                e,
            )

        else:
            groups.append([
                current_start,
                current_end,
            ])

            current_start = s
            current_end = e

    groups.append([
        current_start,
        current_end,
    ])

    padded = []

    for s, e in groups:
        padded.append([
            max(
                0.0,
                s - WINDOW_PADDING,
            ),
            min(
                audio_duration,
                e + WINDOW_PADDING,
            ),
        ])

    return padded


def write_clip(
    source,
    destination,
    start,
    end,
):
    framerate = source.getframerate()

    start_frame = int(
        start * framerate
    )

    end_frame = int(
        end * framerate
    )

    source.setpos(start_frame)

    frames = source.readframes(
        end_frame - start_frame
    )

    with wave.open(
        str(destination),
        "wb"
    ) as dst:

        dst.setnchannels(
            source.getnchannels()
        )

        dst.setsampwidth(
            source.getsampwidth()
        )

        dst.setframerate(
            framerate
        )

        dst.writeframes(frames)


def refine_dialogue(
    video_id,
    model=None,
):
    run_dir = get_run_dir(video_id)

    dialogue_file = (
        run_dir
        / "05_translation"
        / "dialogue_source.json"
    )

    output_file = (
        run_dir
        / "05_translation"
        / "dialogue_refined.json"
    )

    diar_file = (
        run_dir
        / "03_diarization"
        / "diarization.json"
    )

    mapping_file = (
        run_dir
        / "04_characters"
        / "speaker_character_map.json"
    )

    audio_file = (
        run_dir
        / "01_audio"
        / "speech_16k_mono.wav"
    )

    # ============================
    # CACHE
    # ============================

    if output_file.exists():
        print(
            "♻️ Dialogue ASR déjà raffiné"
        )

        with open(
            output_file,
            encoding="utf-8",
        ) as f:
            return json.load(f)

    with open(
        dialogue_file,
        encoding="utf-8",
    ) as f:
        dialogue = json.load(f)

    with open(
        diar_file,
        encoding="utf-8",
    ) as f:
        diar_data = json.load(f)

    with open(
        mapping_file,
        encoding="utf-8",
    ) as f:
        mapping_data = json.load(f)

    diarization = diar_data[
        "exclusive_diarization"
    ]

    speaker_map = {
        speaker: info["character_id"]
        for speaker, info
        in mapping_data["mappings"].items()
    }

    suspect = []

    for seg in dialogue["segments"]:

        duration = (
            float(seg["end"])
            - float(seg["start"])
        )

        bad = (
            seg.get(
                "needs_asr_review",
                False,
            )
            or float(
                seg.get(
                    "speaker_confidence",
                    1.0,
                )
            ) < SPEAKER_THRESHOLD
            or float(
                seg.get(
                    "speech_coverage",
                    1.0,
                )
            ) < COVERAGE_THRESHOLD
            or duration
            > LONG_SEGMENT_SECONDS
        )

        if bad:
            suspect.append(seg)

    print(
        "🔎 Segments ASR suspects :",
        len(suspect),
    )

    # Aucun problème :
    # on crée simplement la version raffinée.
    if not suspect:

        data = {
            **dialogue,
            "refinement": {
                "model": MODEL_SIZE,
                "suspect_segments": 0,
                "refined": False,
            },
        }

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return data

    # ============================
    # GPU obligatoire
    # ============================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Le raffinement ASR nécessite "
            "un GPU CUDA."
        )

    if model is None:

        print(
            "🧠 Chargement Whisper QA :",
            MODEL_SIZE,
        )

        model = WhisperModel(
            MODEL_SIZE,
            device="cuda",
            compute_type="float16",
        )

    suspect_ids = {
        int(seg["id"])
        for seg in suspect
    }

    refined_segments = []

    with wave.open(
        str(audio_file),
        "rb"
    ) as source:

        audio_duration = (
            source.getnframes()
            / source.getframerate()
        )

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            for original in dialogue["segments"]:

                source_id = int(
                    original["id"]
                )

                # ------------------------
                # Segment déjà fiable
                # ------------------------

                if source_id not in suspect_ids:

                    refined_segments.append({
                        **original,
                        "source_id": source_id,
                        "asr_refined": False,
                    })

                    continue

                print()
                print(
                    f"🔧 Raffinement segment "
                    f"{source_id:03d}"
                )

                print(
                    "   Ancien :",
                    original["text_ja"],
                )

                windows = build_windows(
                    original,
                    diarization,
                    audio_duration,
                )

                replacements = []

                for window_index, (
                    window_start,
                    window_end,
                ) in enumerate(
                    windows,
                    1,
                ):

                    clip_file = (
                        tmp
                        / (
                            f"{source_id}_"
                            f"{window_index}.wav"
                        )
                    )

                    write_clip(
                        source,
                        clip_file,
                        window_start,
                        window_end,
                    )

                    segments, _ = (
                        model.transcribe(
                            str(clip_file),
                            language="ja",
                            beam_size=5,
                            word_timestamps=True,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            temperature=0.0,
                        )
                    )

                    for result in segments:

                        text = (
                            result.text
                            .strip()
                        )

                        if not text:
                            continue

                        usable, rejection_reason = (
                            candidate_is_reliable(
                                result,
                                text,
                            )
                        )

                        if not usable:
                            print(
                                "   🚫 Rejet ASR :",
                                repr(text[:80]),
                                f"({rejection_reason})"
                            )
                            continue

                        abs_start = (
                            window_start
                            + float(
                                result.start
                            )
                        )

                        abs_end = (
                            window_start
                            + float(
                                result.end
                            )
                        )

                        speaker, conf, coverage = (
                            speaker_stats(
                                abs_start,
                                abs_end,
                                diarization,
                            )
                        )

                        character_id = (
                            speaker_map.get(
                                speaker
                            )
                            if speaker
                            else None
                        )

                        duration = (
                            abs_end
                            - abs_start
                        )

                        issues = []

                        if conf < SPEAKER_THRESHOLD:
                            issues.append(
                                "uncertain_speaker"
                            )

                        if coverage < COVERAGE_THRESHOLD:
                            issues.append(
                                "low_speech_coverage"
                            )

                        if (
                            duration
                            > LONG_SEGMENT_SECONDS
                        ):
                            issues.append(
                                "suspicious_long_asr_segment"
                            )

                        replacements.append({
                            "id": None,
                            "source_id": source_id,

                            "start": round(
                                abs_start,
                                3,
                            ),

                            "end": round(
                                abs_end,
                                3,
                            ),

                            "duration": round(
                                duration,
                                3,
                            ),

                            "speaker": speaker,
                            "character_id":
                                character_id,

                            "text_ja": text,

                            "speaker_confidence":
                                round(conf, 4),

                            "speech_coverage":
                                round(
                                    coverage,
                                    4,
                                ),

                            "needs_asr_review":
                                bool(issues),

                            "issues": issues,

                            "translation_fr": None,
                            "adapted_fr": None,

                            "asr_refined": True,
                        })

                # Vérifie que la seconde passe a réellement
                # récupéré une quantité suffisante de parole.
                source_speech = detected_speech_seconds(
                    float(original["start"]),
                    float(original["end"]),
                    diarization,
                )

                replacement_speech = sum(
                    float(item["end"])
                    - float(item["start"])
                    for item in replacements
                )

                replacement_coverage = (
                    replacement_speech / source_speech
                    if source_speech > 0
                    else 0.0
                )

                if (
                    replacements
                    and source_speech >= 1.0
                    and replacement_coverage
                    < MIN_REPLACEMENT_COVERAGE
                ):
                    print(
                        "   🚫 Raffinement rejeté : "
                        f"couverture {replacement_coverage:.2f}"
                    )

                    replacements = []

                # Si la seconde passe n'est pas meilleure,
                # on conserve la première transcription.
                if not replacements:

                    fallback = {
                        **original,
                        "source_id": source_id,
                        "asr_refined": False,
                        "refinement_failed": True,
                    }

                    refined_segments.append(
                        fallback
                    )

                    print(
                        "   ⚠️ Aucun remplacement"
                    )

                else:

                    replacements.sort(
                        key=lambda x: x["start"]
                    )

                    refined_segments.extend(
                        replacements
                    )

                    for item in replacements:
                        print(
                            f'   → '
                            f'[{item["start"]:.2f}'
                            f' → '
                            f'{item["end"]:.2f}] '
                            f'{item["text_ja"]}'
                        )

    # ============================
    # TRI + NOUVEAUX IDs
    # ============================

    refined_segments.sort(
        key=lambda x: (
            float(x["start"]),
            float(x["end"]),
        )
    )

    for index, item in enumerate(
        refined_segments,
        1,
    ):
        item["id"] = index

    data = {
        "version": 2,
        "video_id": video_id,

        "refinement": {
            "model": MODEL_SIZE,
            "suspect_segments":
                len(suspect),
            "refined": True,
        },

        "segments": refined_segments,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # La traduction précédente n'est plus valide.
    translation_file = (
        run_dir
        / "05_translation"
        / "translation.json"
    )

    if translation_file.exists():
        translation_file.unlink()

    reset_stage(
        video_id,
        "translation",
    )

    print()
    print(
        "✅ Raffinement ASR terminé"
    )

    print(
        "Nouveaux segments :",
        len(refined_segments),
    )

    print(
        "♻️ Ancienne traduction invalidée"
    )

    print(
        "📁",
        output_file,
    )

    return data
