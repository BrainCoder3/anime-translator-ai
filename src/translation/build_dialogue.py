
import json

from src.core.run_manager import get_run_dir


def build_dialogue(video_id):
    run_dir = get_run_dir(video_id)

    alignment_file = (
        run_dir
        / "03_diarization"
        / "asr_speaker_alignment.json"
    )

    mapping_file = (
        run_dir
        / "04_characters"
        / "speaker_character_map.json"
    )

    output_dir = (
        run_dir
        / "05_translation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "dialogue_source.json"
    )

    with open(
        alignment_file,
        encoding="utf-8"
    ) as f:
        alignment = json.load(f)

    with open(
        mapping_file,
        encoding="utf-8"
    ) as f:
        mapping = json.load(f)

    speaker_map = {
        speaker: info["character_id"]
        for speaker, info
        in mapping["mappings"].items()
    }

    dialogue = []

    for seg in alignment["segments"]:

        speaker = seg.get("speaker")

        character_id = (
            speaker_map.get(speaker)
            if speaker
            else None
        )

        confidence = float(
            seg["speaker_confidence"]
        )

        coverage = float(
            seg["speech_coverage"]
        )

        duration = (
            float(seg["end"])
            - float(seg["start"])
        )

        issues = []

        if confidence < 0.65:
            issues.append(
                "uncertain_speaker"
            )

        if coverage < 0.30:
            issues.append(
                "low_speech_coverage"
            )

        if duration > 12.0:
            issues.append(
                "suspicious_long_asr_segment"
            )

        needs_review = bool(issues)

        dialogue.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "duration": round(
                duration,
                3,
            ),

            "speaker": speaker,
            "character_id": character_id,

            "text_ja": seg["text"],

            "speaker_confidence": confidence,
            "speech_coverage": coverage,

            "needs_asr_review": needs_review,
            "issues": issues,

            "translation_fr": None,
            "adapted_fr": None,
        })

    data = {
        "version": 1,
        "video_id": video_id,
        "segments": dialogue,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("✅ Dialogue canonique créé")
    print()

    review_count = 0

    for seg in dialogue:

        flag = ""

        if seg["needs_asr_review"]:
            flag = " ⚠️"
            review_count += 1

        character = (
            seg["character_id"]
            or "UNKNOWN"
        )

        print(
            f'{int(seg["id"]):03d}. '
            f'{character} '
            f'[{seg["start"]:.2f}'
            f' → {seg["end"]:.2f}]'
            f'{flag}'
        )

        print(
            "     ",
            seg["text_ja"]
        )

        if seg["issues"]:
            print(
                "      ↳",
                ", ".join(seg["issues"])
            )

    print()

    print(
        "Segments :",
        len(dialogue)
    )

    print(
        "À vérifier automatiquement :",
        review_count
    )

    print()

    print("📁", output_file)

    return data
