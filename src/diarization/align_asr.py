
import json
from src.core.run_manager import get_run_dir


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0.0,
        min(a_end, b_end) - max(a_start, b_start)
    )


def align_asr_with_speakers(video_id):
    run_dir = get_run_dir(video_id)

    asr_file = (
        run_dir
        / "02_asr"
        / "transcription.json"
    )

    diar_file = (
        run_dir
        / "03_diarization"
        / "diarization.json"
    )

    output_file = (
        run_dir
        / "03_diarization"
        / "asr_speaker_alignment.json"
    )

    if not asr_file.exists():
        raise FileNotFoundError(asr_file)

    if not diar_file.exists():
        raise FileNotFoundError(diar_file)

    with open(asr_file, encoding="utf-8") as f:
        asr = json.load(f)

    with open(diar_file, encoding="utf-8") as f:
        diar = json.load(f)

    # Supporte quelques formats possibles
    asr_segments = (
        asr.get("segments")
        or asr.get("transcription")
        or []
    )

    diar_segments = diar["exclusive_diarization"]

    results = []

    for index, seg in enumerate(asr_segments, 1):

        start = float(seg["start"])
        end = float(seg["end"])
        text = seg.get("text", "").strip()

        scores = {}

        for d in diar_segments:
            amount = overlap(
                start,
                end,
                float(d["start"]),
                float(d["end"]),
            )

            if amount > 0:
                speaker = d["speaker"]

                scores[speaker] = (
                    scores.get(speaker, 0.0)
                    + amount
                )

        duration = max(end - start, 0.001)

        if scores:
            speaker = max(
                scores,
                key=scores.get
            )

            speaker_overlap = scores[speaker]

            total_speech_overlap = sum(
                scores.values()
            )

            confidence = (
                speaker_overlap
                / total_speech_overlap
                if total_speech_overlap > 0
                else 0.0
            )

            coverage = min(
                total_speech_overlap / duration,
                1.0,
            )

        else:
            speaker = None
            confidence = 0.0
            coverage = 0.0

        item = {
            "id": seg.get("id", index),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "speaker": speaker,
            "speaker_confidence": round(
                confidence,
                4,
            ),
            "speech_coverage": round(
                coverage,
                4,
            ),
            "speaker_overlaps": {
                key: round(value, 3)
                for key, value in sorted(
                    scores.items()
                )
            },
        }

        results.append(item)

    data = {
        "version": 1,
        "video_id": video_id,
        "segments": results,
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

    print("✅ Alignement ASR ↔ locuteurs terminé")
    print()

    for item in results:

        speaker = (
            item["speaker"]
            if item["speaker"]
            else "UNKNOWN"
        )

        warning = ""

        if item["speaker_confidence"] < 0.65:
            warning += " ⚠️ speaker"

        if item["speech_coverage"] < 0.30:
            warning += " ⚠️ coverage"

        print(
            f'{int(item["id"]):03d}. '
            f'[{item["start"]:.2f} -> '
            f'{item["end"]:.2f}] '
            f'{speaker} '
            f'conf={item["speaker_confidence"]:.2f} '
            f'cov={item["speech_coverage"]:.2f}'
            f'{warning}'
        )

        print(
            "     ",
            item["text"],
        )

    print()
    print("📁", output_file)

    return data
