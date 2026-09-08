
import json
from collections import defaultdict

from src.core.run_manager import get_run_dir


def analyze_speakers(video_id):
    run_dir = get_run_dir(video_id)

    diar_file = (
        run_dir
        / "03_diarization"
        / "diarization.json"
    )

    alignment_file = (
        run_dir
        / "03_diarization"
        / "asr_speaker_alignment.json"
    )

    output_file = (
        run_dir
        / "04_characters"
        / "speaker_analysis.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(diar_file, encoding="utf-8") as f:
        diar = json.load(f)

    with open(alignment_file, encoding="utf-8") as f:
        alignment = json.load(f)

    stats = defaultdict(
        lambda: {
            "diarization_duration": 0.0,
            "diarization_turns": 0,
            "asr_segments": [],
            "clean_asr_segments": [],
        }
    )

    # ---------------------------------
    # Durée réelle selon Pyannote
    # ---------------------------------

    for item in diar["exclusive_diarization"]:
        speaker = item["speaker"]

        duration = (
            float(item["end"])
            - float(item["start"])
        )

        stats[speaker]["diarization_duration"] += duration
        stats[speaker]["diarization_turns"] += 1

    # ---------------------------------
    # Segments Whisper associés
    # ---------------------------------

    for seg in alignment["segments"]:
        speaker = seg.get("speaker")

        if speaker is None:
            continue

        item = {
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker_confidence": seg[
                "speaker_confidence"
            ],
            "speech_coverage": seg[
                "speech_coverage"
            ],
        }

        stats[speaker]["asr_segments"].append(
            item
        )

        # Segment suffisamment fiable
        # pour servir plus tard de référence
        if (
            seg["speaker_confidence"] >= 0.85
            and seg["speech_coverage"] >= 0.70
            and (seg["end"] - seg["start"]) >= 1.0
            and (seg["end"] - seg["start"]) <= 12.0
        ):
            stats[speaker][
                "clean_asr_segments"
            ].append(item)

    # ---------------------------------
    # Score de quantité exploitable
    # ---------------------------------

    speakers = {}

    for speaker, info in stats.items():

        clean_duration = sum(
            seg["end"] - seg["start"]
            for seg in info["clean_asr_segments"]
        )

        speakers[speaker] = {
            "diarization_duration": round(
                info["diarization_duration"],
                3,
            ),
            "diarization_turns": info[
                "diarization_turns"
            ],
            "asr_segment_count": len(
                info["asr_segments"]
            ),
            "clean_segment_count": len(
                info["clean_asr_segments"]
            ),
            "clean_duration": round(
                clean_duration,
                3,
            ),
            "asr_segments": info[
                "asr_segments"
            ],
            "clean_asr_segments": info[
                "clean_asr_segments"
            ],
        }

    data = {
        "version": 1,
        "video_id": video_id,
        "speakers": speakers,
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

    print(
        "✅ Analyse des speakers terminée"
    )
    print()

    ranking = sorted(
        speakers.items(),
        key=lambda x: x[1][
            "diarization_duration"
        ],
        reverse=True,
    )

    for speaker, info in ranking:

        print(f"🎙️ {speaker}")
        print(
            "   Durée parole :",
            f'{info["diarization_duration"]:.2f}s'
        )
        print(
            "   Tours        :",
            info["diarization_turns"]
        )
        print(
            "   Phrases ASR  :",
            info["asr_segment_count"]
        )
        print(
            "   Propres      :",
            info["clean_segment_count"],
        )
        print(
            "   Audio propre :",
            f'{info["clean_duration"]:.2f}s'
        )

        for seg in info["clean_asr_segments"]:
            print(
                f'      [{seg["start"]:.2f}'
                f' → {seg["end"]:.2f}] '
                f'{seg["text"]}'
            )

        print()

    print("📁", output_file)

    return data
