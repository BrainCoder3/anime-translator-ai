
import json
import wave

from src.core.run_manager import (
    get_run,
    get_run_dir,
    mark_stage_completed,
)

from src.core.anime_memory import (
    get_anime_dir,
    load_json,
    save_json,
    now,
)


MIN_REFERENCE_SECONDS = 5.0
TARGET_REFERENCE_SECONDS = 10.0
MIN_SEGMENT_SECONDS = 1.2
MAX_SEGMENT_SECONDS = 8.0
SILENCE_BETWEEN_SEGMENTS = 0.12


def extract_voice_references(video_id):

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    anime_dir = get_anime_dir(
        run["anime_id"]
    )

    audio_file = (
        run_dir
        / "01_audio"
        / "speech_16k_mono.wav"
    )

    analysis_file = (
        run_dir
        / "04_characters"
        / "speaker_analysis.json"
    )

    mapping_file = (
        run_dir
        / "04_characters"
        / "speaker_character_map.json"
    )

    characters_file = (
        anime_dir
        / "characters.json"
    )

    references_dir = (
        anime_dir
        / "voice_references"
    )

    references_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    analysis = load_json(analysis_file)
    mapping = load_json(mapping_file)
    characters_db = load_json(characters_file)

    if not audio_file.exists():
        raise FileNotFoundError(audio_file)

    results = {}

    with wave.open(str(audio_file), "rb") as src:

        framerate = src.getframerate()
        channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        comptype = src.getcomptype()
        compname = src.getcompname()

        silence_frames = int(
            SILENCE_BETWEEN_SEGMENTS
            * framerate
        )

        silence = (
            b"\x00"
            * silence_frames
            * channels
            * sampwidth
        )

        for speaker, map_info in mapping[
            "mappings"
        ].items():

            character_id = map_info[
                "character_id"
            ]

            speaker_info = analysis[
                "speakers"
            ][speaker]

            candidates = []

            for seg in speaker_info[
                "clean_asr_segments"
            ]:

                duration = (
                    float(seg["end"])
                    - float(seg["start"])
                )

                if not (
                    MIN_SEGMENT_SECONDS
                    <= duration
                    <= MAX_SEGMENT_SECONDS
                ):
                    continue

                quality = (
                    float(
                        seg["speaker_confidence"]
                    )
                    *
                    float(
                        seg["speech_coverage"]
                    )
                )

                candidates.append(
                    {
                        **seg,
                        "duration": duration,
                        "quality": quality,
                    }
                )

            # Les meilleures portions en premier
            candidates.sort(
                key=lambda x: (
                    x["quality"],
                    x["duration"],
                ),
                reverse=True,
            )

            available = sum(
                x["duration"]
                for x in candidates
            )

            character = characters_db[
                "characters"
            ][character_id]

            # Pas assez d'audio fiable
            if available < MIN_REFERENCE_SECONDS:

                character[
                    "voice_reference_status"
                ] = "insufficient_audio"

                character[
                    "updated_at"
                ] = now()

                results[character_id] = {
                    "speaker": speaker,
                    "status": "insufficient_audio",
                    "available_seconds": round(
                        available,
                        3,
                    ),
                }

                continue

            selected = []
            total = 0.0

            for seg in candidates:

                selected.append(seg)
                total += seg["duration"]

                if total >= TARGET_REFERENCE_SECONDS:
                    break

            output_file = (
                references_dir
                / f"{character_id}_reference_ja.wav"
            )

            with wave.open(
                str(output_file),
                "wb"
            ) as dst:

                dst.setnchannels(channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.setcomptype(
                    comptype,
                    compname,
                )

                for index, seg in enumerate(
                    selected
                ):

                    start_frame = int(
                        seg["start"]
                        * framerate
                    )

                    end_frame = int(
                        seg["end"]
                        * framerate
                    )

                    src.setpos(start_frame)

                    frames = src.readframes(
                        end_frame
                        - start_frame
                    )

                    dst.writeframes(frames)

                    if index < len(selected) - 1:
                        dst.writeframes(
                            silence
                        )

            reference_duration = (
                total
                + SILENCE_BETWEEN_SEGMENTS
                * max(
                    len(selected) - 1,
                    0,
                )
            )

            relative_path = (
                "voice_references/"
                f"{character_id}_reference_ja.wav"
            )

            character[
                "voice_reference"
            ] = relative_path

            character[
                "voice_reference_status"
            ] = "ready"

            character[
                "voice_reference_seconds"
            ] = round(
                reference_duration,
                3,
            )

            character[
                "voice_reference_source_video"
            ] = video_id

            character[
                "voice_reference_segments"
            ] = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "quality": round(
                        seg["quality"],
                        4,
                    ),
                }
                for seg in selected
            ]

            character[
                "updated_at"
            ] = now()

            results[character_id] = {
                "speaker": speaker,
                "status": "ready",
                "reference": str(
                    output_file
                ),
                "duration": round(
                    reference_duration,
                    3,
                ),
                "segments": len(selected),
            }

    save_json(
        characters_file,
        characters_db,
    )

    summary_file = (
        run_dir
        / "04_characters"
        / "voice_references.json"
    )

    save_json(
        summary_file,
        {
            "version": 1,
            "video_id": video_id,
            "anime_id": run["anime_id"],
            "characters": results,
        },
    )

    mark_stage_completed(
        video_id,
        "characters",
        summary_file,
    )

    print(
        "✅ Références vocales générées"
    )
    print()

    for character_id, info in results.items():

        print(
            character_id,
            "←",
            info["speaker"],
        )

        if info["status"] == "ready":

            print(
                "   ✅",
                f'{info["duration"]:.2f}s',
                "-",
                info["segments"],
                "segments",
            )

        else:

            print(
                "   ⚠️ Audio insuffisant :",
                f'{info["available_seconds"]:.2f}s',
            )

    print()
    print(
        "✅ Stage characters = completed"
    )
    print(
        "📁",
        summary_file,
    )

    return results
