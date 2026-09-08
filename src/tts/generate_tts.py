
import gc
import json
import os
import subprocess
import wave
from pathlib import Path

import torch

from src.core.run_manager import (
    get_run,
    get_run_dir,
    stage_done,
    mark_stage_completed,
)

from src.core.anime_memory import (
    get_anime_dir,
    load_json,
    save_json,
)


DEFAULT_TTS_PYTHON = (
    "/content/chatterbox_env/bin/python"
)


def extract_audio_clip(
    source_file,
    output_file,
    start,
    end,
):
    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(source_file),
        "rb",
    ) as src:

        rate = src.getframerate()

        start_frame = max(
            0,
            int(start * rate),
        )

        end_frame = min(
            src.getnframes(),
            int(end * rate),
        )

        if end_frame <= start_frame:
            raise RuntimeError(
                "Intervalle audio invalide."
            )

        src.setpos(
            start_frame
        )

        frames = src.readframes(
            end_frame
            - start_frame
        )

        with wave.open(
            str(output_file),
            "wb",
        ) as dst:

            dst.setnchannels(
                src.getnchannels()
            )

            dst.setsampwidth(
                src.getsampwidth()
            )

            dst.setframerate(
                rate
            )

            dst.writeframes(
                frames
            )

    return output_file


def find_best_fallback_segment(
    character_id,
    character_to_speaker,
    speaker_analysis,
):

    speaker = (
        character_to_speaker.get(
            character_id
        )
    )

    if not speaker:
        return None

    info = (
        speaker_analysis
        .get("speakers", {})
        .get(speaker)
    )

    if not info:
        return None

    candidates = (
        info.get(
            "clean_asr_segments",
            []
        )
        or
        info.get(
            "asr_segments",
            []
        )
    )

    if not candidates:
        return None

    # Meilleur compromis :
    # confiance + couverture + durée.
    def score(seg):

        duration = (
            float(seg["end"])
            - float(seg["start"])
        )

        confidence = float(
            seg.get(
                "speaker_confidence",
                1.0,
            )
        )

        coverage = float(
            seg.get(
                "speech_coverage",
                1.0,
            )
        )

        return (
            confidence
            * coverage
            * min(duration, 8.0)
        )

    return max(
        candidates,
        key=score,
    )


def generate_tts(video_id):

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    anime_dir = get_anime_dir(
        run["anime_id"]
    )

    adaptation_file = (
        run_dir
        / "06_adaptation"
        / "adaptation.json"
    )

    output_dir = (
        run_dir
        / "08_tts"
    )

    raw_dir = (
        output_dir
        / "raw"
    )

    fallback_dir = (
        output_dir
        / "fallback_refs"
    )

    output_file = (
        output_dir
        / "tts.json"
    )

    job_file = (
        output_dir
        / "tts_job.json"
    )

    worker_progress = (
        output_dir
        / "worker_progress.json"
    )

    worker_result = (
        output_dir
        / "worker_result.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fallback_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =============================
    # CACHE
    # =============================

    if (
        stage_done(video_id, "tts")
        and output_file.exists()
    ):
        print(
            "♻️ TTS déjà disponible"
        )
        print("📁", output_file)

        return load_json(
            output_file
        )

    if not adaptation_file.exists():
        raise FileNotFoundError(
            adaptation_file
        )

    adaptation = load_json(
        adaptation_file
    )

    characters_db = load_json(
        anime_dir
        / "characters.json",
        {"characters": {}},
    )

    characters = characters_db[
        "characters"
    ]

    mapping = load_json(
        run_dir
        / "04_characters"
        / "speaker_character_map.json",
        {"mappings": {}},
    )

    speaker_analysis = load_json(
        run_dir
        / "04_characters"
        / "speaker_analysis.json",
        {"speakers": {}},
    )

    source_audio = (
        run_dir
        / "01_audio"
        / "speech_16k_mono.wav"
    )

    # CHARACTER -> SPEAKER dans ce run
    character_to_speaker = {
        info["character_id"]: speaker
        for speaker, info
        in mapping.get(
            "mappings",
            {}
        ).items()
    }

    job_segments = []

    reference_cache = {}

    adaptation_segments = adaptation["segments"]

    for seg_index, seg in enumerate(adaptation_segments):

        segment_id = int(
            seg["id"]
        )

        character_id = seg.get(
            "character_id"
        )

        text = (
            seg.get("adapted_fr")
            or seg.get(
                "translation_fr"
            )
            or ""
        ).strip()

        if not text:
            continue

        start_seconds = float(seg["start"])
        end_seconds = float(seg["end"])

        target_seconds = float(
            seg.get(
                "duration",
                end_seconds - start_seconds,
            )
        )

        # Fenêtre réellement disponible avant
        # la prochaine réplique.
        #
        # On conserve 80 ms de marge afin
        # d'éviter de coller deux voix.
        if seg_index + 1 < len(adaptation_segments):

            next_start = float(
                adaptation_segments[
                    seg_index + 1
                ]["start"]
            )

            available_seconds = max(
                target_seconds,
                next_start
                - start_seconds
                - 0.08,
            )

        else:
            available_seconds = (
                target_seconds
            )

        reference_path = None
        reference_type = None

        # =============================
        # RÉFÉRENCE MÉMOIRE
        # =============================

        if character_id:

            character = characters.get(
                character_id,
                {},
            )

            relative_ref = (
                character.get(
                    "voice_reference"
                )
            )

            if relative_ref:

                candidate = (
                    anime_dir
                    / relative_ref
                )

                if candidate.exists():

                    reference_path = (
                        candidate
                    )

                    reference_type = (
                        "memory"
                    )

        # =============================
        # FALLBACK COURT
        # =============================

        if (
            reference_path is None
            and character_id
        ):

            if (
                character_id
                in reference_cache
            ):

                reference_path = Path(
                    reference_cache[
                        character_id
                    ]
                )

                reference_type = (
                    "short_fallback"
                )

            else:

                candidate = (
                    find_best_fallback_segment(
                        character_id,
                        character_to_speaker,
                        speaker_analysis,
                    )
                )

                if candidate:

                    fallback_file = (
                        fallback_dir
                        / (
                            f"{character_id}"
                            "_reference_ja.wav"
                        )
                    )

                    extract_audio_clip(
                        source_audio,
                        fallback_file,
                        float(
                            candidate["start"]
                        ),
                        float(
                            candidate["end"]
                        ),
                    )

                    reference_path = (
                        fallback_file
                    )

                    reference_type = (
                        "short_fallback"
                    )

                    reference_cache[
                        character_id
                    ] = str(
                        fallback_file
                    )

        # =============================
        # FALLBACK DU SEGMENT LUI-MÊME
        # =============================

        if reference_path is None:

            fallback_file = (
                fallback_dir
                / (
                    f"SEGMENT_"
                    f"{segment_id:04d}"
                    "_reference_ja.wav"
                )
            )

            start = float(
                seg["start"]
            )

            end = float(
                seg["end"]
            )

            # Minimum pratique :
            # un peu de padding.
            start = max(
                0.0,
                start - 0.15,
            )

            end = end + 0.15

            extract_audio_clip(
                source_audio,
                fallback_file,
                start,
                end,
            )

            reference_path = (
                fallback_file
            )

            reference_type = (
                "segment_fallback"
            )

        output_path = (
            raw_dir
            / (
                f"{segment_id:04d}.wav"
            )
        )

        job_segments.append({
            "id": segment_id,

            "character_id":
                character_id,

            "text": text,

            "target_seconds":
                round(
                    target_seconds,
                    3,
                ),

            "available_seconds":
                round(
                    available_seconds,
                    3,
                ),

            "reference_path":
                str(
                    reference_path
                ),

            "reference_type":
                reference_type,

            "output_path":
                str(
                    output_path
                ),
        })

    job = {
        "version": 1,

        "video_id": video_id,

        "engine":
            "chatterbox-multilingual-v3",

        "seed": 1234,

        "progress_file":
            str(worker_progress),

        "result_file":
            str(worker_result),

        "segments":
            job_segments,
    }

    save_json(
        job_file,
        job,
    )

    # =============================
    # ENVIRONNEMENT ISOLÉ
    # =============================

    tts_python = Path(
        os.environ.get(
            "ANIME_TRANSLATOR_TTS_PYTHON",
            DEFAULT_TTS_PYTHON,
        )
    )

    if not tts_python.exists():

        raise RuntimeError(
            "Environnement Chatterbox absent : "
            f"{tts_python}\n"
            "Le bootstrap TTS doit être exécuté "
            "dans ce runtime."
        )

    worker_script = (
        Path(__file__)
        .resolve()
        .parent
        / "chatterbox_worker.py"
    )

    print(
        "🎙️ Lancement TTS "
        "Chatterbox Multilingual V3"
    )

    print(
        "Segments :",
        len(job_segments),
    )

    # Libère autant que possible la VRAM
    # du processus principal avant de
    # démarrer Chatterbox.
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = subprocess.run(
        [
            str(tts_python),
            str(worker_script),
            "--job",
            str(job_file),
        ],
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Le worker Chatterbox "
            "s'est arrêté avec une erreur."
        )

    if not worker_result.exists():
        raise RuntimeError(
            "Résultat worker TTS absent."
        )

    worker_data = load_json(
        worker_result
    )

    results = (
        worker_data["segments"]
    )

    too_long = sum(
        1
        for item in results
        if item.get(
            "needs_speedup",
            False,
        )
    )

    data = {
        "version": 1,

        "video_id": video_id,

        "anime_id":
            run["anime_id"],

        "engine":
            "chatterbox-multilingual-v3",

        "segments":
            results,

        "summary": {
            "total":
                len(results),

            "too_long":
                too_long,
        },
    }

    save_json(
        output_file,
        data,
    )

    mark_stage_completed(
        video_id,
        "tts",
        output_file,
    )

    print()
    print(
        "✅ TTS brut terminé"
    )

    print(
        "Segments :",
        len(results),
    )

    print(
        "⏱️ Trop longs après TTS réel :",
        too_long,
    )

    print(
        "📁",
        output_file,
    )

    return data
