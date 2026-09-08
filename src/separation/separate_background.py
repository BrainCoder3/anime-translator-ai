
import json
import os
import shutil
import subprocess
from pathlib import Path

from src.core.run_manager import get_run_dir


DEFAULT_DEMUCS_PYTHON = (
    "/content/separation_env/bin/python"
)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def separate_background(
    video_id,
    video_path,
):
    video_path = Path(video_path).resolve()
    run_dir = get_run_dir(video_id)

    separation_dir = (
        run_dir
        / "09_mix"
        / "separation"
    )

    input_audio = (
        separation_dir
        / "original_audio.wav"
    )

    demucs_output = (
        separation_dir
        / "demucs"
    )

    vocals_file = (
        separation_dir
        / "vocals.wav"
    )

    background_file = (
        separation_dir
        / "no_vocals.wav"
    )

    report_file = (
        separation_dir
        / "separation.json"
    )

    separation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================
    # CACHE
    # ==========================================

    if (
        vocals_file.exists()
        and background_file.exists()
    ):
        print(
            "♻️ Séparation audio déjà disponible"
        )
        print(
            "🎤 Voix JP :",
            vocals_file,
        )
        print(
            "🎵 Fond    :",
            background_file,
        )

        return {
            "vocals": vocals_file,
            "background": background_file,
        }

    # ==========================================
    # EXTRACTION AUDIO ORIGINAL STÉRÉO
    # ==========================================

    print(
        "🎧 Extraction de la bande originale..."
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(input_audio),
        ],
        check=True,
    )

    print(
        "✅ Audio original extrait"
    )

    # ==========================================
    # DEMUCS
    # ==========================================

    demucs_python = Path(
        os.environ.get(
            "ANIME_TRANSLATOR_DEMUCS_PYTHON",
            DEFAULT_DEMUCS_PYTHON,
        )
    )

    if not demucs_python.exists():
        raise RuntimeError(
            "Environnement Demucs absent : "
            f"{demucs_python}"
        )

    print()
    print(
        "🧠 Séparation Demucs sur GPU..."
    )
    print(
        "   modèle : htdemucs"
    )
    print(
        "   mode   : vocals / no_vocals"
    )

    subprocess.run(
        [
            str(demucs_python),
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems",
            "vocals",
            "-d",
            "cuda",
            "-o",
            str(demucs_output),
            str(input_audio),
        ],
        check=True,
    )

    # Demucs crée :
    # demucs/htdemucs/original_audio/
    #     vocals.wav
    #     no_vocals.wav

    generated_dir = (
        demucs_output
        / "htdemucs"
        / input_audio.stem
    )

    generated_vocals = (
        generated_dir
        / "vocals.wav"
    )

    generated_background = (
        generated_dir
        / "no_vocals.wav"
    )

    if not generated_vocals.exists():
        raise FileNotFoundError(
            generated_vocals
        )

    if not generated_background.exists():
        raise FileNotFoundError(
            generated_background
        )

    shutil.copy2(
        generated_vocals,
        vocals_file,
    )

    shutil.copy2(
        generated_background,
        background_file,
    )

    save_json(
        report_file,
        {
            "version": 1,
            "video_id": video_id,
            "model": "htdemucs",
            "vocals":
                str(vocals_file),
            "background":
                str(background_file),
            "status": "completed",
        },
    )

    print()
    print(
        "✅ SÉPARATION TERMINÉE"
    )
    print(
        "🎤 Dialogue/voix :",
        vocals_file,
    )
    print(
        "🎵 Fond sans voix :",
        background_file,
    )

    return {
        "vocals": vocals_file,
        "background": background_file,
    }
