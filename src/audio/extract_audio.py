import subprocess
from pathlib import Path

from src.core.run_manager import (
    get_run,
    get_run_dir,
    stage_done,
    mark_stage_completed,
)


def extract_audio(video_id):
    """
    Extrait une piste WAV mono 16 kHz destinée à :
    - Whisper / ASR
    - Pyannote
    - embeddings voix
    - analyse prosodique

    Si l'étape a déjà été réalisée, elle est réutilisée.
    """

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    video_path = Path(run["video_path"])

    audio_dir = run_dir / "01_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    output_path = audio_dir / "speech_16k_mono.wav"

    # -------------------------------
    # CACHE
    # -------------------------------

    if stage_done(video_id, "audio"):

        if output_path.exists():
            print("♻️ Audio déjà extrait")
            print("📁", output_path)
            return output_path

        print(
            "⚠️ Le cache disait terminé mais "
            "le fichier audio est absent."
        )
        print("↪️ Reconstruction...")

    # -------------------------------
    # EXTRACTION FFMPEG
    # -------------------------------

    print("🎧 Extraction audio...")

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),

        "-vn",

        # Mono
        "-ac", "1",

        # 16 kHz
        "-ar", "16000",

        # PCM 16 bits
        "-c:a", "pcm_s16le",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Erreur FFmpeg :\n"
            + result.stderr[-3000:]
        )

    if not output_path.exists():
        raise RuntimeError(
            "FFmpeg n'a produit aucun fichier audio."
        )

    # -------------------------------
    # CACHE RUN
    # -------------------------------

    mark_stage_completed(
        video_id,
        "audio",
        output_path
    )

    print("✅ Audio extrait")
    print("📁", output_path)

    return output_path
