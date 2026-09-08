
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np

from src.core.run_manager import (
    get_run_dir,
    stage_done,
    mark_stage_completed,
)


TTS_SR = 24000


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return float(result.stdout.strip())


def video_has_audio(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return bool(result.stdout.strip())


def read_pcm_wav(path):
    with wave.open(str(path), "rb") as wf:

        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()

        if width != 2:
            raise RuntimeError(
                f"WAV non PCM16 : {path}"
            )

        raw = wf.readframes(
            wf.getnframes()
        )

    data = np.frombuffer(
        raw,
        dtype=np.int16,
    ).astype(np.float32)

    if channels > 1:
        data = data.reshape(
            -1,
            channels,
        ).mean(axis=1)

    data /= 32768.0

    return data, sr


def write_pcm16(path, audio, sr):
    path = Path(path)

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm = (
        audio * 32767.0
    ).astype(np.int16)

    with wave.open(
        str(path),
        "wb",
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(
            pcm.tobytes()
        )


def build_dialogue_track(
    video_id,
    video_path,
):

    video_path = Path(video_path)

    run_dir = get_run_dir(video_id)

    timing_file = (
        run_dir
        / "08_tts"
        / "timing_final.json"
    )

    adaptation_file = (
        run_dir
        / "06_adaptation"
        / "adaptation.json"
    )

    mix_dir = (
        run_dir
        / "09_mix"
    )

    mix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dialogue_file = (
        mix_dir
        / "dialogue_fr.wav"
    )

    timing = load_json(
        timing_file
    )

    adaptation = load_json(
        adaptation_file
    )

    start_by_id = {
        int(seg["id"]):
            float(seg["start"])
        for seg
        in adaptation["segments"]
    }

    duration = probe_duration(
        video_path
    )

    total_samples = math.ceil(
        (duration + 0.5)
        * TTS_SR
    )

    canvas = np.zeros(
        total_samples,
        dtype=np.float32,
    )

    placed = []

    for item in timing["segments"]:

        seg_id = int(
            item["id"]
        )

        wav_path = Path(
            item["final_path"]
        )

        if seg_id not in start_by_id:
            raise RuntimeError(
                f"Timestamp absent : {seg_id}"
            )

        audio, sr = read_pcm_wav(
            wav_path
        )

        if sr != TTS_SR:
            raise RuntimeError(
                f"Sample rate inattendu "
                f"pour {wav_path}: {sr}"
            )

        start_seconds = (
            start_by_id[seg_id]
        )

        start_sample = round(
            start_seconds * TTS_SR
        )

        end_sample = min(
            start_sample + len(audio),
            len(canvas),
        )

        usable = (
            end_sample
            - start_sample
        )

        if usable <= 0:
            continue

        canvas[
            start_sample:end_sample
        ] += audio[:usable]

        placed.append({
            "id": seg_id,
            "start":
                round(start_seconds, 3),
            "duration":
                round(
                    usable / TTS_SR,
                    3,
                ),
        })

    # Protection contre les éventuels
    # chevauchements de répliques.
    peak = float(
        np.max(
            np.abs(canvas)
        )
    )

    if peak > 0.96:
        canvas *= (
            0.96 / peak
        )

    write_pcm16(
        dialogue_file,
        canvas,
        TTS_SR,
    )

    print(
        "✅ Timeline FR construite"
    )
    print(
        "Répliques placées :",
        len(placed),
    )
    print(
        "📁",
        dialogue_file,
    )

    return (
        dialogue_file,
        placed,
    )


def build_final_video(
    video_id,
    video_path,
):

    video_path = Path(
        video_path
    ).resolve()

    run_dir = get_run_dir(
        video_id
    )

    mix_dir = (
        run_dir
        / "09_mix"
    )

    final_dir = (
        run_dir
        / "10_final"
    )

    mix_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mix_file = (
        mix_dir
        / "audio_fr_mix.wav"
    )

    final_file = (
        final_dir
        / "video_finale_fr.mp4"
    )

    report_file = (
        final_dir
        / "final.json"
    )

    # ============================================
    # CACHE FINAL
    # ============================================

    if (
        stage_done(video_id, "final")
        and final_file.exists()
    ):

        print(
            "♻️ Vidéo française "
            "déjà disponible"
        )

        print(
            "📁",
            final_file,
        )

        return final_file

    # ============================================
    # TIMELINE FR
    # ============================================

    dialogue_file, placed = (
        build_dialogue_track(
            video_id,
            video_path,
        )
    )

    # ============================================
    # MIXAGE
    # ============================================

    print()

    separated_background = (
        run_dir
        / "09_mix"
        / "separation"
        / "no_vocals.wav"
    )

    if separated_background.exists():

        print(
            "🎚️ Mixage fond séparé + voix FR..."
        )

        print(
            "🎵 Utilisation du fond sans voix japonaises"
        )

        filter_complex = (
            "[0:a:0]"
            "aresample=48000,"
            "aformat=channel_layouts=stereo,"
            "volume=1.0"
            "[bg];"

            "[1:a:0]"
            "aresample=48000,"
            "aformat=channel_layouts=mono,"
            "pan=stereo|c0=c0|c1=c0,"
            "volume=1.15"
            "[dub];"

            "[bg][dub]"
            "amix="
            "inputs=2:"
            "duration=longest:"
            "dropout_transition=0:"
            "normalize=0,"
            "alimiter=limit=0.95"
            "[mixed]"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",

                "-i",
                str(separated_background),

                "-i",
                str(dialogue_file),

                "-filter_complex",
                filter_complex,

                "-map",
                "[mixed]",

                "-ar",
                "48000",

                "-ac",
                "2",

                "-c:a",
                "pcm_s16le",

                str(mix_file),
            ],
            check=True,
        )

    elif video_has_audio(video_path):

        print(
            "⚠️ Pas de séparation Demucs"
        )

        print(
            "🎚️ Fallback : bande originale atténuée + voix FR"
        )

        filter_complex = (
            "[0:a:0]"
            "aresample=48000,"
            "aformat=channel_layouts=stereo"
            "[orig];"

            "[1:a:0]"
            "aresample=48000,"
            "aformat=channel_layouts=mono,"
            "pan=stereo|c0=c0|c1=c0,"
            "volume=1.15,"
            "asplit=2"
            "[dub_sc][dub_mix];"

            "[orig][dub_sc]"
            "sidechaincompress="
            "threshold=0.012:"
            "ratio=14:"
            "attack=5:"
            "release=220:"
            "makeup=1"
            "[ducked];"

            "[ducked][dub_mix]"
            "amix="
            "inputs=2:"
            "duration=first:"
            "dropout_transition=0:"
            "normalize=0,"
            "alimiter=limit=0.95"
            "[mixed]"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",

                "-i",
                str(video_path),

                "-i",
                str(dialogue_file),

                "-filter_complex",
                filter_complex,

                "-map",
                "[mixed]",

                "-ar",
                "48000",

                "-ac",
                "2",

                "-c:a",
                "pcm_s16le",

                str(mix_file),
            ],
            check=True,
        )

    else:

        print(
            "🎚️ Vidéo sans bande audio : voix FR uniquement"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",

                "-i",
                str(dialogue_file),

                "-ar",
                "48000",

                "-ac",
                "2",

                "-c:a",
                "pcm_s16le",

                str(mix_file),
            ],
            check=True,
        )

    mark_stage_completed(
        video_id,
        "mix",
        mix_file,
    )

    print(
        "✅ Mix audio terminé"
    )
    print(
        "📁",
        mix_file,
    )

    # ============================================
    # REMUX VIDÉO
    # ============================================

    print()
    print(
        "🎬 Création MP4 français..."
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",

            "-i",
            str(video_path),

            "-i",
            str(mix_file),

            "-map",
            "0:v:0",

            "-map",
            "1:a:0",

            # Pas de réencodage vidéo.
            "-c:v",
            "copy",

            "-c:a",
            "aac",

            "-b:a",
            "192k",

            "-movflags",
            "+faststart",

            "-shortest",

            str(final_file),
        ],
        check=True,
    )

    mark_stage_completed(
        video_id,
        "final",
        final_file,
    )

    report = {
        "version": 1,
        "video_id": video_id,
        "source_video":
            str(video_path),
        "dialogue_track":
            str(dialogue_file),
        "mix_track":
            str(mix_file),
        "final_video":
            str(final_file),
        "segments":
            len(placed),
        "status":
            "completed",
    }

    save_json(
        report_file,
        report,
    )

    print()
    print(
        "🎉 VIDÉO FRANÇAISE TERMINÉE"
    )

    print(
        "Répliques :",
        len(placed),
    )

    print(
        "📁",
        final_file,
    )

    return final_file
