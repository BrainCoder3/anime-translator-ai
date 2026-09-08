
import json
import shutil
import subprocess
from pathlib import Path

from src.core.run_manager import (
    get_run_dir,
)


MAX_SAFE_RATIO = 1.30
TOLERANCE = 1.08


def load_json(path):
    with open(
        path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


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


def audio_duration(path):

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

    return float(
        result.stdout.strip()
    )


def finalize_tts_timing(video_id):

    run_dir = get_run_dir(
        video_id
    )

    tts_dir = (
        run_dir
        / "08_tts"
    )

    tts_file = (
        tts_dir
        / "tts.json"
    )

    final_dir = (
        tts_dir
        / "final"
    )

    report_file = (
        tts_dir
        / "timing_final.json"
    )

    final_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not tts_file.exists():
        raise FileNotFoundError(
            tts_file
        )

    data = load_json(
        tts_file
    )

    results = []

    accelerated = 0
    unchanged = 0
    failures = []

    print()
    print(
        "⏱️ FINALISATION TIMING TTS"
    )
    print(
        "=" * 50
    )

    for item in data["segments"]:

        seg_id = int(
            item["id"]
        )

        src = Path(
            item["output_path"]
        )

        if not src.exists():
            raise FileNotFoundError(
                src
            )

        dst = (
            final_dir
            / f"{seg_id:04d}.wav"
        )

        generated = audio_duration(
            src
        )

        available = float(
            item.get(
                "available_seconds",
                item[
                    "target_seconds"
                ],
            )
        )

        ratio = (
            generated
            / available
            if available > 0
            else 1.0
        )

        # ====================================
        # Cas anormal : le TTS aurait dû
        # être régénéré avant cette étape.
        # ====================================

        if ratio > MAX_SAFE_RATIO:

            failures.append({
                "id": seg_id,
                "ratio": round(
                    ratio,
                    3,
                ),
            })

            print(
                f"❌ {seg_id:03d} "
                f"{ratio:.2f}x "
                "→ dépasse 1.30x"
            )

            continue

        # ====================================
        # Petite accélération naturelle
        # ====================================

        if ratio > TOLERANCE:

            speed = ratio

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",

                    "-i",
                    str(src),

                    "-filter:a",
                    f"atempo={speed:.6f}",

                    "-ar",
                    "24000",

                    "-ac",
                    "1",

                    "-c:a",
                    "pcm_s16le",

                    str(dst),
                ],
                check=True,
            )

            accelerated += 1

            action = (
                f"speed_{speed:.3f}x"
            )

            icon = "⚡"

        else:

            # Conversion WAV homogène.
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",

                    "-i",
                    str(src),

                    "-ar",
                    "24000",

                    "-ac",
                    "1",

                    "-c:a",
                    "pcm_s16le",

                    str(dst),
                ],
                check=True,
            )

            unchanged += 1

            action = "unchanged"

            icon = "✅"

        final_duration = (
            audio_duration(
                dst
            )
        )

        final_ratio = (
            final_duration
            / available
            if available > 0
            else 1.0
        )

        print(
            f"{icon} {seg_id:03d} | "
            f"{generated:.2f}s → "
            f"{final_duration:.2f}s | "
            f"fenêtre={available:.2f}s | "
            f"{final_ratio:.2f}x"
        )

        results.append({
            **item,

            "source_path":
                str(src),

            "final_path":
                str(dst),

            "before_seconds":
                round(
                    generated,
                    3,
                ),

            "final_seconds":
                round(
                    final_duration,
                    3,
                ),

            "available_seconds":
                round(
                    available,
                    3,
                ),

            "ratio_before":
                round(
                    ratio,
                    3,
                ),

            "ratio_final":
                round(
                    final_ratio,
                    3,
                ),

            "timing_action":
                action,
        })

    if failures:

        save_json(
            report_file,
            {
                "video_id":
                    video_id,

                "status":
                    "failed",

                "failures":
                    failures,
            },
        )

        raise RuntimeError(
            "Certains segments dépassent "
            "encore 1.30x. "
            "Ils doivent être régénérés."
        )

    report = {
        "version": 1,

        "video_id":
            video_id,

        "status":
            "completed",

        "segments":
            results,

        "summary": {
            "total":
                len(results),

            "unchanged":
                unchanged,

            "accelerated":
                accelerated,

            "failures":
                0,
        },
    }

    save_json(
        report_file,
        report,
    )

    print()
    print(
        "✅ TIMING TTS FINALISÉ"
    )

    print(
        "Audios inchangés :",
        unchanged,
    )

    print(
        "Audios accélérés :",
        accelerated,
    )

    print(
        "Échecs            : 0"
    )

    print(
        "📁",
        final_dir,
    )

    return report
