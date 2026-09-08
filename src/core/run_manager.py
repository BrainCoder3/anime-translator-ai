import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs"

STAGES = [
    "audio",
    "asr",
    "diarization",
    "characters",
    "translation",
    "adaptation",
    "emotions",
    "tts",
    "mix",
    "final",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_video_path(video_path):
    path = Path(video_path)

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Vidéo introuvable : {path}"
        )

    return path


def calculate_video_hash(video_path):
    video_path = resolve_video_path(video_path)

    sha256 = hashlib.sha256()

    with open(video_path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def prepare_run(video_path, anime_id):
    video_path = resolve_video_path(video_path)

    full_hash = calculate_video_hash(video_path)
    video_id = full_hash[:16]

    run_dir = RUNS_ROOT / video_id
    metadata_file = run_dir / "run.json"

    # ---------------------------------
    # Run déjà existant
    # ---------------------------------

    if metadata_file.exists():
        metadata = load_json(metadata_file)

        old_anime = metadata["anime_id"]

        if old_anime != anime_id:
            raise RuntimeError(
                f"La vidéo {video_id} est déjà associée "
                f"à {old_anime}, pas à {anime_id}. "
                "Refus pour éviter de mélanger les mémoires."
            )

        print(
            f"♻️ Run existant : {video_id}"
        )

        return metadata

    # ---------------------------------
    # Nouveau run
    # ---------------------------------

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    directories = {
        "audio": "01_audio",
        "asr": "02_asr",
        "diarization": "03_diarization",
        "characters": "04_characters",
        "translation": "05_translation",
        "adaptation": "06_adaptation",
        "emotions": "07_emotions",
        "tts": "08_tts",
        "mix": "09_mix",
        "final": "10_final",
    }

    for directory in directories.values():
        (run_dir / directory).mkdir(
            parents=True,
            exist_ok=True
        )

    metadata = {
        "schema_version": 1,
        "video_id": video_id,
        "sha256": full_hash,
        "video_path": str(video_path),
        "anime_id": anime_id,
        "created_at": now(),
        "updated_at": now(),

        "directories": directories,

        "stages": {
            stage: {
                "status": "pending",
                "completed_at": None,
                "output": None
            }
            for stage in STAGES
        }
    }

    save_json(
        metadata_file,
        metadata
    )

    print(
        f"🆕 Nouveau run : {video_id}"
    )

    return metadata


def get_run(video_id):
    metadata_file = (
        RUNS_ROOT
        / video_id
        / "run.json"
    )

    if not metadata_file.exists():
        raise ValueError(
            f"Run inconnu : {video_id}"
        )

    return load_json(metadata_file)


def get_run_dir(video_id):
    run_dir = RUNS_ROOT / video_id

    if not run_dir.exists():
        raise ValueError(
            f"Run inconnu : {video_id}"
        )

    return run_dir


def stage_done(video_id, stage):
    run = get_run(video_id)

    if stage not in run["stages"]:
        raise ValueError(
            f"Étape inconnue : {stage}"
        )

    return (
        run["stages"][stage]["status"]
        == "completed"
    )


def mark_stage_completed(
    video_id,
    stage,
    output=None
):
    run = get_run(video_id)

    if stage not in run["stages"]:
        raise ValueError(
            f"Étape inconnue : {stage}"
        )

    run["stages"][stage] = {
        "status": "completed",
        "completed_at": now(),
        "output": (
            str(output)
            if output is not None
            else None
        )
    }

    run["updated_at"] = now()

    save_json(
        RUNS_ROOT / video_id / "run.json",
        run
    )


def reset_stage(video_id, stage):
    run = get_run(video_id)

    if stage not in run["stages"]:
        raise ValueError(
            f"Étape inconnue : {stage}"
        )

    run["stages"][stage] = {
        "status": "pending",
        "completed_at": None,
        "output": None
    }

    run["updated_at"] = now()

    save_json(
        RUNS_ROOT / video_id / "run.json",
        run
    )
