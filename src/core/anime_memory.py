import json
from pathlib import Path
from datetime import datetime, timezone


# src/core/anime_memory.py -> racine du projet
ROOT = Path(__file__).resolve().parents[2]

MEMORY_ROOT = ROOT / "anime_memory"
REGISTRY_FILE = MEMORY_ROOT / "registry.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default=None):
    path = Path(path)

    if not path.exists():
        return default

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
            indent=2
        )


def load_registry():
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

    return load_json(
        REGISTRY_FILE,
        {
            "version": 1,
            "anime": {}
        }
    )


def save_registry(registry):
    save_json(REGISTRY_FILE, registry)


def list_anime():
    return load_registry()["anime"]


def anime_exists(anime_id):
    return anime_id in load_registry()["anime"]


def create_anime(name):
    registry = load_registry()

    numbers = []

    for anime_id in registry["anime"]:
        try:
            numbers.append(
                int(anime_id.split("_")[-1])
            )
        except (ValueError, IndexError):
            pass

    next_number = max(numbers, default=0) + 1
    anime_id = f"anime_{next_number:03d}"

    anime_dir = MEMORY_ROOT / anime_id

    (anime_dir / "voice_references").mkdir(
        parents=True,
        exist_ok=True
    )

    (anime_dir / "voice_embeddings").mkdir(
        parents=True,
        exist_ok=True
    )

    anime_data = {
        "anime_id": anime_id,
        "name": name,
        "created_at": now(),
        "updated_at": now(),
        "episodes_processed": 0
    }

    registry["anime"][anime_id] = anime_data
    save_registry(registry)

    save_json(
        anime_dir / "anime.json",
        anime_data
    )

    save_json(
        anime_dir / "characters.json",
        {
            "version": 1,
            "characters": {}
        }
    )

    save_json(
        anime_dir / "glossary.json",
        {
            "version": 1,
            "terms": {}
        }
    )

    save_json(
        anime_dir / "context.json",
        {
            "version": 1,
            "series_summary": "",
            "important_events": []
        }
    )

    return anime_id


def get_anime(anime_id):
    registry = load_registry()

    if anime_id not in registry["anime"]:
        raise ValueError(
            f"Anime inconnu : {anime_id}"
        )

    return registry["anime"][anime_id]


def get_anime_dir(anime_id):
    get_anime(anime_id)
    return MEMORY_ROOT / anime_id
