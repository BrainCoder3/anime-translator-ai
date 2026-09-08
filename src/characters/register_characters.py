
import json
import numpy as np

from src.core.run_manager import (
    get_run,
    get_run_dir,
)

from src.core.anime_memory import (
    get_anime_dir,
    load_json,
    save_json,
    now,
)


def register_characters(video_id):

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    anime_id = run["anime_id"]
    anime_dir = get_anime_dir(anime_id)

    analysis_file = (
        run_dir
        / "04_characters"
        / "speaker_analysis.json"
    )

    diar_file = (
        run_dir
        / "03_diarization"
        / "diarization.json"
    )

    embeddings_file = (
        run_dir
        / "03_diarization"
        / "speaker_embeddings.npy"
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

    embedding_dir = (
        anime_dir
        / "voice_embeddings"
    )

    embedding_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================
    # CACHE : éviter de recréer les persos
    # ======================================

    if mapping_file.exists():

        print("♻️ Personnages déjà enregistrés")

        with open(
            mapping_file,
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        for speaker, item in data["mappings"].items():
            print(
                speaker,
                "→",
                item["character_id"],
            )

        return data

    # ======================================
    # CHARGEMENTS
    # ======================================

    analysis = load_json(
        analysis_file,
        {"speakers": {}},
    )

    diar = load_json(
        diar_file,
        {},
    )

    character_db = load_json(
        characters_file,
        {
            "version": 1,
            "characters": {},
        },
    )

    embeddings = np.load(
        embeddings_file
    )

    embedding_order = diar.get(
        "speaker_embedding_order",
        diar["speakers"],
    )

    embedding_lookup = {
        speaker: embeddings[index]
        for index, speaker
        in enumerate(embedding_order)
    }

    # ======================================
    # ORDRE = quantité de parole
    # ======================================

    ranked_speakers = sorted(
        analysis["speakers"].items(),
        key=lambda item: item[1][
            "diarization_duration"
        ],
        reverse=True,
    )

    # ======================================
    # PROCHAIN ID PERSONNAGE
    # ======================================

    existing_numbers = []

    for character_id in character_db["characters"]:
        try:
            existing_numbers.append(
                int(
                    character_id.split("_")[-1]
                )
            )
        except Exception:
            pass

    next_number = (
        max(existing_numbers, default=0)
        + 1
    )

    mappings = {}

    # ======================================
    # CRÉATION
    # ======================================

    for speaker, info in ranked_speakers:

        character_id = (
            f"CHARACTER_{next_number:03d}"
        )

        next_number += 1

        vector = np.asarray(
            embedding_lookup[speaker],
            dtype=np.float32,
        )

        # Normalisation pour comparaison future
        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        embedding_filename = (
            f"{character_id}.npy"
        )

        embedding_path = (
            embedding_dir
            / embedding_filename
        )

        np.save(
            embedding_path,
            vector,
        )

        clean_duration = float(
            info["clean_duration"]
        )

        if clean_duration >= 5.0:
            reference_status = "ready"
        else:
            reference_status = (
                "insufficient_audio"
            )

        character = {
            "character_id": character_id,

            # Le vrai nom sera déterminé
            # plus tard avec le contexte.
            "name": None,

            "status": "provisional",

            "created_at": now(),
            "updated_at": now(),

            "voice_embedding": (
                f"voice_embeddings/"
                f"{embedding_filename}"
            ),

            "voice_reference": None,

            "voice_reference_status": (
                reference_status
            ),

            "total_clean_voice_seconds": (
                round(clean_duration, 3)
            ),

            "observations": [
                {
                    "video_id": video_id,
                    "speaker": speaker,
                    "speech_seconds": round(
                        float(
                            info[
                                "diarization_duration"
                            ]
                        ),
                        3,
                    ),
                }
            ],
        }

        character_db[
            "characters"
        ][character_id] = character

        mappings[speaker] = {
            "character_id": character_id,
            "clean_voice_seconds": round(
                clean_duration,
                3,
            ),
            "voice_reference_status": (
                reference_status
            ),
        }

    # ======================================
    # SAUVEGARDE MÉMOIRE ANIME
    # ======================================

    save_json(
        characters_file,
        character_db,
    )

    data = {
        "version": 1,
        "video_id": video_id,
        "anime_id": anime_id,
        "mappings": mappings,
    }

    save_json(
        mapping_file,
        data,
    )

    # ======================================
    # AFFICHAGE
    # ======================================

    print(
        "✅ Personnages enregistrés "
        "dans la mémoire de l'anime"
    )

    print()

    for speaker, item in mappings.items():

        status = (
            "✅ voix exploitable"
            if item[
                "voice_reference_status"
            ] == "ready"
            else "⚠️ référence trop courte"
        )

        print(
            f'{speaker} → '
            f'{item["character_id"]}'
        )

        print(
            f'   Audio propre : '
            f'{item["clean_voice_seconds"]:.2f}s'
        )

        print(
            f'   {status}'
        )

    print()
    print(
        "📁",
        characters_file,
    )

    return data
