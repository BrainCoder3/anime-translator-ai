
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


# Seuils conservateurs pour éviter de fusionner
# accidentellement deux personnages différents.
MATCH_THRESHOLD = 0.55
MATCH_MARGIN = 0.08
UPDATE_EMBEDDING_THRESHOLD = 0.65


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


def cosine(a, b):
    a = normalize(a)
    b = normalize(b)

    return float(np.dot(a, b))


def next_character_id(characters):
    numbers = []

    for character_id in characters:
        try:
            numbers.append(
                int(character_id.split("_")[-1])
            )
        except Exception:
            pass

    number = max(numbers, default=0) + 1

    return f"CHARACTER_{number:03d}"


def match_or_register_characters(video_id):

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    anime_id = run["anime_id"]
    anime_dir = get_anime_dir(anime_id)

    mapping_file = (
        run_dir
        / "04_characters"
        / "speaker_character_map.json"
    )

    # ===================================
    # CACHE
    # ===================================

    if mapping_file.exists():

        data = load_json(mapping_file)

        print("♻️ Mapping personnages déjà disponible")

        for speaker, info in data["mappings"].items():
            print(
                f'{speaker} → {info["character_id"]}'
            )

        return data

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

    characters_file = (
        anime_dir
        / "characters.json"
    )

    memory_embedding_dir = (
        anime_dir
        / "voice_embeddings"
    )

    memory_embedding_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis = load_json(analysis_file)
    diar = load_json(diar_file)

    character_db = load_json(
        characters_file,
        {
            "version": 1,
            "characters": {},
        }
    )

    current_embeddings = np.load(
        embeddings_file
    )

    embedding_order = diar.get(
        "speaker_embedding_order",
        diar["speakers"],
    )

    current_lookup = {
        speaker: normalize(current_embeddings[index])
        for index, speaker
        in enumerate(embedding_order)
    }

    # ===================================
    # EMBEDDINGS DÉJÀ CONNUS
    # ===================================

    known_embeddings = {}

    for character_id, character in (
        character_db["characters"].items()
    ):

        relative = character.get(
            "voice_embedding"
        )

        if not relative:
            continue

        path = anime_dir / relative

        if not path.exists():
            continue

        known_embeddings[character_id] = (
            normalize(np.load(path))
        )

    # Les speakers les plus importants
    # sont résolus d'abord.
    ranked_speakers = sorted(
        analysis["speakers"].items(),
        key=lambda item: item[1][
            "diarization_duration"
        ],
        reverse=True,
    )

    mappings = {}

    # Évite d'attribuer deux speakers Pyannote
    # différents au même personnage dans une vidéo.
    used_characters = set()

    for speaker, info in ranked_speakers:

        vector = current_lookup[speaker]

        candidates = []

        for character_id, memory_vector in (
            known_embeddings.items()
        ):

            if character_id in used_characters:
                continue

            score = cosine(
                vector,
                memory_vector,
            )

            candidates.append(
                (character_id, score)
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        matched_character = None
        best_score = None
        second_score = None

        if candidates:

            matched_character = candidates[0][0]
            best_score = candidates[0][1]

            if len(candidates) > 1:
                second_score = candidates[1][1]
            else:
                second_score = -1.0

            margin = (
                best_score - second_score
            )

            accepted = (
                best_score >= MATCH_THRESHOLD
                and margin >= MATCH_MARGIN
            )

            if not accepted:
                matched_character = None

        # ===================================
        # PERSONNAGE CONNU
        # ===================================

        if matched_character is not None:

            character_id = matched_character
            used_characters.add(character_id)

            character = character_db[
                "characters"
            ][character_id]

            character.setdefault(
                "observations",
                []
            )

            character["observations"].append({
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
                "match_score": round(
                    best_score,
                    4,
                ),
            })

            character[
                "total_clean_voice_seconds"
            ] = round(
                float(
                    character.get(
                        "total_clean_voice_seconds",
                        0.0,
                    )
                )
                + float(
                    info["clean_duration"]
                ),
                3,
            )

            character["updated_at"] = now()

            # Mise à jour prudente de l'empreinte
            # uniquement si la correspondance est forte.
            if (
                best_score
                >= UPDATE_EMBEDDING_THRESHOLD
            ):

                old_vector = known_embeddings[
                    character_id
                ]

                updated = normalize(
                    old_vector + vector
                )

                embedding_path = (
                    anime_dir
                    / character[
                        "voice_embedding"
                    ]
                )

                np.save(
                    embedding_path,
                    updated,
                )

                known_embeddings[
                    character_id
                ] = updated

            mappings[speaker] = {
                "character_id": character_id,
                "action": "matched",
                "similarity": round(
                    best_score,
                    4,
                ),
                "clean_voice_seconds": round(
                    float(
                        info["clean_duration"]
                    ),
                    3,
                ),
            }

        # ===================================
        # NOUVEAU PERSONNAGE
        # ===================================

        else:

            character_id = next_character_id(
                character_db["characters"]
            )

            embedding_filename = (
                f"{character_id}.npy"
            )

            embedding_path = (
                memory_embedding_dir
                / embedding_filename
            )

            np.save(
                embedding_path,
                vector,
            )

            clean_duration = float(
                info["clean_duration"]
            )

            character_db[
                "characters"
            ][character_id] = {
                "character_id": character_id,
                "name": None,
                "status": "provisional",

                "created_at": now(),
                "updated_at": now(),

                "voice_embedding": (
                    "voice_embeddings/"
                    + embedding_filename
                ),

                "voice_reference": None,

                "voice_reference_status": (
                    "ready"
                    if clean_duration >= 5.0
                    else "insufficient_audio"
                ),

                "total_clean_voice_seconds": round(
                    clean_duration,
                    3,
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

            known_embeddings[
                character_id
            ] = vector

            used_characters.add(
                character_id
            )

            mappings[speaker] = {
                "character_id": character_id,
                "action": "created",
                "similarity": (
                    round(best_score, 4)
                    if best_score is not None
                    else None
                ),
                "clean_voice_seconds": round(
                    clean_duration,
                    3,
                ),
            }

    save_json(
        characters_file,
        character_db,
    )

    data = {
        "version": 2,
        "video_id": video_id,
        "anime_id": anime_id,
        "match_threshold": MATCH_THRESHOLD,
        "match_margin": MATCH_MARGIN,
        "mappings": mappings,
    }

    save_json(
        mapping_file,
        data,
    )

    print()
    print("✅ Résolution des personnages terminée")
    print()

    for speaker, item in mappings.items():

        action = (
            "♻️ reconnu"
            if item["action"] == "matched"
            else "🆕 nouveau"
        )

        similarity = item["similarity"]

        score_text = (
            f"{similarity:.3f}"
            if similarity is not None
            else "-"
        )

        print(
            f'{speaker} → '
            f'{item["character_id"]} '
            f'[{action}] '
            f'score={score_text}'
        )

    print()
    print("📁", mapping_file)

    return data
