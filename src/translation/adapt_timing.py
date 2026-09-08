
import re

import torch

from src.core.run_manager import (
    get_run,
    get_run_dir,
    stage_done,
    mark_stage_completed,
)

from src.core.anime_memory import (
    load_json,
    save_json,
)

from src.translation.runtime import (
    MODEL_NAME,
    load_translation_runtime,
)


# Estimation initiale.
# Le TTS réel affinera ensuite le timing.
TARGET_CPS = 14.0

# On ne réécrit pas une phrase pour
# quelques millisecondes de dépassement.
TOLERANCE = 1.12

CHUNK_SIZE = 6


def clean_length(text):
    text = re.sub(
        r"\s+",
        " ",
        text.strip(),
    )

    return len(text)


def estimate_seconds(text):
    return (
        clean_length(text)
        / TARGET_CPS
    )


def parse_lines(text, expected_ids):
    results = {}

    for raw in text.splitlines():

        line = raw.strip()

        if "|||" not in line:
            continue

        left, value = line.split(
            "|||",
            1,
        )

        left = left.strip().rstrip(".:")

        try:
            item_id = int(left)
        except ValueError:
            continue

        value = value.strip()

        if (
            item_id in expected_ids
            and value
        ):
            results[item_id] = value

    return results


def adapt_timing(
    video_id,
    model=None,
    tokenizer=None,
):

    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    translation_file = (
        run_dir
        / "05_translation"
        / "translation.json"
    )

    output_dir = (
        run_dir
        / "06_adaptation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "adaptation.json"
    )

    partial_file = (
        output_dir
        / "adaptation_partial.json"
    )

    # ==============================
    # CACHE
    # ==============================

    if (
        stage_done(video_id, "adaptation")
        and output_file.exists()
    ):
        print(
            "♻️ Adaptation déjà disponible"
        )
        print("📁", output_file)

        return load_json(output_file)

    if not translation_file.exists():
        raise FileNotFoundError(
            translation_file
        )

    translation = load_json(
        translation_file
    )

    segments = translation["segments"]

    # ==============================
    # ANALYSE DES DURÉES
    # ==============================

    pending = []

    for seg in segments:

        duration = float(
            seg.get(
                "duration",
                float(seg["end"])
                - float(seg["start"]),
            )
        )

        text = seg["translation_fr"]

        estimated = estimate_seconds(
            text
        )

        max_chars = max(
            6,
            int(duration * TARGET_CPS),
        )

        if (
            estimated
            > duration * TOLERANCE
        ):
            pending.append({
                "id": int(seg["id"]),
                "japanese": seg["text_ja"],
                "french": text,
                "duration": round(
                    duration,
                    3,
                ),
                "estimated_seconds": round(
                    estimated,
                    3,
                ),
                "max_chars": max_chars,
                "character_id": seg.get(
                    "character_id"
                ),
            })

    print(
        "⏱️ Segments trop longs :",
        len(pending),
        "/",
        len(segments),
    )

    # ==============================
    # CHECKPOINT
    # ==============================

    adapted_by_id = {}

    if partial_file.exists():

        partial = load_json(
            partial_file,
            {"adapted": {}},
        )

        adapted_by_id = {
            int(k): v
            for k, v
            in partial.get(
                "adapted",
                {}
            ).items()
        }

        print(
            "♻️ Adaptation partielle :",
            len(adapted_by_id),
        )

    # Aucun segment à raccourcir
    if not pending:

        result_segments = []

        for seg in segments:

            result_segments.append({
                **seg,
                "adapted_fr":
                    seg["translation_fr"],
                "timing_adapted": False,
            })

        data = {
            "version": 1,
            "video_id": video_id,
            "model": MODEL_NAME,
            "target_cps": TARGET_CPS,
            "segments": result_segments,
        }

        save_json(
            output_file,
            data,
        )

        mark_stage_completed(
            video_id,
            "adaptation",
            output_file,
        )

        return data

    # ==============================
    # MODÈLE
    # ==============================

    owns_model = False

    if (
        model is None
        or tokenizer is None
    ):

        model, tokenizer = (
            load_translation_runtime()
        )

        owns_model = True

    # ==============================
    # ADAPTATION PAR BLOCS
    # ==============================

    for start in range(
        0,
        len(pending),
        CHUNK_SIZE,
    ):

        chunk = pending[
            start:start + CHUNK_SIZE
        ]

        chunk = [
            item
            for item in chunk
            if item["id"]
            not in adapted_by_id
        ]

        if not chunk:
            continue

        prompt = f"""
Tu adaptes des dialogues français destinés
au doublage d'un anime.

Chaque traduction a déjà été vérifiée.
Ton travail est UNIQUEMENT de la rendre
assez courte pour tenir naturellement dans
le temps disponible.

RÈGLES ABSOLUES :
- conserve le sens ;
- conserve les nombres ;
- conserve les noms propres ;
- conserve l'intention et le registre ;
- français oral naturel ;
- ne rajoute aucune information ;
- raccourcis seulement ce qui est nécessaire ;
- n'utilise pas d'abréviations artificielles ;
- une phrase déjà concise doit rester naturelle.

"max_chars" est un OBJECTIF pratique,
pas une raison de produire du mauvais français.

Données :

{chunk}

Réponds STRICTEMENT :

ID|||phrase française adaptée

Une ligne par ID.
Rien d'autre.
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu adaptes des dialogues "
                    "français pour le doublage."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            k: v.to(device)
            for k, v
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
            )

        generated = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        expected = {
            item["id"]
            for item in chunk
        }

        parsed = parse_lines(
            answer,
            expected,
        )

        print()
        print(
            "🎬 Adaptation timing"
        )

        for item in chunk:

            item_id = item["id"]

            original = item["french"]

            adapted = parsed.get(
                item_id,
                original,
            )

            # Si le modèle produit quelque
            # chose d'encore plus long,
            # on garde la version précédente.
            if (
                clean_length(adapted)
                > clean_length(original)
            ):
                adapted = original

            adapted_by_id[
                item_id
            ] = adapted

            before = estimate_seconds(
                original
            )

            after = estimate_seconds(
                adapted
            )

            print(
                f'{item_id:03d}. '
                f'{item["duration"]:.2f}s'
            )

            print(
                "     Avant :",
                original,
                f"(~{before:.2f}s)"
            )

            print(
                "     Après :",
                adapted,
                f"(~{after:.2f}s)"
            )

        save_json(
            partial_file,
            {
                "version": 1,
                "video_id": video_id,
                "adapted": {
                    str(k): v
                    for k, v
                    in adapted_by_id.items()
                },
            },
        )

        print(
            "💾 Checkpoint adaptation :",
            len(adapted_by_id),
            "/",
            len(pending),
        )

    # ==============================
    # CONSTRUCTION FINALE
    # ==============================

    result_segments = []

    too_long_after = 0

    for seg in segments:

        item_id = int(
            seg["id"]
        )

        original = seg[
            "translation_fr"
        ]

        adapted = adapted_by_id.get(
            item_id,
            original,
        )

        duration = float(
            seg.get(
                "duration",
                float(seg["end"])
                - float(seg["start"]),
            )
        )

        estimated = estimate_seconds(
            adapted
        )

        ratio = (
            estimated / duration
            if duration > 0
            else 1.0
        )

        if ratio > 1.25:
            status = "still_long"
            too_long_after += 1

        elif item_id in adapted_by_id:
            status = "adapted"

        else:
            status = "ok"

        result_segments.append({
            **seg,

            "adapted_fr": adapted,

            "timing": {
                "target_seconds": round(
                    duration,
                    3,
                ),

                "estimated_seconds": round(
                    estimated,
                    3,
                ),

                "estimated_ratio": round(
                    ratio,
                    3,
                ),

                "status": status,
            },

            "timing_adapted": (
                item_id
                in adapted_by_id
            ),
        })

    data = {
        "version": 1,
        "video_id": video_id,
        "anime_id": run["anime_id"],
        "model": MODEL_NAME,
        "target_cps": TARGET_CPS,
        "segments": result_segments,
    }

    save_json(
        output_file,
        data,
    )

    if partial_file.exists():
        partial_file.unlink()

    mark_stage_completed(
        video_id,
        "adaptation",
        output_file,
    )

    print()
    print(
        "✅ Adaptation temporelle terminée"
    )

    print(
        "⚠️ Encore trop longs :",
        too_long_after,
    )

    print(
        "📁",
        output_file,
    )

    return data
