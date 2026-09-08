
import gc

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.core.run_manager import (
    get_run,
    get_run_dir,
)

from src.core.anime_memory import (
    get_anime_dir,
    load_json,
    save_json,
)


MODEL_NAME = "Qwen/Qwen3-8B"
CHUNK_SIZE = 6
REVIEW_VERSION = 2


def parse_lines(text, expected_ids):

    # Qwen3 thinking mode :
    # on ignore son raisonnement interne.
    if "</think>" in text:
        text = text.rsplit(
            "</think>",
            1,
        )[-1]

    results = {}

    for raw in text.splitlines():
        line = raw.strip()

        if "|||" not in line:
            continue

        left, right = line.split(
            "|||",
            1,
        )

        left = (
            left.strip()
            .rstrip(".:")
        )

        try:
            item_id = int(left)
        except ValueError:
            continue

        translation = right.strip()

        if (
            item_id in expected_ids
            and translation
        ):
            results[item_id] = translation

    return results


def review_translation(
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

    partial_file = (
        run_dir
        / "05_translation"
        / "translation_review_partial.json"
    )

    if not translation_file.exists():
        raise FileNotFoundError(
            translation_file
        )

    data = load_json(translation_file)

    # =========================
    # CACHE
    # =========================

    if (
        data.get("reviewed") is True
        and data.get("review_model") == MODEL_NAME
        and data.get("review_version") == REVIEW_VERSION
    ):
        print(
            "♻️ Traduction déjà vérifiée"
        )
        print("📁", translation_file)

        return data

    anime_dir = get_anime_dir(
        run["anime_id"]
    )

    glossary = load_json(
        anime_dir / "glossary.json",
        {"terms": {}},
    )

    context = load_json(
        anime_dir / "context.json",
        {},
    )

    segments = data["segments"]

    # =========================
    # CHECKPOINT
    # =========================

    reviewed_by_id = {}

    if partial_file.exists():
        partial = load_json(
            partial_file,
            {"segments": []},
        )

        for item in partial["segments"]:
            reviewed_by_id[
                int(item["id"])
            ] = item["translation_fr"]

        print(
            "♻️ Review partielle récupérée :",
            len(reviewed_by_id),
            "segments",
        )

    # =========================
    # LIBÉRER VRAM
    # =========================

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================
    # MODÈLE
    # =========================

    if tokenizer is None:

        print(
            "📥 Reviewer tokenizer :",
            MODEL_NAME
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                MODEL_NAME
            )
        )

    if model is None:

        print(
            "🧠 Reviewer traduction :",
            MODEL_NAME
        )

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.float16
                ),
                bnb_4bit_use_double_quant=True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                MODEL_NAME,
                quantization_config=(
                    quantization_config
                ),
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        )

        model.eval()

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    # =========================
    # REVIEW
    # =========================

    for start in range(
        0,
        len(segments),
        CHUNK_SIZE,
    ):

        chunk = segments[
            start:start + CHUNK_SIZE
        ]

        pending = [
            item
            for item in chunk
            if int(item["id"])
            not in reviewed_by_id
        ]

        if not pending:
            continue

        payload = []

        for item in pending:

            payload.append({
                "id": int(item["id"]),
                "japanese": item["text_ja"],
                "draft_fr": item.get(
                    "translation_fr_draft",
                    item["translation_fr"],
                ),
                "character_id": item.get(
                    "character_id"
                ),
                "asr_uncertain": item.get(
                    "needs_asr_review",
                    False,
                ),
            })

        prompt = f"""
Tu es le contrôleur qualité FINAL d'un doublage
japonais vers le français.

IMPORTANT :
la traduction française fournie est un BROUILLON
NON FIABLE.

Pour chaque ID :

1. Traduis d'abord mentalement le japonais ORIGINAL,
   indépendamment du brouillon.
2. Vérifie ensuite le brouillon.
3. Produis la meilleure traduction française finale.

La source japonaise est TOUJOURS prioritaire.

Vérifications obligatoires :

- sens exact ;
- négations ;
- sujet et intention ;
- nombres ;
- quantités ;
- registre du personnage ;
- français naturel à l'oral ;
- grammaire française ;
- interjections japonaises ;
- mots japonais laissés en romaji ;
- mots anglais laissés inutilement ;
- cohérence des noms propres.

RÈGLES NUMÉRIQUES JAPONAISES :
- 万 = 10 000
- 10万 = 100 000
- 100万 = 1 000 000
- 300万 = 3 000 000
- 億 = 100 000 000

Ne laisse donc JAMAIS "man" dans une traduction
française lorsqu'il représente l'unité 万.

RÈGLE DE LANGUE :
La sortie doit être du français.
Ne laisse pas des mots japonais romanisés comme
une traduction de vocabulaire courant ou
d'interjections.

Par exemple, une interjection japonaise doit être
rendue naturellement en français selon le contexte,
pas simplement romanisée.

Exception :
un véritable nom propre ou terme spécifique de
l'univers peut rester un nom propre.

Si le japonais est manifestement incertain à cause
de l'ASR, choisis une formulation prudente et
n'invente pas de détails.

Glossaire :
{glossary}

Contexte de série :
{context}

Lignes à contrôler :
{payload}

Réponds STRICTEMENT à la fin avec :

ID|||traduction française finale

Une ligne par ID.
Tous les ID doivent apparaître.
Aucun commentaire dans la réponse finale.
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu vérifies et corriges "
                    "des traductions japonaises "
                    "vers le français."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        inputs = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )

        generated = output[0][
            inputs["input_ids"].shape[-1]:
        ]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        expected_ids = {
            int(item["id"])
            for item in pending
        }

        parsed = parse_lines(
            answer,
            expected_ids,
        )

        print()
        print(
            f"🔍 Review "
            f"{start + 1}"
            f" → "
            f"{start + len(chunk)}"
            f" / {len(segments)}"
        )

        for item in pending:

            item_id = int(
                item["id"]
            )

            draft = item[
                "translation_fr"
            ]

            # Si le reviewer oublie une ligne,
            # on conserve le draft au lieu
            # de planter le pipeline.
            final = parsed.get(
                item_id,
                draft,
            )

            reviewed_by_id[
                item_id
            ] = final

            changed = (
                "🔧"
                if final != draft
                else "✅"
            )

            print(
                f"{item_id:03d}. "
                f"{changed} "
                f"{final}"
            )

        partial_segments = [
            {
                "id": item_id,
                "translation_fr":
                    reviewed_by_id[item_id],
            }
            for item_id
            in sorted(reviewed_by_id)
        ]

        save_json(
            partial_file,
            {
                "version": 1,
                "video_id": video_id,
                "segments":
                    partial_segments,
            },
        )

        print(
            "💾 Review checkpoint :",
            len(reviewed_by_id),
            "/",
            len(segments),
        )

    # =========================
    # VERSION FINALE
    # =========================

    for item in segments:

        item_id = int(
            item["id"]
        )

        draft = item[
            "translation_fr"
        ]

        final = reviewed_by_id.get(
            item_id,
            draft,
        )

        item[
            "translation_fr_draft"
        ] = draft

        item[
            "translation_fr"
        ] = final

    data["segments"] = segments
    data["reviewed"] = True
    data["review_model"] = MODEL_NAME
    data["review_version"] = REVIEW_VERSION

    save_json(
        translation_file,
        data,
    )

    if partial_file.exists():
        partial_file.unlink()

    print()
    print(
        "✅ Review traduction terminée"
    )

    print(
        "📁",
        translation_file,
    )

    return data
