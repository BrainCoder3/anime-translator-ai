
import json
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.core.run_manager import (
    get_run,
    get_run_dir,
    stage_done,
    mark_stage_completed,
)

from src.core.anime_memory import (
    get_anime_dir,
    load_json,
    save_json,
)


MODEL_NAME = "Qwen/Qwen3-8B"
CHUNK_SIZE = 6


def parse_translation_lines(text, expected_ids):
    """
    Format attendu :
    ID|||traduction française

    Exemple :
    25|||Il faut sérieusement y réfléchir.
    """

    results = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or "|||" not in line:
            continue

        left, translation = line.split(
            "|||",
            1,
        )

        left = left.strip()

        # Tolère "025", "25.", "25:"
        left = left.rstrip(".:")

        try:
            item_id = int(left)
        except ValueError:
            continue

        translation = translation.strip()

        if (
            item_id in expected_ids
            and translation
        ):
            results[item_id] = {
                "id": item_id,
                "translation_fr": translation,
            }

    missing = [
        item_id
        for item_id in expected_ids
        if item_id not in results
    ]

    if missing:
        raise ValueError(
            "Traductions absentes dans "
            "la réponse Qwen : "
            + str(missing)
            + "\\nRéponse brute :\\n"
            + text[:2000]
        )

    return [
        results[item_id]
        for item_id in expected_ids
    ]


def build_prompt(
    segments,
    glossary,
    series_context,
    characters,
    previous_context,
):
    payload = []

    for seg in segments:
        payload.append({
            "id": int(seg["id"]),
            "character_id": seg.get(
                "character_id"
            ),
            "japanese": seg["text_ja"],
            "needs_asr_review": seg.get(
                "needs_asr_review",
                False,
            ),
            "issues": seg.get(
                "issues",
                [],
            ),
        })

    return f"""
Tu es un traducteur professionnel spécialisé
dans la localisation d'anime japonais vers
le français.

Traduis les dialogues en français naturel,
fluide et crédible à l'oral.

Règles :
- respecte le sens japonais ;
- conserve les noms propres ;
- conserve la personnalité du personnage ;
- utilise le glossaire si nécessaire ;
- tiens compte du contexte des lignes voisines ;
- n'invente aucune information ;
- ne fusionne aucune ligne ;
- ne supprime aucune ligne ;
- retourne exactement UNE ligne par dialogue ;
- utilise STRICTEMENT le format :
  ID|||traduction française
- aucun JSON ;
- aucun markdown ;
- aucun commentaire ;
- aucun texte avant ou après ;
- conserve exactement tous les identifiants.

Contexte de la série :
{json.dumps(series_context, ensure_ascii=False)}

Glossaire :
{json.dumps(glossary, ensure_ascii=False)}

Personnages :
{json.dumps(characters, ensure_ascii=False)}

Contexte précédent :
{json.dumps(previous_context, ensure_ascii=False)}

Dialogues :
{json.dumps(payload, ensure_ascii=False)}

Format attendu :

[
  {{
    "id": 1,
    "translation_fr": "Traduction française"
  }}
]
"""


def translate_dialogue(
    video_id,
    model=None,
    tokenizer=None,
):
    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    translation_dir = (
        run_dir / "05_translation"
    )

    dialogue_file = (
        translation_dir
        / "dialogue_source.json"
    )

    output_file = (
        translation_dir
        / "translation.json"
    )

    # -------------------------
    # Cache
    # -------------------------

    if (
        stage_done(video_id, "translation")
        and output_file.exists()
    ):
        print("♻️ Traduction déjà disponible")
        print("📁", output_file)

        return load_json(output_file)

    if not dialogue_file.exists():
        raise FileNotFoundError(
            dialogue_file
        )

    dialogue = load_json(dialogue_file)

    anime_dir = get_anime_dir(
        run["anime_id"]
    )

    glossary = load_json(
        anime_dir / "glossary.json",
        {"version": 1, "terms": {}},
    )

    series_context = load_json(
        anime_dir / "context.json",
        {
            "version": 1,
            "series_summary": "",
            "important_events": [],
        },
    )

    characters_db = load_json(
        anime_dir / "characters.json",
        {
            "version": 1,
            "characters": {},
        },
    )

    characters = {}

    for character_id, info in (
        characters_db
        .get("characters", {})
        .items()
    ):
        characters[character_id] = {
            "name": info.get("name"),
            "status": info.get("status"),
        }

    # -------------------------
    # Modèle
    # -------------------------

    if tokenizer is None:
        print(
            "📥 Tokenizer :",
            MODEL_NAME,
        )

        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

    if model is None:
        print(
            "🧠 Modèle traduction :",
            MODEL_NAME,
        )

        print("⚙️ Quantification : 4-bit NF4")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
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
        else "CPU",
    )

    segments = dialogue["segments"]

    translated = []
    previous_context = []

    # -------------------------
    # Traduction par blocs
    # -------------------------

    for start in range(
        0,
        len(segments),
        CHUNK_SIZE,
    ):
        chunk = segments[
            start:start + CHUNK_SIZE
        ]

        print()
        print(
            f"🌍 Traduction "
            f"{start + 1} → "
            f"{start + len(chunk)} "
            f"/ {len(segments)}"
        )

        prompt = build_prompt(
            chunk,
            glossary,
            series_context,
            characters,
            previous_context[-4:],
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu traduis professionnellement "
                    "des dialogues japonais "
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
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        )

        device = next(
            model.parameters()
        ).device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=True,
                temperature=0.5,
                top_p=0.8,
                top_k=20,
            )

        prompt_length = (
            inputs["input_ids"].shape[-1]
        )

        generated = outputs[0][
            prompt_length:
        ]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        expected = [
            int(seg["id"])
            for seg in chunk
        ]

        result = parse_translation_lines(
            answer,
            expected,
        )

        by_id = {
            int(item["id"]): item
            for item in result
        }

        missing = [
            seg_id
            for seg_id in expected
            if seg_id not in by_id
        ]

        if missing:
            raise RuntimeError(
                "Traductions manquantes : "
                + str(missing)
            )

        for seg in chunk:
            seg_id = int(seg["id"])

            fr = (
                by_id[seg_id]
                ["translation_fr"]
                .strip()
            )

            item = {
                **seg,
                "translation_fr": fr,
            }

            translated.append(item)

            previous_context.append({
                "character_id":
                    seg.get("character_id"),
                "ja": seg["text_ja"],
                "fr": fr,
            })

            print(
                f"{seg_id:03d}. "
                f'{seg["text_ja"]}'
            )

            print(
                "     →",
                fr,
            )

    # -------------------------
    # Sauvegarde
    # -------------------------

    data = {
        "version": 1,
        "video_id": video_id,
        "anime_id": run["anime_id"],
        "model": MODEL_NAME,
        "segments": translated,
    }

    save_json(
        output_file,
        data,
    )

    dialogue["segments"] = translated

    save_json(
        dialogue_file,
        dialogue,
    )

    mark_stage_completed(
        video_id,
        "translation",
        output_file,
    )

    print()
    print("✅ Traduction terminée")
    print("📁", output_file)

    return data
