
import argparse
import json
import random
import math
from pathlib import Path

import numpy as np
import torch
import torchaudio as ta

from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
)


def duration_of(path):
    wav, sr = ta.load(str(path))

    return (
        wav.shape[-1]
        / float(sr)
    )


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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job",
        required=True,
    )

    args = parser.parse_args()

    job_file = Path(args.job)

    with open(
        job_file,
        encoding="utf-8",
    ) as f:
        job = json.load(f)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponible pour Chatterbox."
        )

    # Reproductibilité raisonnable
    seed = int(
        job.get("seed", 1234)
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(
        "🧠 Chargement Chatterbox "
        "Multilingual V3..."
    )

    model = (
        ChatterboxMultilingualTTS
        .from_pretrained(
            device="cuda",
            t3_model="v3",
        )
    )

    print(
        "✅ Modèle chargé"
    )

    print(
        "Sample rate :",
        model.sr,
    )

    # =======================================================
    # CONTRÔLE DE DURÉE T3
    # =======================================================

    # Chatterbox utilise normalement un max_new_tokens
    # très permissif. Pour de courtes répliques cela peut
    # provoquer des générations de plusieurs secondes
    # totalement disproportionnées.
    #
    # On intercepte l'appel au T3 pour fixer une limite
    # spécifique à chaque segment.

    original_t3_inference = (
        model.t3.inference
    )

    token_limit = {
        "value": 1000
    }

    def capped_t3_inference(
        *args,
        **kwargs,
    ):

        kwargs[
            "max_new_tokens"
        ] = token_limit["value"]

        return original_t3_inference(
            *args,
            **kwargs,
        )

    model.t3.inference = (
        capped_t3_inference
    )


    def calculate_token_cap(
        text,
        target_seconds,
    ):

        # Estimation de débit français.
        text_seconds = max(
            0.40,
            len(text.strip()) / 14.0,
        )

        # Marge laissée au modèle,
        # mais sans possibilité de runaway.
        allowed_seconds = max(
            target_seconds * 1.45,
            text_seconds * 1.15,
            0.80,
        )

        max_tokens = (
            math.ceil(
                allowed_seconds * 25
            )
            + 3
        )

        return (
            max_tokens,
            allowed_seconds,
        )

    segments = job["segments"]

    progress_file = Path(
        job["progress_file"]
    )

    result_file = Path(
        job["result_file"]
    )

    results = []

    last_reference = None

    for index, item in enumerate(
        segments,
        1,
    ):

        segment_id = int(
            item["id"]
        )

        text = item["text"].strip()

        output_path = Path(
            item["output_path"]
        )

        reference_path = Path(
            item["reference_path"]
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            f"🎙️ {index}/{len(segments)} "
            f"— segment {segment_id:03d}"
        )

        print(
            "   Personnage :",
            item.get("character_id"),
        )

        print(
            "   Texte :",
            text,
        )

        # ===========================
        # TIMING
        # ===========================

        target_seconds = float(
            item["target_seconds"]
        )

        available_seconds = float(
            item.get(
                "available_seconds",
                target_seconds,
            )
        )

        existing_seconds = None
        must_regenerate = False

        # ===========================
        # ANALYSE DU CACHE
        # ===========================

        if output_path.exists():

            existing_seconds = (
                duration_of(
                    output_path
                )
            )

            existing_ratio = (
                existing_seconds
                / available_seconds
                if available_seconds > 0
                else 1.0
            )

            # Durée naturelle estimée indépendamment
            # du silence disponible après la réplique.
            (
                _natural_tokens,
                natural_limit,
            ) = calculate_token_cap(
                text,
                target_seconds,
            )

            natural_ratio = (
                existing_seconds
                / natural_limit
                if natural_limit > 0
                else 1.0
            )

            # Deux raisons de régénérer :
            #
            # 1. la voix ne rentre pas dans la timeline ;
            # 2. elle rentre uniquement parce qu'il existe
            #    beaucoup de silence après, mais elle est
            #    anormalement lente pour le texte.
            timeline_bad = (
                existing_ratio > 1.30
            )

            unnatural_bad = (
                natural_ratio > 1.35
            )

            if not timeline_bad and not unnatural_bad:

                generated_seconds = (
                    existing_seconds
                )

                print(
                    "   ♻️ Audio conservé"
                )

                print(
                    "   Fenêtre :",
                    f"{available_seconds:.2f}s"
                )

                print(
                    "   Ratio réel :",
                    f"{existing_ratio:.2f}x"
                )

            else:

                must_regenerate = True

                if unnatural_bad and not timeline_bad:
                    print(
                        "   🔄 Voix anormalement lente"
                    )
                    print(
                        "   Audio :",
                        f"{existing_seconds:.2f}s"
                    )
                    print(
                        "   Durée naturelle ≈",
                        f"{natural_limit:.2f}s"
                    )
                    print(
                        "   Ratio naturel :",
                        f"{natural_ratio:.2f}x"
                    )
                else:
                    print(
                        "   🔄 Audio trop long"
                    )

                print(
                    "   Ancien :",
                    f"{existing_seconds:.2f}s"
                )

                print(
                    "   Fenêtre :",
                    f"{available_seconds:.2f}s"
                )

                print(
                    "   Ratio :",
                    f"{existing_ratio:.2f}x"
                )

        else:
            must_regenerate = True

        # ===========================
        # GÉNÉRATION / RÉGÉNÉRATION
        # ===========================

        if must_regenerate:

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Référence absente : "
                    f"{reference_path}"
                )

            (
                max_tokens,
                allowed_seconds,
            ) = calculate_token_cap(
                text,
                target_seconds,
            )

            token_limit["value"] = (
                max_tokens
            )

            print(
                "   🎯 Limite T3 :",
                max_tokens,
                "tokens"
            )

            print(
                "   Durée T3 ≈",
                f"{allowed_seconds:.2f}s"
            )

            kwargs = {
                "language_id": "fr",

                # Référence japonaise
                # vers parole française.
                "cfg_weight": 0.0,

                "exaggeration": 0.5,

                # Plus stable pour les
                # régénérations timing.
                "temperature": 0.55,

                "repetition_penalty": 1.35,

                "min_p": 0.03,

                "top_p": 0.90,
            }

            reference_str = str(
                reference_path
            )

            if (
                reference_str
                != last_reference
            ):

                kwargs[
                    "audio_prompt_path"
                ] = reference_str

                last_reference = (
                    reference_str
                )

                print(
                    "   🎭 Référence :",
                    reference_path.name
                )

            # Seed propre au segment :
            # reproductible mais différente.
            segment_seed = (
                seed
                + segment_id * 97
            )

            random.seed(
                segment_seed
            )

            np.random.seed(
                segment_seed
            )

            torch.manual_seed(
                segment_seed
            )

            torch.cuda.manual_seed_all(
                segment_seed
            )

            wav = model.generate(
                text[:300],
                **kwargs,
            )

            ta.save(
                str(output_path),
                wav.cpu(),
                model.sr,
            )

            generated_seconds = (
                wav.shape[-1]
                / float(model.sr)
            )

            print(
                "   ✅ Régénéré :",
                f"{generated_seconds:.2f}s"
            )

            # ==================================================
            # SECOND ESSAI : HARD TIMING
            # ==================================================
            #
            # Si même la génération plafonnée dépasse encore
            # fortement la vraie fenêtre disponible, on impose
            # une limite directement basée sur cette fenêtre.
            #
            # Cela concerne surtout les répliques très courtes :
            # "Hey", "Fais-le bien", etc.

            first_ratio = (
                generated_seconds
                / available_seconds
                if available_seconds > 0
                else 1.0
            )

            if first_ratio > 1.30:

                print(
                    "   ⚠️ Toujours trop long :",
                    f"{first_ratio:.2f}x"
                )

                # On autorise environ 12 % de marge.
                hard_allowed_seconds = max(
                    available_seconds * 1.12,
                    0.60,
                )

                hard_tokens = (
                    math.ceil(
                        hard_allowed_seconds * 25
                    )
                    + 2
                )

                token_limit["value"] = (
                    hard_tokens
                )

                print(
                    "   🔥 Retry timing strict :",
                    hard_tokens,
                    "tokens"
                )

                best_wav = wav
                best_duration = generated_seconds

                # Deux seeds supplémentaires.
                for retry_index in range(2):

                    retry_seed = (
                        segment_seed
                        + 10000
                        + retry_index * 1000
                    )

                    random.seed(
                        retry_seed
                    )

                    np.random.seed(
                        retry_seed
                    )

                    torch.manual_seed(
                        retry_seed
                    )

                    torch.cuda.manual_seed_all(
                        retry_seed
                    )

                    retry_wav = model.generate(
                        text[:300],
                        **kwargs,
                    )

                    retry_duration = (
                        retry_wav.shape[-1]
                        / float(model.sr)
                    )

                    retry_ratio = (
                        retry_duration
                        / available_seconds
                        if available_seconds > 0
                        else 1.0
                    )

                    print(
                        f"      essai {retry_index + 1} :",
                        f"{retry_duration:.2f}s",
                        f"({retry_ratio:.2f}x)"
                    )

                    # Priorité :
                    # 1. rentrer sous 1.30x
                    # 2. sinon prendre la plus courte.
                    best_ratio = (
                        best_duration
                        / available_seconds
                        if available_seconds > 0
                        else 1.0
                    )

                    if (
                        retry_ratio <= 1.30
                        and best_ratio > 1.30
                    ):
                        best_wav = retry_wav
                        best_duration = retry_duration

                    elif (
                        retry_ratio <= 1.30
                        and best_ratio <= 1.30
                        and retry_duration > best_duration
                    ):
                        # Parmi les audios qui rentrent,
                        # on préfère celui qui utilise le plus
                        # naturellement la fenêtre.
                        best_wav = retry_wav
                        best_duration = retry_duration

                    elif (
                        retry_ratio > 1.30
                        and best_ratio > 1.30
                        and retry_duration < best_duration
                    ):
                        best_wav = retry_wav
                        best_duration = retry_duration

                wav = best_wav
                generated_seconds = best_duration

                ta.save(
                    str(output_path),
                    wav.cpu(),
                    model.sr,
                )

                final_retry_ratio = (
                    generated_seconds
                    / available_seconds
                    if available_seconds > 0
                    else 1.0
                )

                print(
                    "   🎯 Meilleur résultat :",
                    f"{generated_seconds:.2f}s",
                    f"({final_retry_ratio:.2f}x)"
                )


        ratio_target = (
            generated_seconds
            / target_seconds
            if target_seconds > 0
            else 1.0
        )

        ratio_available = (
            generated_seconds
            / available_seconds
            if available_seconds > 0
            else 1.0
        )

        result = {
            **item,

            "generated_seconds": round(
                generated_seconds,
                3,
            ),

            "duration_ratio_target":
                round(
                    ratio_target,
                    3,
                ),

            "duration_ratio_available":
                round(
                    ratio_available,
                    3,
                ),

            "available_seconds":
                round(
                    available_seconds,
                    3,
                ),

            "needs_speedup": (
                ratio_available > 1.08
            ),

            "needs_regeneration": (
                ratio_available > 1.30
            ),
        }

        results.append(
            result
        )

        save_json(
            progress_file,
            {
                "version": 1,
                "completed":
                    len(results),
                "total":
                    len(segments),
                "segments":
                    results,
            },
        )

        print(
            "   Cible ASR :",
            f"{target_seconds:.2f}s",
        )

        print(
            "   Fenêtre dispo :",
            f"{available_seconds:.2f}s",
        )

        print(
            "   Ratio fenêtre :",
            f"{ratio_available:.2f}x",
        )

    save_json(
        result_file,
        {
            "version": 1,
            "engine":
                "chatterbox-multilingual-v3",
            "segments":
                results,
        },
    )

    print()
    print(
        "✅ Génération Chatterbox terminée"
    )

    print(
        "📁",
        result_file,
    )


if __name__ == "__main__":
    main()
