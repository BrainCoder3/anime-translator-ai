
import os
import json
import argparse

from src.core.anime_memory import (
    create_anime,
    get_anime,
)

from src.core.run_manager import (
    RUNS_ROOT,
    resolve_video_path,
    calculate_video_hash,
    prepare_run,
    get_run_dir,
    stage_done,
)

from src.audio.extract_audio import extract_audio
from src.asr.transcribe import transcribe
from src.asr.refine_dialogue import refine_dialogue

from src.diarization.diarize import diarize
from src.diarization.align_asr import (
    align_asr_with_speakers,
)

from src.characters.analyze_speakers import (
    analyze_speakers,
)
from src.characters.compare_speakers import (
    compare_speakers,
)
from src.characters.match_characters import (
    match_or_register_characters,
)
from src.characters.extract_voice_references import (
    extract_voice_references,
)

from src.translation.build_dialogue import (
    build_dialogue,
)
from src.translation.translate_dialogue import (
    translate_dialogue,
)
from src.translation.review_translation import (
    review_translation,
)
from src.translation.adapt_timing import (
    adapt_timing,
)
from src.tts.generate_tts import (
    generate_tts,
)
from src.tts.finalize_timing import (
    finalize_tts_timing,
)
from src.mix.build_final_video import (
    build_final_video,
)
from src.translation.runtime import (
    load_translation_runtime,
)


def existing_run_for_video(video_path):
    video_path = resolve_video_path(video_path)

    full_hash = calculate_video_hash(video_path)
    video_id = full_hash[:16]

    run_file = (
        RUNS_ROOT
        / video_id
        / "run.json"
    )

    if not run_file.exists():
        return None

    with open(run_file, encoding="utf-8") as f:
        return json.load(f)


def resolve_anime(
    video_path,
    anime_id=None,
    new_anime=False,
    anime_name=None,
):
    existing = existing_run_for_video(video_path)

    # Vidéo déjà connue
    if existing is not None:
        existing_anime = existing["anime_id"]

        if new_anime:
            raise RuntimeError(
                "Cette vidéo est déjà liée à "
                f"{existing_anime}."
            )

        if (
            anime_id is not None
            and anime_id != existing_anime
        ):
            raise RuntimeError(
                f"Cette vidéo appartient à "
                f"{existing_anime}, pas à {anime_id}."
            )

        print(
            "♻️ Vidéo déjà connue →",
            existing_anime,
        )

        return existing_anime

    # Nouvelle vidéo + nouvel anime
    if new_anime:
        if not anime_name:
            raise ValueError(
                "--anime-name est obligatoire "
                "avec --new-anime"
            )

        anime_id = create_anime(anime_name)

        print(
            "🆕 Nouvel anime créé :",
            anime_id,
        )

        return anime_id

    # Nouvelle vidéo + anime existant
    if anime_id:
        get_anime(anime_id)

        print(
            "📺 Anime existant :",
            anime_id,
        )

        return anime_id

    raise ValueError(
        "Pour une nouvelle vidéo, utilise soit :\n"
        '--new-anime --anime-name "Nom"\n'
        "soit :\n"
        "--anime anime_001"
    )


def process_video(
    video_path,
    anime_id=None,
    new_anime=False,
    anime_name=None,
):

    print()
    print("=" * 60)
    print("🎬 ANIME TRANSLATOR AI")
    print("=" * 60)

    video_path = resolve_video_path(video_path)

    print("📹 Vidéo :", video_path)

    anime_id = resolve_anime(
        video_path=video_path,
        anime_id=anime_id,
        new_anime=new_anime,
        anime_name=anime_name,
    )

    run = prepare_run(
        video_path,
        anime_id,
    )

    video_id = run["video_id"]
    run_dir = get_run_dir(video_id)

    print("🆔 Video ID :", video_id)
    print("📚 Anime    :", anime_id)
    print()

    # ==================================================
    # 1 — AUDIO
    # ==================================================

    print("─── 1/12 AUDIO ───")

    extract_audio(video_id)

    # ==================================================
    # 2 — ASR
    # ==================================================

    print()
    print("─── 2/12 ASR JAPONAIS ───")

    transcribe(video_id)

    # ==================================================
    # 3 — DIARISATION
    # ==================================================

    print()
    print("─── 3/12 DIARISATION ───")

    if stage_done(video_id, "diarization"):
        print("♻️ Diarisation déjà effectuée")

    else:
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN absent de l'environnement."
            )

        diarize(
            video_id,
            hf_token=hf_token,
        )

    # ==================================================
    # 4 — ALIGNEMENT ASR / SPEAKERS
    # ==================================================

    print()
    print("─── 4/12 ALIGNEMENT ───")

    alignment_file = (
        run_dir
        / "03_diarization"
        / "asr_speaker_alignment.json"
    )

    if alignment_file.exists():
        print("♻️ Alignement déjà disponible")
    else:
        align_asr_with_speakers(video_id)

    # ==================================================
    # 5 — PERSONNAGES
    # ==================================================

    print()
    print("─── 5/12 PERSONNAGES ───")

    characters_summary = (
        run_dir
        / "04_characters"
        / "voice_references.json"
    )

    if (
        stage_done(video_id, "characters")
        and characters_summary.exists()
    ):
        print("♻️ Personnages déjà analysés")

    else:
        analyze_speakers(video_id)
        compare_speakers(video_id)

        match_or_register_characters(
            video_id
        )

        extract_voice_references(
            video_id
        )

    # ==================================================
    # 6 — DIALOGUE SOURCE
    # ==================================================

    print()
    print("─── 6/12 DIALOGUE SOURCE ───")

    dialogue_file = (
        run_dir
        / "05_translation"
        / "dialogue_source.json"
    )

    if dialogue_file.exists():
        print(
            "♻️ Dialogue source déjà construit"
        )
    else:
        build_dialogue(video_id)

    # ==================================================
    # 7 — QA / RAFFINEMENT ASR
    # ==================================================

    print()
    print("─── 7/12 QA ASR AUTOMATIQUE ───")

    refine_dialogue(video_id)

    # ==================================================
    # 8 — TRADUCTION
    # ==================================================

    print()
    print("─── 8/12 TRADUCTION FR ───")

    translate_dialogue(video_id)

    # ==================================================
    # 9 — QA TRADUCTION
    # ==================================================

    print()
    print("─── 9/12 QA TRADUCTION ───")

    review_translation(video_id)

    # ==================================================
    # 10 — ADAPTATION TEMPORELLE
    # ==================================================

    print()
    print("─── 10/12 ADAPTATION TEMPORELLE ───")

    adapt_timing(video_id)

    # ==================================================
    # 11 — TTS
    # ==================================================

    print()
    print("─── 11/12 TTS FRANÇAIS ───")

    generate_tts(video_id)

    # Timing TTS final :
    # légère accélération uniquement lorsque nécessaire.
    finalize_tts_timing(video_id)

    # ==================================================
    # 12 — MIXAGE + VIDÉO FINALE
    # ==================================================

    print()
    print("─── 12/12 MIXAGE + VIDÉO FINALE ───")

    build_final_video(
        video_id,
        video_path,
    )

    # ==================================================
    # ÉTAT
    # ==================================================

    print()
    print("=" * 60)
    print("🎉 TRAITEMENT TERMINÉ")
    print("=" * 60)

    print()
    print("Video ID :", video_id)
    print("Anime    :", anime_id)

    print()
    print("audio        ✅")
    print("ASR          ✅")
    print("diarisation  ✅")
    print("personnages  ✅")
    print("dialogue     ✅")
    print("traduction   ✅")
    print("adaptation   ✅")
    print("TTS          ✅")
    print("vidéo finale ✅")

    print()
    print(
        "📁 Run :",
        run_dir,
    )

    return {
        "video_id": video_id,
        "anime_id": anime_id,
        "run_dir": str(run_dir),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Anime Translator AI - "
            "pipeline universel"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Chemin de la vidéo",
    )

    parser.add_argument(
        "--anime",
        dest="anime_id",
        help="Anime existant, ex: anime_001",
    )

    parser.add_argument(
        "--new-anime",
        action="store_true",
        help="Créer un nouvel anime",
    )

    parser.add_argument(
        "--anime-name",
        help="Nom du nouvel anime",
    )

    args = parser.parse_args()

    if args.new_anime and args.anime_id:
        parser.error(
            "--new-anime et --anime "
            "sont incompatibles."
        )

    process_video(
        video_path=args.video,
        anime_id=args.anime_id,
        new_anime=args.new_anime,
        anime_name=args.anime_name,
    )


if __name__ == "__main__":
    main()
