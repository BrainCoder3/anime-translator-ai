
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.core.run_manager import (
    get_run_dir,
    stage_done,
    mark_stage_completed,
)


MODEL_NAME = "pyannote/speaker-diarization-community-1"


def _annotation_to_list(annotation):
    items = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        items.append({
            "start": round(float(turn.start), 3),
            "end": round(float(turn.end), 3),
            "speaker": str(speaker),
        })

    return items


def diarize(video_id, pipeline=None, hf_token=None):
    run_dir = get_run_dir(video_id)

    audio_path = run_dir / "01_audio" / "speech_16k_mono.wav"
    output_dir = run_dir / "03_diarization"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / "diarization.json"
    embeddings_file = output_dir / "speaker_embeddings.npy"

    # =========================
    # CACHE
    # =========================

    if stage_done(video_id, "diarization") and output_json.exists():
        print("♻️ Diarisation déjà effectuée")
        print("📁", output_json)

        with open(output_json, encoding="utf-8") as f:
            return json.load(f)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio introuvable : {audio_path}"
        )

    # =========================
    # CHARGEMENT PIPELINE
    # =========================

    if pipeline is None:
        from pyannote.audio import Pipeline

        hf_token = hf_token or os.environ.get("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN absent. Passe hf_token=... ou définis HF_TOKEN."
            )

        print("🧠 Chargement Pyannote Community-1...")

        pipeline = Pipeline.from_pretrained(
            MODEL_NAME,
            token=hf_token,
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        pipeline.to(device)

        print("Device :", device)

    # =========================
    # DIARISATION
    # =========================

    print("👥 Diarisation des locuteurs...")

    result = pipeline(str(audio_path))

    regular = _annotation_to_list(
        result.speaker_diarization
    )

    exclusive = _annotation_to_list(
        result.exclusive_speaker_diarization
    )

    speakers = list(
        result.speaker_diarization.labels()
    )

    # =========================
    # EMBEDDINGS LOCUTEURS
    # =========================

    embeddings_saved = False

    embeddings = getattr(
        result,
        "speaker_embeddings",
        None,
    )

    if embeddings is not None:
        embeddings = np.asarray(embeddings)

        np.save(
            embeddings_file,
            embeddings,
        )

        embeddings_saved = True

    # =========================
    # JSON
    # =========================

    data = {
        "version": 1,
        "model": MODEL_NAME,
        "video_id": video_id,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "speaker_embedding_order": (
            speakers if embeddings_saved else []
        ),
        "speaker_embeddings_file": (
            str(embeddings_file)
            if embeddings_saved
            else None
        ),
        "regular_diarization": regular,
        "exclusive_diarization": exclusive,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    mark_stage_completed(
        video_id,
        "diarization",
        output_json,
    )

    # =========================
    # AFFICHAGE
    # =========================

    print()
    print("✅ Diarisation terminée")
    print("Locuteurs :", len(speakers))
    print("IDs       :", ", ".join(speakers))

    if embeddings_saved:
        print(
            "Embeddings :",
            embeddings.shape,
        )

    print()

    for i, item in enumerate(exclusive, 1):
        print(
            f'{i:03d}. '
            f'[{item["start"]:.2f} -> {item["end"]:.2f}] '
            f'{item["speaker"]}'
        )

    print()
    print("📁", output_json)

    return data
