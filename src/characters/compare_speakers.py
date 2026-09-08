
import json
import numpy as np

from src.core.run_manager import get_run_dir


def compare_speakers(video_id):
    run_dir = get_run_dir(video_id)

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

    output_file = (
        run_dir
        / "04_characters"
        / "speaker_similarity.json"
    )

    with open(diar_file, encoding="utf-8") as f:
        diar = json.load(f)

    embeddings = np.load(embeddings_file)

    speakers = diar.get(
        "speaker_embedding_order",
        diar["speakers"]
    )

    if len(speakers) != len(embeddings):
        raise RuntimeError(
            f"{len(speakers)} speakers mais "
            f"{len(embeddings)} embeddings"
        )

    # Normalisation L2
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / np.clip(
        norms,
        1e-12,
        None
    )

    # Similarité cosinus
    matrix = normalized @ normalized.T

    pairs = []

    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):

            score = float(matrix[i, j])

            pairs.append({
                "speaker_a": speakers[i],
                "speaker_b": speakers[j],
                "cosine_similarity": round(
                    score,
                    4
                ),
            })

    pairs.sort(
        key=lambda x: x["cosine_similarity"],
        reverse=True
    )

    data = {
        "version": 1,
        "video_id": video_id,
        "speakers": speakers,
        "matrix": {
            speakers[i]: {
                speakers[j]: round(
                    float(matrix[i, j]),
                    4
                )
                for j in range(len(speakers))
            }
            for i in range(len(speakers))
        },
        "pairs": pairs,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("✅ Comparaison des voix terminée")
    print()

    for pair in pairs:
        print(
            f'{pair["speaker_a"]} ↔ '
            f'{pair["speaker_b"]} : '
            f'{pair["cosine_similarity"]:.4f}'
        )

    print()
    print("📁", output_file)

    return data
