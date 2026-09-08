import json
from pathlib import Path

from faster_whisper import WhisperModel

from src.core.run_manager import (
    get_run,
    get_run_dir,
    stage_done,
    mark_stage_completed,
)


MODEL_NAME = "large-v3"
LANGUAGE = "ja"


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


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def transcribe(video_id):
    run = get_run(video_id)
    run_dir = get_run_dir(video_id)

    audio_path = (
        run_dir
        / "01_audio"
        / "speech_16k_mono.wav"
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio introuvable : {audio_path}"
        )

    asr_dir = run_dir / "02_asr"
    asr_dir.mkdir(parents=True, exist_ok=True)

    output_path = asr_dir / "transcription.json"

    # ---------------------------------
    # CACHE
    # ---------------------------------

    if stage_done(video_id, "asr"):
        if output_path.exists():
            print("♻️ Transcription déjà disponible")
            print("📁", output_path)
            return load_json(output_path)

        print(
            "⚠️ Cache ASR marqué terminé, "
            "mais transcription absente."
        )
        print("↪️ Reconstruction...")

    # ---------------------------------
    # MODELE
    # ---------------------------------

    print("🧠 Chargement Whisper :", MODEL_NAME)
    print("GPU : CUDA | float16")

    model = WhisperModel(
        MODEL_NAME,
        device="cuda",
        compute_type="float16"
    )

    # ---------------------------------
    # TRANSCRIPTION
    # ---------------------------------

    print("🎙️ Transcription japonaise...")

    segments_generator, info = model.transcribe(
        str(audio_path),

        language=LANGUAGE,

        beam_size=5,

        word_timestamps=True,

        vad_filter=True,

        vad_parameters={
            "min_silence_duration_ms": 350
        },

        condition_on_previous_text=True,
    )

    segments = []

    for index, segment in enumerate(
        segments_generator,
        start=1
    ):
        words = []

        if segment.words:
            for word in segment.words:
                words.append(
                    {
                        "start": round(
                            float(word.start),
                            3
                        ),
                        "end": round(
                            float(word.end),
                            3
                        ),
                        "word": word.word,
                        "probability": round(
                            float(word.probability),
                            4
                        ),
                    }
                )

        item = {
            "id": index,
            "start": round(
                float(segment.start),
                3
            ),
            "end": round(
                float(segment.end),
                3
            ),
            "text": segment.text.strip(),
            "avg_logprob": round(
                float(segment.avg_logprob),
                4
            ),
            "no_speech_prob": round(
                float(segment.no_speech_prob),
                4
            ),
            "words": words,
        }

        segments.append(item)

        print(
            f"{index:03}. "
            f"[{item['start']:.2f} -> "
            f"{item['end']:.2f}] "
            f"{item['text']}"
        )

    # ---------------------------------
    # RESULTAT
    # ---------------------------------

    result = {
        "schema_version": 1,

        "video_id": video_id,

        "model": MODEL_NAME,

        "language": {
            "requested": LANGUAGE,
            "detected": info.language,
            "probability": round(
                float(info.language_probability),
                4
            ),
        },

        "segment_count": len(segments),

        "segments": segments,
    }

    save_json(
        output_path,
        result
    )

    mark_stage_completed(
        video_id,
        "asr",
        output_path
    )

    print()
    print("✅ Transcription terminée")
    print("Segments :", len(segments))
    print("📁", output_path)

    return result
