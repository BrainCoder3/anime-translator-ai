
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3-8B"


def load_translation_runtime():

    print(
        "🧠 Chargement moteur linguistique :",
        MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.eval()

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    return model, tokenizer
