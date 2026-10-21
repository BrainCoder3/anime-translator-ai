"""Translation and JSON-mode LLM calls through a local Ollama server."""
from __future__ import annotations

import re
import time

import requests

LANG_NAMES = {"ja": "japonais", "fr": "français", "en": "anglais", "es": "espagnol"}


class OllamaClient:
    def __init__(self, url: str = "http://127.0.0.1:11434", model: str = "qwen3:4b-instruct-2507-q4_K_M",
                 timeout_s: int = 180, retries: int = 1, think: bool | None = None) -> None:
        self.url, self.model, self.timeout_s, self.retries, self.think = url.rstrip("/"), model, timeout_s, retries, think

    def chat(self, prompt: str, *, json_mode: bool = False, temperature: float = 0.2,
             num_predict: int = 400, num_ctx: int = 4096) -> str:
        body: dict = {"model": self.model, "stream": False,
                      "messages": [{"role": "user", "content": prompt}],
                      "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx}}
        if json_mode:
            body["format"] = "json"
        if self.think is not None:           # only for models that have a thinking mode
            body["think"] = self.think
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(f"{self.url}/api/chat", json=body, timeout=(10, self.timeout_s))
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as exc:
                last = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed after {self.retries + 1} attempt(s): {last}")


def clean_translation(raw: str) -> str:
    """Drop reasoning tags, quotes and line breaks from a model answer."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = " ".join(lines)
    return text.strip().strip('"«»“”').strip()


class OllamaTranslator:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def translate(self, text: str, source_lang: str, target_lang: str, *, previous: str = "",
                  next_: str = "") -> str:
        src, tgt = LANG_NAMES.get(source_lang, source_lang), LANG_NAMES.get(target_lang, target_lang)
        prompt = (f"Tu traduis des répliques de dialogue ({src} vers {tgt}) pour un doublage.\n"
                  f"Phrase précédente (contexte) : {previous or '(aucune)'}\n"
                  f"Phrase suivante (contexte) : {next_ or '(aucune)'}\n\n"
                  f"Traduis uniquement cette phrase en {tgt} oral et naturel, sans commentaire ni guillemets :\n{text}")
        out = clean_translation(self.client.chat(prompt))
        if not out:
            raise RuntimeError("empty translation")
        return out

    def ask(self, prompt: str) -> str:
        return self.client.chat(prompt, json_mode=True, num_predict=500)
