from __future__ import annotations

from typing import List

import numpy as np
import requests


class EmbeddingBackend:
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEmbeddingBackend(EmbeddingBackend):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=float)
        return np.asarray(self._model.encode(texts, show_progress_bar=False), dtype=float)


class RemoteEmbeddingBackend(EmbeddingBackend):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "",
        truncate_prompt_tokens: int = 256,
        timeout: float = 120,
    ) -> None:
        cleaned = base_url.rstrip("/")
        if cleaned.endswith("/embeddings"):
            self.endpoint = cleaned
        else:
            self.endpoint = f"{cleaned}/embeddings"
        self.model_name = model_name
        self.api_key = api_key
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=float)

        payload = {
            "model": self.model_name,
            "input": texts,
            "truncate_prompt_tokens": self.truncate_prompt_tokens,
        }
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("data", [])
        if not isinstance(items, list):
            raise RuntimeError(f"Embedding payload missing data list: {body}")

        ordered = sorted(items, key=lambda item: item.get("index", 0))
        vectors = []
        for item in ordered:
            vector = item.get("embedding")
            if not vector:
                raise RuntimeError(f"Embedding item missing vector: {item}")
            vectors.append(np.asarray(vector, dtype=float))
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding length mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        return np.asarray(vectors, dtype=float)


def build_embedding_backend(
    mode: str,
    model_name: str,
    base_url: str = "",
    api_key: str = "",
    truncate_prompt_tokens: int = 256,
    timeout: float = 120,
) -> EmbeddingBackend:
    normalized = (mode or "local").strip().lower()
    if normalized == "remote":
        return RemoteEmbeddingBackend(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            truncate_prompt_tokens=truncate_prompt_tokens,
            timeout=timeout,
        )
    if normalized == "local":
        return LocalEmbeddingBackend(model_name)
    raise ValueError(f"Unsupported embedding mode: {mode}")
