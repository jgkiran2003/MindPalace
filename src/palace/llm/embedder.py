"""Embedding client for Ollama's single and batch embedding endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import requests

from palace.config import Settings, get_settings


class OllamaEmbedder:
    """Synchronous adapter around Ollama's embedding APIs."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        host = self.settings.embedding.ollama_host.rstrip("/")
        self._session = session or requests.Session()
        self._url = f"{host}/api/embeddings"
        self._batch_url = f"{host}/api/embed"

    def _validate_vector(self, embedding: Any) -> list[float]:
        if not isinstance(embedding, list):
            raise ValueError("Ollama response did not contain an embedding list")
        vector = [float(value) for value in embedding]
        expected = self.settings.embedding.dimension
        if len(vector) != expected:
            raise ValueError(
                f"Ollama returned an embedding with {len(vector)} dimensions; "
                f"expected {expected}"
            )
        return vector

    def embed(self, text: str) -> list[float]:
        response = self._session.post(
            self._url,
            json={"model": self.settings.embedding.model, "prompt": text},
            timeout=self.settings.llm.request_timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return self._validate_vector(payload.get("embedding"))

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed all texts in one native Ollama ``/api/embed`` request."""
        inputs = list(texts)
        if not inputs:
            return []
        response = self._session.post(
            self._batch_url,
            json={"model": self.settings.embedding.model, "input": inputs},
            timeout=self.settings.llm.request_timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Ollama response did not contain an 'embeddings' list")
        if len(embeddings) != len(inputs):
            raise ValueError(
                "Ollama returned a different number of embeddings than inputs"
            )
        return [self._validate_vector(embedding) for embedding in embeddings]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """Backward-compatible alias for native batch embedding."""
        return self.embed_batch(texts)
