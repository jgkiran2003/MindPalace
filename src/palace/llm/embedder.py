"""Embedding client for Ollama's legacy ``/api/embeddings`` endpoint."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import requests

from palace.config import Settings, get_settings


class OllamaEmbedder:
    """Small synchronous adapter around the Ollama embeddings API."""

    def __init__(
        self,
        settings: Settings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._session = session or requests.Session()
        self._url = f"{self.settings.embedding.ollama_host.rstrip('/')}/api/embeddings"

    def embed(self, text: str) -> list[float]:
        response = self._session.post(
            self._url,
            json={"model": self.settings.embedding.model, "prompt": text},
            timeout=self.settings.llm.request_timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Ollama response did not contain an 'embedding' list")
        vector = [float(value) for value in embedding]
        expected = self.settings.embedding.dimension
        if len(vector) != expected:
            raise ValueError(
                f"Ollama returned an embedding with {len(vector)} dimensions; "
                f"expected {expected}"
            )
        return vector

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        # /api/embeddings accepts one prompt per request. Keeping batching here
        # gives the indexer a stable interface if Ollama's batch API is adopted.
        return [self.embed(text) for text in texts]

