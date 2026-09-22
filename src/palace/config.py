"""
MindPalace — Settings loader
Single source of truth for every config-derived constant used across the
codebase. Nothing downstream should hardcode a value that appears here —
import `get_settings()` instead. This is what makes settings.toml the one
place a value like `raw_status_penalty` or `dimension` is ever defined.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

DEFAULT_SETTINGS_PATH = "config/settings.toml"


class PathsConfig(BaseModel):
    vault_dir: str
    lance_dir: str
    claims_db_path: str
    schema_sql_path: str
    temp_audio_dir: str


class EmbeddingConfig(BaseModel):
    model: str
    dimension: int
    ollama_host: str
    batch_size: int


class RerankerConfig(BaseModel):
    model: str
    max_length: int
    device: str


class RetrievalConfig(BaseModel):
    top_k: int
    retrieval_top_k: int
    claims_search_k: int
    rrf_k: int
    raw_status_penalty: float


class LLMConfig(BaseModel):
    ollama_host: str
    fast_tier_model: str
    reasoning_tier_model: str
    fast_tier_temperature: float
    reasoning_tier_temperature: float
    request_timeout_seconds: int


class ClaimsConfig(BaseModel):
    high_severity_blocks_auto_resolve: bool
    compaction_age_days: int
    compaction_min_count: int


class IndexingConfig(BaseModel):
    optimize_after_writes: int


class Settings(BaseModel):
    paths: PathsConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    claims: ClaimsConfig
    indexing: IndexingConfig


@lru_cache(maxsize=1)
def get_settings(path: str = DEFAULT_SETTINGS_PATH) -> Settings:
    """
    Cached on purpose — settings.toml is read once per process. Tests that
    need a different config should call `get_settings.cache_clear()` first.
    """
    with open(Path(path), "rb") as f:
        data = tomllib.load(f)
    return Settings(**data)
