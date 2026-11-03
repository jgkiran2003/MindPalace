"""Shared pytest guards."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _forbid_default_vault_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A default ``VaultIndexer`` writes to the real data/lance and calls Ollama.

    Tests must inject an indexer (or patch the name themselves, which overrides
    this guard) so a green run can never touch a developer's real index.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "test constructed a default VaultIndexer; inject a fake or tmp-backed one"
        )

    monkeypatch.setattr("palace.compilation.compiler.VaultIndexer", refuse)
    monkeypatch.setattr("palace.cli.VaultIndexer", refuse)
