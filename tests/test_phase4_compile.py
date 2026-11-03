"""Compile lifecycle: write -> index -> archive, with recoverable failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests
import yaml
from click.testing import CliRunner

import palace.cli as cli
from palace.compilation.compiler import CompilationError, compile_raw_notes
from palace.config import get_settings
from palace.ingestion.chunker import MarkdownChunker, is_archived_note
from palace.ingestion.indexer import VaultIndexer
from palace.stores.file_store import LanceFileStore

COMPILED_TEXT = "# Consolidated Notes\n\nZebracorn deployment uses blue green releases."


class _Embedder:
    """Deterministic 768-d vectors; optionally fails."""

    def __init__(self) -> None:
        self.fail = False

    def embed_many(self, texts: Any) -> list[list[float]]:
        if self.fail:
            raise ConnectionError("embedding service down")
        return [[1.0] + [0.0] * 767 for _ in texts]


class _LLM:
    def __init__(self, reply: str = COMPILED_TEXT, error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls = 0

    def complete(self, prompt: str, *, tier: str, system: str | None = None) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.reply


class _FlakyIndexer(VaultIndexer):
    """Real indexer that raises for chosen paths a limited number of times."""

    def __init__(self, *args: Any, fail_for: dict[str, int] | None = None, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.fail_for = dict(fail_for or {})

    def index_file(self, path: str | Path) -> int:
        key = Path(path).name
        if self.fail_for.get(key, 0) > 0:
            self.fail_for[key] -= 1
            raise OSError(f"disk error for {key}")
        return super().index_file(path)


@pytest.fixture
def embedder() -> _Embedder:
    return _Embedder()


@pytest.fixture
def store(tmp_path: Path) -> LanceFileStore:
    return LanceFileStore(tmp_path / "lance")


@pytest.fixture
def indexer(store: LanceFileStore, embedder: _Embedder) -> VaultIndexer:
    return VaultIndexer(
        settings=get_settings(),
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
    )


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    path = tmp_path / "vault"
    path.mkdir()
    return path


def _make_sources(vault: Path, count: int = 3) -> list[Path]:
    paths = []
    for index in range(1, count + 1):
        path = vault / f"raw-{index}.md"
        path.write_text(
            "---\n"
            f"title: Source {index}\n"
            "status: raw\n"
            "source_type: document\n"
            "---\n\n"
            f"# Source {index}\n\nObservation number {index} about quokkas.\n\n"
            "Trailing text stays exactly as written.\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _rows(store: LanceFileStore, path: Path) -> list[dict[str, Any]]:
    escaped = str(path).replace("'", "''")
    return (
        store.table.search()
        .where(f"doc_path = '{escaped}'")
        .limit(100)
        .to_arrow()
        .to_pylist()
    )


def _frontmatter(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def _snapshot(paths: list[Path]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def _compile(vault: Path, indexer: Any, llm: Any | None = None) -> Path | None:
    return compile_raw_notes(vault, client=llm or _LLM(), indexer=indexer)  # type: ignore[arg-type]


# --------------------------------------------------------------- happy path


def test_compile_indexes_note_archives_sources_and_links_provenance(
    vault: Path, indexer: VaultIndexer, store: LanceFileStore
) -> None:
    sources = _make_sources(vault)
    for source in sources:  # the raw notes were indexed before compiling
        indexer.index_file(source)
        assert _rows(store, source)[0]["status"] == "raw"

    output = _compile(vault, indexer)

    assert output is not None and output.exists()
    meta = _frontmatter(output)
    assert meta["status"] == "compiled"
    assert meta["generated"] is True and "generated" in meta["tags"]
    assert meta["derived_from"] == ["Source 1", "Source 2", "Source 3"]
    assert meta["derived_from_paths"] == ["raw-1.md", "raw-2.md", "raw-3.md"]
    assert all(source.name in meta["derived_from_paths"] for source in sources)

    compiled_rows = _rows(store, output)
    assert compiled_rows and {row["status"] for row in compiled_rows} == {"compiled"}
    hits = store.fts_search("Zebracorn", limit=5)
    assert [hit.doc_path for hit in hits] == [str(output)]

    for source in sources:
        assert _rows(store, source) == []  # archived out of the index
        src_meta = _frontmatter(source)
        assert src_meta["status"] == "raw" and src_meta["processed"] is True
        assert src_meta["compiled_into"] == output.name
        assert is_archived_note(source)


def test_sources_are_preserved_byte_for_byte_apart_from_two_marker_lines(
    vault: Path, indexer: VaultIndexer
) -> None:
    sources = _make_sources(vault)
    before = _snapshot(sources)

    output = _compile(vault, indexer)

    assert output is not None
    for source in sources:
        expected = before[source].decode().replace(
            "---\n\n",
            f"processed: true\ncompiled_into: {output.name}\n---\n\n",
            1,
        )
        assert source.read_text(encoding="utf-8") == expected


def test_rerun_is_a_no_op_and_full_reindex_skips_archived_notes(
    vault: Path, indexer: VaultIndexer, store: LanceFileStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _make_sources(vault)
    llm = _LLM()
    output = _compile(vault, indexer, llm)
    assert output is not None

    assert _compile(vault, indexer, llm) is None
    assert llm.calls == 1
    assert len(list(vault.glob("compiled-*.md"))) == 1

    settings = get_settings().model_copy(
        update={"paths": get_settings().paths.model_copy(update={"vault_dir": str(vault)})}
    )
    full = VaultIndexer(
        settings=settings,
        store=store,
        embedder=_Embedder(),  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
    )
    stats = full.index()
    assert (stats.documents, stats.archived) == (4, 3)
    assert _rows(store, output)
    assert all(_rows(store, source) == [] for source in sources)


def test_below_threshold_does_nothing(vault: Path, indexer: VaultIndexer) -> None:
    sources = _make_sources(vault, count=2)
    before = _snapshot(sources)
    llm = _LLM()

    assert _compile(vault, indexer, llm) is None
    assert llm.calls == 0
    assert _snapshot(sources) == before


# ------------------------------------------------------------------ failures


def test_llm_failure_changes_nothing(vault: Path, indexer: VaultIndexer) -> None:
    sources = _make_sources(vault)
    before = _snapshot(sources)

    with pytest.raises(requests.ConnectionError):
        _compile(vault, indexer, _LLM(error=requests.ConnectionError("ollama down")))

    assert _snapshot(sources) == before
    assert list(vault.glob("compiled-*.md")) == []


@pytest.mark.parametrize("reply", ["", "   \n", "```markdown\n```", "---\ntitle: x\n---\n"])
def test_blank_model_output_is_rejected_before_anything_is_written(
    vault: Path, indexer: VaultIndexer, reply: str
) -> None:
    sources = _make_sources(vault)
    before = _snapshot(sources)

    with pytest.raises(CompilationError, match="no usable content"):
        _compile(vault, indexer, _LLM(reply=reply))

    assert _snapshot(sources) == before
    assert list(vault.glob("compiled-*.md")) == []


def test_index_failure_rolls_back_and_a_retry_reuses_the_same_filename(
    vault: Path, store: LanceFileStore, embedder: _Embedder
) -> None:
    sources = _make_sources(vault)
    before = _snapshot(sources)
    flaky = _FlakyIndexer(
        settings=get_settings(),
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
        fail_for={},
    )
    embedder.fail = True

    with pytest.raises(CompilationError, match="rolled back"):
        _compile(vault, flaky)

    assert _snapshot(sources) == before  # no source touched
    assert list(vault.glob("compiled-*.md")) == []  # no half-published note
    assert store.table.count_rows() == 0

    embedder.fail = False
    output = _compile(vault, flaky)
    assert output is not None
    assert len(list(vault.glob("compiled-*.md"))) == 1
    assert _rows(store, output)


def test_archive_failure_is_reported_and_repaired_by_a_normal_index(
    vault: Path, store: LanceFileStore, embedder: _Embedder
) -> None:
    sources = _make_sources(vault)
    flaky = _FlakyIndexer(
        settings=get_settings(),
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
        fail_for={"raw-2.md": 1},
    )
    plain = VaultIndexer(
        settings=get_settings(),
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
    )
    for source in sources:
        plain.index_file(source)

    with pytest.raises(CompilationError, match="raw-2.md.*palace index"):
        _compile(vault, flaky)

    output = next(vault.glob("compiled-*.md"))
    assert _rows(store, output)  # compiled note is indexed
    assert _frontmatter(sources[1])["processed"] is True  # vault already says archived
    assert _rows(store, sources[1])  # ...but its stale chunks are still indexed
    assert _rows(store, sources[0]) == [] and _rows(store, sources[2]) == []

    settings = get_settings().model_copy(
        update={"paths": get_settings().paths.model_copy(update={"vault_dir": str(vault)})}
    )
    VaultIndexer(
        settings=settings,
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        chunker=MarkdownChunker(),
    ).index()
    assert _rows(store, sources[1]) == []  # reconciled from the vault


def test_unreadable_source_aborts_before_any_change(
    vault: Path, indexer: VaultIndexer
) -> None:
    sources = _make_sources(vault)
    (vault / "broken.md").write_text("---\nstatus: [unclosed\n---\nbody\n", encoding="utf-8")
    before = _snapshot(sources)
    llm = _LLM()

    with pytest.raises(CompilationError, match="broken.md"):
        _compile(vault, indexer, llm)

    assert llm.calls == 0 and _snapshot(sources) == before


# ---------------------------------------------------------- indexer safety


def test_embedding_failure_keeps_a_documents_existing_chunks(
    vault: Path, indexer: VaultIndexer, store: LanceFileStore, embedder: _Embedder
) -> None:
    note = _make_sources(vault, count=1)[0]
    indexer.index_file(note)
    before = _rows(store, note)
    assert before

    embedder.fail = True
    with pytest.raises(ConnectionError):
        indexer.index_file(note)

    assert [row["chunk_id"] for row in _rows(store, note)] == [
        row["chunk_id"] for row in before
    ]


# --------------------------------------------------------------- CLI wiring


def _patch_cli(monkeypatch: pytest.MonkeyPatch, vault: Path, llm: Any, indexer: Any) -> None:
    settings = get_settings().model_copy(
        update={"paths": get_settings().paths.model_copy(update={"vault_dir": str(vault)})}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "OllamaClient", lambda **kw: llm)
    monkeypatch.setattr(cli, "VaultIndexer", lambda **kw: indexer)


def test_cli_compile_reports_success_and_failure(
    vault: Path, indexer: VaultIndexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sources(vault)
    _patch_cli(monkeypatch, vault, _LLM(), indexer)
    runner = CliRunner()

    ok = runner.invoke(cli.app, ["compile"])
    assert ok.exit_code == 0, ok.output
    assert "Created and indexed compiled note" in ok.output

    again = runner.invoke(cli.app, ["compile"])
    assert again.exit_code == 0
    assert "Not enough unprocessed raw notes" in again.output


def test_cli_compile_failure_is_a_clean_nonzero_exit(
    vault: Path, indexer: VaultIndexer, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_sources(vault)
    _patch_cli(monkeypatch, vault, _LLM(reply=""), indexer)

    result = CliRunner().invoke(cli.app, ["compile"])

    assert result.exit_code == 1
    assert "no usable content" in result.output
