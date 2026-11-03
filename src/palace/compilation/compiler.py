"""Compile processed batches of raw Markdown notes into a cohesive note."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml

from palace.ingestion.indexer import VaultIndexer
from palace.llm.client import OllamaClient

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_MARKER_KEYS = re.compile(r"^(processed|compiled_into)\s*:")


class CompilationError(RuntimeError):
    """Compilation stopped; the message says what state the vault was left in."""


class _Indexer(Protocol):
    def index_file(self, path: str | Path) -> int: ...

    def remove_file(self, path: str | Path) -> int: ...


@dataclass
class _RawNote:
    path: Path
    metadata: dict[str, Any]
    body: str
    source_name: str


def _read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        closing = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return {}, text
    metadata = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"frontmatter must be a mapping: {path}")
    return metadata, "\n".join(lines[closing + 1 :]).strip()


def _render_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{body.strip()}\n"


def _source_name(path: Path, metadata: dict[str, Any], body: str) -> str:
    title = metadata.get("title")
    if title:
        return str(title)
    heading = _HEADING.search(body)
    return heading.group(1).strip() if heading else path.stem


def _compiled_body(response: str) -> str:
    body = response.strip()
    if body.startswith("```markdown") or body.startswith("```md"):
        body = "\n".join(body.splitlines()[1:])
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3].rstrip()
    if body.startswith("---"):
        _, body = _read_frontmatter_text(body)
    return body.strip()


def _read_frontmatter_text(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        closing = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return {}, text
    metadata = yaml.safe_load("\n".join(lines[1:closing])) or {}
    return metadata if isinstance(metadata, dict) else {}, "\n".join(lines[closing + 1 :])


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory so readers never see a torso."""
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        if path.exists():
            shutil.copymode(path, temp_name)  # mkstemp would otherwise force 0600
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temp_name, 0o666 & ~umask)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _mark_processed(text: str, compiled_name: str) -> str:
    """Add ``processed``/``compiled_into`` to a note's frontmatter in place.

    Only the frontmatter block is touched: the body and every other key are
    preserved exactly, so the original note stays recoverable.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise CompilationError("raw note has no frontmatter to update")
    try:
        closing = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration as exc:
        raise CompilationError("raw note frontmatter is not closed") from exc
    kept = [line for line in lines[1:closing] if not _MARKER_KEYS.match(line)]
    if kept and not kept[-1].endswith(("\n", "\r")):
        kept[-1] += "\n"
    markers = ["processed: true\n", f"compiled_into: {compiled_name}\n"]
    return "".join([lines[0], *kept, *markers, *lines[closing:]])


def compile_raw_notes(
    vault_dir: Path,
    min_raw_chunks: int = 3,
    *,
    client: OllamaClient | None = None,
    indexer: _Indexer | None = None,
) -> Optional[Path]:
    """Compile unprocessed raw notes into one indexed, linked compiled note.

    Order matters for recoverability:

    1. Generate the text (no side effects; an empty answer aborts here).
    2. Write the compiled note atomically, remembering what it replaced.
    3. Index it. On failure the previous file state is restored and the
       sources are left untouched, so the whole run can simply be repeated.
    4. Only then archive each source: mark it ``processed`` in its frontmatter
       (body untouched) and drop its chunks from the index. The vault is the
       source of truth, so a failure here is repaired by ``palace index``.
    """
    if min_raw_chunks < 1:
        raise ValueError("min_raw_chunks must be at least 1")
    vault_dir.mkdir(parents=True, exist_ok=True)

    raw_notes: list[_RawNote] = []
    for path in sorted(vault_dir.rglob("*.md")):
        try:
            metadata, body = _read_markdown(path)
        except (ValueError, yaml.YAMLError) as exc:
            raise CompilationError(f"cannot read {path}: {exc}") from exc
        if metadata.get("status") != "raw" or metadata.get("processed") is True:
            continue
        raw_notes.append(
            _RawNote(
                path=path,
                metadata=metadata,
                body=body,
                source_name=_source_name(path, metadata, body),
            )
        )

    if len(raw_notes) < min_raw_chunks:
        return None

    source_material = "\n\n".join(
        f"## Source: {note.source_name}\n{note.body}" for note in raw_notes
    )
    prompt = (
        "Synthesize the following raw notes into one cohesive, structured Markdown "
        "document. Preserve concrete facts and uncertainty, remove repetition, use "
        "descriptive headings, and return Markdown body only.\n\n"
        f"{source_material}"
    )
    body = _compiled_body(
        (client or OllamaClient()).complete(
            prompt,
            tier="reasoning",
            system="You compile personal knowledge notes without inventing facts.",
        )
    )
    if not body:
        raise CompilationError(
            "the model returned no usable content; nothing was written or archived"
        )

    source_names = [note.source_name for note in raw_notes]
    source_paths = [note.path.relative_to(vault_dir).as_posix() for note in raw_notes]
    compilation_id = uuid.uuid5(uuid.NAMESPACE_URL, "\n".join(source_paths))
    output_path = vault_dir / f"compiled-{compilation_id}.md"
    indexer = indexer or VaultIndexer()

    previous = (
        output_path.read_text(encoding="utf-8") if output_path.exists() else None
    )
    _atomic_write(
        output_path,
        _render_markdown(
            {
                "status": "compiled",
                "source_type": "document",
                "generated": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tags": ["generated"],
                "derived_from": source_names,
                "derived_from_paths": source_paths,
            },
            body,
        ),
    )
    try:
        indexer.index_file(output_path)
    except Exception as exc:
        try:
            if previous is None:
                output_path.unlink(missing_ok=True)
                indexer.remove_file(output_path)
            else:
                _atomic_write(output_path, previous)
                indexer.index_file(output_path)
        except Exception:
            pass  # best effort; the original error below is what matters
        raise CompilationError(
            f"indexing the compiled note failed ({type(exc).__name__}: {exc}); "
            "the compiled note was rolled back and no source note was changed"
        ) from exc

    failures: list[str] = []
    for note, relative in zip(raw_notes, source_paths, strict=True):
        try:
            text = note.path.read_text(encoding="utf-8")
            _atomic_write(note.path, _mark_processed(text, output_path.name))
            indexer.index_file(note.path)
        except Exception as exc:
            failures.append(f"{relative} ({type(exc).__name__}: {exc})")
    if failures:
        raise CompilationError(
            f"{output_path.name} was created and indexed, but archiving failed for: "
            + "; ".join(failures)
            + ". Fix the cause and run `palace index` to reconcile the index."
        )

    return output_path
