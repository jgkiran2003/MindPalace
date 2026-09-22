"""Markdown/frontmatter parsing into the locked ``DocumentChunk`` model."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from palace.config import Settings, get_settings
from palace.domain.models import ChunkStatus, DocumentChunk, Severity, SourceType

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_CHUNK_NAMESPACE = uuid.UUID("9e9da7d6-7f7a-4a0f-b9f0-1b84f3202671")


@dataclass(frozen=True)
class MarkdownSection:
    """A minimal AST node: its heading ancestry and body blocks."""

    headings: tuple[str, ...]
    body: str

    @property
    def text(self) -> str:
        prefix = "\n".join(f"{'#' * (index + 1)} {heading}" for index, heading in enumerate(self.headings))
        return "\n\n".join(part for part in (prefix, self.body.strip()) if part).strip()


def _split_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown
    try:
        closing = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration:
        return {}, markdown
    loaded = yaml.safe_load("\n".join(lines[1:closing])) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Markdown YAML frontmatter must be a mapping")
    return loaded, "\n".join(lines[closing + 1 :])


def _parse_sections(markdown: str) -> list[MarkdownSection]:
    """Parse heading/paragraph nodes while treating fenced code as opaque."""
    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush(*, include_heading_only: bool = False) -> None:
        text = "\n".join(body).strip()
        if text or (include_heading_only and heading_stack):
            sections.append(MarkdownSection(tuple(heading_stack), text))
        body.clear()

    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            body.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_stack[level - 1 :] = [match.group(2).strip()]
        elif not in_fence and not line.strip():
            flush()
        else:
            body.append(line)
    flush(include_heading_only=True)
    return [section for section in sections if section.text]


def _as_string(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def chunk_markdown(
    path: str | Path,
    *,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    """Read one Markdown file and produce one chunk per heading section."""
    resolved_settings = settings or get_settings()
    source = Path(path)
    metadata, markdown = _split_frontmatter(source.read_text(encoding="utf-8"))
    sections = _parse_sections(markdown)
    doc_path = str(source)
    zero_vector = [0.0] * resolved_settings.embedding.dimension

    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, str):
        tags = [item.strip() for item in raw_tags.split(",") if item.strip()]
    elif isinstance(raw_tags, list):
        tags = [str(item) for item in raw_tags]
    else:
        raise ValueError("frontmatter 'tags' must be a list or comma-separated string")

    common: dict[str, Any] = {
        "doc_path": doc_path,
        "vector": zero_vector,
        "status": ChunkStatus(metadata.get("status", ChunkStatus.COMPILED.value)),
        "derived_from_chunk_id": metadata.get("derived_from_chunk_id"),
        "tags": tags,
        "severity": Severity(metadata.get("severity", Severity.LOW.value)),
        "vault_id": metadata.get("vault_id"),
        "source_type": SourceType(metadata.get("source_type", SourceType.DOCUMENT.value)),
    }
    if metadata.get("created_at") is not None:
        common["created_at"] = _as_string(metadata["created_at"])
    if metadata.get("updated_at") is not None:
        common["updated_at"] = _as_string(metadata["updated_at"])

    chunks: list[DocumentChunk] = []
    for index, section in enumerate(sections):
        chunk_id = str(uuid.uuid5(_CHUNK_NAMESPACE, f"{doc_path}:{index}"))
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_index=index,
                text=section.text,
                **common,
            )
        )
    return chunks


class MarkdownChunker:
    """Callable class form for dependency injection in the indexer."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def chunk(self, path: str | Path) -> list[DocumentChunk]:
        return chunk_markdown(path, settings=self.settings)
