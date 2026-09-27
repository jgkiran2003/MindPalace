"""Compile processed batches of raw Markdown notes into a cohesive note."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from palace.llm.client import OllamaClient

_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


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


def compile_raw_notes(
    vault_dir: Path,
    min_raw_chunks: int = 3,
    *,
    client: OllamaClient | None = None,
) -> Optional[Path]:
    """Compile unprocessed raw notes and mark each source as processed."""
    if min_raw_chunks < 1:
        raise ValueError("min_raw_chunks must be at least 1")
    vault_dir.mkdir(parents=True, exist_ok=True)

    raw_notes: list[_RawNote] = []
    for path in sorted(vault_dir.rglob("*.md")):
        metadata, body = _read_markdown(path)
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

    source_names = [note.source_name for note in raw_notes]
    compilation_id = uuid.uuid5(uuid.NAMESPACE_URL, "\n".join(source_names))
    output_path = vault_dir / f"compiled-{compilation_id}.md"
    output_path.write_text(
        _render_markdown(
            {
                "status": "compiled",
                "source_type": "document",
                "derived_from": source_names,
            },
            body,
        ),
        encoding="utf-8",
    )

    for note in raw_notes:
        note.metadata["processed"] = True
        note.metadata["compiled_into"] = output_path.name
        note.path.write_text(
            _render_markdown(note.metadata, note.body),
            encoding="utf-8",
        )

    return output_path
