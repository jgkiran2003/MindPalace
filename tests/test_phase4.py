from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from palace.claims.extractor import (
    ExtractedClaim,
    ExtractedClaims,
    extract_claim_proposals,
    parse_claim_proposals,
)
from palace.claims.resolver import resolve_and_store_proposals
from palace.compilation.compiler import compile_raw_notes
from palace.config import get_settings
from palace.domain.models import ClaimProposal, Severity, SourceType
from palace.llm.client import OllamaClient
from palace.stores.claims_store import SQLiteClaimsStore


SCHEMA_PATH = Path("src/palace/stores/schema.sql")


def _proposal(value: str = "MyStorage") -> ClaimProposal:
    return ClaimProposal(
        entity="user",
        attribute="employer",
        value=value,
        source_type=SourceType.CHAT,
        severity=Severity.LOW,
        is_explicit=True,
    )


def _read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    return yaml.safe_load(frontmatter), body.strip()


@pytest.mark.parametrize(
    "response",
    [
        {
            "claims": [
                {
                    "entity": "user",
                    "attribute": "employer",
                    "value": "MyStorage",
                    "is_explicit": True,
                    "severity": "low",
                }
            ]
        },
        """```json
        [{
          "entity": "user",
          "attribute": "employer",
          "value": "MyStorage",
          "is_explicit": true,
          "severity": "low"
        }]
        ```""",
    ],
)
def test_extraction_parser_validates_claim_proposals(response: Any) -> None:
    proposals = parse_claim_proposals(response)

    assert len(proposals) == 1
    assert isinstance(proposals[0], ClaimProposal)
    assert proposals[0].entity == "user"
    assert proposals[0].attribute == "employer"
    assert proposals[0].value == "MyStorage"
    assert proposals[0].is_explicit is True
    assert proposals[0].severity == Severity.LOW
    assert proposals[0].source_type == SourceType.CHAT

class _ExtractionClient:
    def __init__(self, result: ExtractedClaims) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def structured(self, prompt: str, response_model: Any, **kwargs: Any) -> Any:
        self.calls.append({"prompt": prompt, "response_model": response_model, **kwargs})
        return self.result


def test_extractor_captures_personal_identity_without_document_context() -> None:
    client = _ExtractionClient(
        ExtractedClaims(
            claims=[
                ExtractedClaim(
                    entity="user",
                    attribute="name",
                    value="Bob",
                    is_explicit=True,
                    severity=Severity.LOW,
                )
            ]
        )
    )

    proposals = extract_claim_proposals(
        "User: Hi, my name is Bob",
        client=client,  # type: ignore[arg-type]
    )

    assert [(claim.entity, claim.attribute, claim.value) for claim in proposals] == [
        ("user", "name", "Bob")
    ]
    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert "personal identity" in call["prompt"]
    assert "document context" not in call["prompt"].lower()
    assert "Retrieved context" not in call["prompt"]


def test_extractor_returns_no_claims_for_pure_query() -> None:
    client = _ExtractionClient(ExtractedClaims(claims=[]))

    proposals = extract_claim_proposals(
        "User: What reverse proxy should I configure?",
        client=client,  # type: ignore[arg-type]
    )

    assert proposals == []
    assert "return an empty list: []" in client.calls[0]["prompt"]


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class _Session:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.requests.append({"url": url, **kwargs})
        return _Response(next(self.responses))


def test_ollama_client_routes_fast_and_reasoning_models() -> None:
    structured_json = (
        '{"claims":[{"entity":"user","attribute":"employer",'
        '"value":"MyStorage","is_explicit":true,"severity":"low"}]}'
    )
    session = _Session(
        [
            {"message": {"content": "reasoned answer"}},
            {"message": {"content": structured_json}},
        ]
    )
    client = OllamaClient(settings=get_settings(), session=session)  # type: ignore[arg-type]

    assert client.complete("question", tier="reasoning") == "reasoned answer"
    extracted = client.structured(
        "extract",
        ExtractedClaims,
        tier="fast",
        temperature=0.0,
    )

    assert extracted.claims[0].value == "MyStorage"
    assert session.requests[0]["json"]["model"] == get_settings().llm.reasoning_tier_model
    assert session.requests[1]["json"]["model"] == get_settings().llm.fast_tier_model
    assert session.requests[1]["json"]["format"]["type"] == "object"


    assert session.requests[1]["json"]["options"]["temperature"] == 0.0
class _Embedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[1.0] + [0.0] * 767 for _ in texts]


def test_resolver_dry_run_does_not_write_and_normal_run_persists(
    tmp_path: Path,
) -> None:
    embedder = _Embedder()
    proposal = _proposal()
    with SQLiteClaimsStore(
        tmp_path / "claims.db",
        schema_path=SCHEMA_PATH,
    ) as store:
        previews = resolve_and_store_proposals(
            [proposal],
            embedder,
            store,
            dry_run=True,
        )
        assert len(previews) == 1
        assert previews[0].value == "MyStorage"
        assert embedder.batches == []
        assert store.db.execute("SELECT count(*) FROM claims").fetchone()[0] == 0

        persisted = resolve_and_store_proposals([proposal], embedder, store)
        assert embedder.batches == [["MyStorage"]]
        assert persisted[0].value == "MyStorage"
        assert store.get_active_claim("user", "employer") == persisted[0]
        assert store.db.execute("SELECT count(*) FROM vec_claims").fetchone()[0] == 1


class _NoopIndexer:
    """Stands in for VaultIndexer; real indexing is covered in test_phase4_compile."""

    def __init__(self) -> None:
        self.indexed: list[Path] = []

    def index_file(self, path: Path) -> int:
        self.indexed.append(Path(path))
        return 0

    def remove_file(self, path: Path) -> int:
        return 0


class _CompilerClient:
    def __init__(self) -> None:
        self.tiers: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        tier: str,
        system: str | None = None,
    ) -> str:
        assert "First observation" in prompt
        assert "Third observation" in prompt
        assert system is not None
        self.tiers.append(tier)
        return "# Consolidated Notes\n\nA cohesive account of the observations."


def test_compiler_writes_frontmatter_and_marks_sources_processed(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    sources = []
    for index, body in enumerate(
        ["First observation", "Second observation", "Third observation"],
        start=1,
    ):
        source = vault / f"raw-{index}.md"
        source.write_text(
            "---\n"
            f"title: Source {index}\n"
            "status: raw\n"
            "source_type: document\n"
            "---\n\n"
            f"# Source {index}\n\n{body}\n",
            encoding="utf-8",
        )
        sources.append(source)

    client = _CompilerClient()
    indexer = _NoopIndexer()
    output = compile_raw_notes(vault, client=client, indexer=indexer)  # type: ignore[arg-type]

    assert output is not None
    assert output.exists()
    assert client.tiers == ["reasoning"]
    metadata, body = _read_markdown(output)
    assert metadata["status"] == "compiled"
    assert metadata["source_type"] == "document"
    assert metadata["derived_from"] == ["Source 1", "Source 2", "Source 3"]
    assert metadata["derived_from_paths"] == ["raw-1.md", "raw-2.md", "raw-3.md"]
    assert metadata["generated"] is True
    assert body.startswith("# Consolidated Notes")

    for source in sources:
        source_metadata, source_body = _read_markdown(source)
        assert source_metadata["status"] == "raw"
        assert source_metadata["processed"] is True
        assert source_metadata["compiled_into"] == output.name
        assert "observation" in source_body.lower()

    # compiled note indexed first, then every source refreshed (archived)
    assert indexer.indexed == [output, *sources]
    assert compile_raw_notes(vault, client=client, indexer=indexer) is None  # type: ignore[arg-type]
