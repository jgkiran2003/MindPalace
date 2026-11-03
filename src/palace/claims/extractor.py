"""LLM-driven extraction of validated claim proposals."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from palace.claims.outcomes import ClaimExtractionLLMError, MalformedExtractionError
from palace.domain.models import ClaimProposal, Severity, SourceType
from palace.llm.client import OllamaClient


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    is_explicit: bool
    severity: Severity


class ExtractedClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim]


def _json_payload(response: str | dict[str, Any] | list[Any]) -> Any:
    if not isinstance(response, str):
        return response
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def parse_claim_proposals(
    response: str | dict[str, Any] | list[Any] | ExtractedClaims,
) -> list[ClaimProposal]:
    """Validate structured JSON and adapt it to the locked domain model."""
    if isinstance(response, ExtractedClaims):
        extracted = response
    else:
        payload = _json_payload(response)
        if isinstance(payload, list):
            payload = {"claims": payload}
        extracted = ExtractedClaims.model_validate(payload)

    return [
        ClaimProposal(
            entity=claim.entity,
            attribute=claim.attribute,
            value=claim.value,
            is_explicit=claim.is_explicit,
            severity=claim.severity,
            source_type=SourceType.CHAT,
        )
        for claim in extracted.claims
    ]


def extract_claim_proposals(
    text: str,
    *,
    client: OllamaClient | None = None,
) -> list[ClaimProposal]:
    """Extract user-asserted durable claims from dialogue only."""
    llm = client or OllamaClient()
    prompt = (
        "Extract ONLY durable, factual claims asserted or confirmed by the USER.\n\n"
        "Capture both personal identity/profile facts and technical facts.\n"
        "- For personal facts, use entity='user' with attributes such as name, "
        "role, workplace, location, and preferences.\n"
        "- For technical facts, capture hardware, software, services, ports, IPs, "
        "and configurations using stable entity and attribute names.\n\n"
        "Use assistant messages in the dialogue strictly to resolve user coreferences "
        "(e.g., Assistant: 'What database?' -> User: 'PostgreSQL' maps to "
        "entity='database', attribute='engine', value='PostgreSQL').\n"
        "Do NOT extract claims from the Assistant's own answers, suggestions, or "
        "mentions.\n"
        "Do NOT ignore personal identity statements made during greetings (e.g., "
        "'My name is Bob' -> entity='user', attribute='name', value='Bob').\n"
        "If the user asks a question or makes no factual assertions, return an empty "
        "list: [].\n\n"
        f"Dialogue:\n{text.strip()}"
    )
    # ValueError covers pydantic ValidationError, JSON decode errors (including
    # requests' own) and empty model content; OSError covers connection,
    # timeout and HTTP status failures. Order matters: requests' JSONDecodeError
    # is both, and means the service answered with garbage.
    try:
        result = llm.structured(
            prompt,
            ExtractedClaims,
            tier="fast",
            system=(
                "You are a precise structured fact extractor. Output only valid JSON "
                "matching the supplied schema, without thoughts or commentary."
            ),
            temperature=0.0,
        )
        return parse_claim_proposals(result)
    except ValueError as exc:
        raise MalformedExtractionError(f"{type(exc).__name__}: {exc}") from exc
    except OSError as exc:
        raise ClaimExtractionLLMError(f"{type(exc).__name__}: {exc}") from exc
