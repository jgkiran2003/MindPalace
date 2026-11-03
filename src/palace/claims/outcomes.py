"""Typed failure and result shapes for the claim extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from palace.domain.models import Claim, ClaimProposal


class ClaimPipelineError(Exception):
    """Base class for failures that are reported rather than swallowed."""


class ClaimExtractionLLMError(ClaimPipelineError):
    """The model could not be reached or answered with an HTTP/timeout error."""


class MalformedExtractionError(ClaimPipelineError):
    """The model answered, but not with valid structured claim output."""


class ClaimEmbeddingError(ClaimPipelineError):
    """Embedding the proposals failed before anything was written."""


class ClaimStorageError(ClaimPipelineError):
    """The claims store failed; ``persisted`` holds claims committed before it."""

    def __init__(self, message: str, persisted: list[Claim] | None = None) -> None:
        super().__init__(message)
        self.persisted: list[Claim] = persisted or []


class ClaimOutcome(str, Enum):
    SUCCESS = "success"
    NO_CLAIMS = "no_claims"
    MALFORMED_OUTPUT = "malformed_output"
    LLM_FAILURE = "llm_failure"
    STORAGE_FAILURE = "storage_failure"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass
class ClaimExchangeResult:
    """What happened to one chat exchange on its way through the pipeline."""

    outcome: ClaimOutcome
    dry_run: bool = False
    proposals: list[ClaimProposal] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in (ClaimOutcome.SUCCESS, ClaimOutcome.NO_CLAIMS)

    def describe(self) -> str | None:
        """One-line, user-facing summary; ``None`` when there is nothing to say."""
        if self.outcome is ClaimOutcome.NO_CLAIMS:
            return None
        if self.outcome is ClaimOutcome.SUCCESS:
            if self.dry_run:
                return f"{len(self.proposals)} claim(s) proposed (dry run, nothing saved)"
            detail = ", ".join(
                f"{claim.claim_key}={claim.value} ({claim.status.value})"
                for claim in self.claims
            )
            return f"saved {len(self.claims)} claim(s): {detail}"
        label = {
            ClaimOutcome.MALFORMED_OUTPUT: "model returned malformed claim output",
            ClaimOutcome.LLM_FAILURE: "model or embedding service unavailable",
            ClaimOutcome.STORAGE_FAILURE: "claims store failed",
            ClaimOutcome.UNEXPECTED_ERROR: "unexpected error",
        }[self.outcome]
        suffix = f": {self.error}" if self.error else ""
        saved = (
            f" ({len(self.claims)} claim(s) were saved before the failure)"
            if self.claims
            else ""
        )
        return f"claim extraction failed, {label}{suffix}{saved}"
