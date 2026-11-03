#!/usr/bin/env python3
"""Benchmark claim-extraction prompts and Ollama models against ground truth."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from palace.claims.extractor import ExtractedClaim, ExtractedClaims  # noqa: E402
from palace.config import Settings, get_settings  # noqa: E402
from palace.llm.client import OllamaClient  # noqa: E402

DEFAULT_MODELS = ("qwen3:4b", "qwen2.5:3b", "qwen2.5:7b")
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "data" / "eval_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval_results.json"


class ExpectedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    attribute: str
    value: str


class DialogueMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant", "user"]
    content: str


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    dialogue: list[DialogueMessage]
    expected_claims: list[ExpectedClaim]


class EvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    cases: list[EvalCase]


class RecordingSession(requests.Session):
    """Capture Ollama's unmodified HTTP payload for audit output."""

    def __init__(self) -> None:
        super().__init__()
        self.last_payload: Any = None
        self.last_status: int | None = None
        self.last_text: str | None = None

    def reset_recording(self) -> None:
        self.last_payload = None
        self.last_status = None
        self.last_text = None

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        response = super().post(url, **kwargs)
        self.last_status = response.status_code
        self.last_text = response.text
        try:
            self.last_payload = response.json()
        except requests.exceptions.JSONDecodeError:
            self.last_payload = response.text
        return response


PromptBuilder = Callable[[str], str]


def variant_concise_rules(dialogue: str) -> str:
    return (
        "Extract durable factual claims asserted or confirmed by the user. "
        "Never extract questions, greetings, opinions, hypotheticals, or assistant "
        "claims. Assistant text may only resolve what a short user answer refers to. "
        "Return no claims when the user asserted no fact.\n\n"
        f"Dialogue:\n{dialogue}"
    )


def variant_few_shot(dialogue: str) -> str:
    return (
        "Extract only durable facts asserted or confirmed by the user. Assistant messages are usable only for coreference.\n\n"
        "Example 1 (Multiple facts & User identity)\n"
        "Dialogue: User: I live in London and my role is devops.\n"
        'Output: {"claims":[{"entity":"user","attribute":"location","value":"London","is_explicit":true,"severity":"low"}, '
        '{"entity":"user","attribute":"role","value":"devops","is_explicit":true,"severity":"low"}]}\n\n'
        "Example 2 (Technical configuration)\n"
        "Dialogue: User: My home server is named atlas.\n"
        'Output: {"claims":[{"entity":"home_server","attribute":"name","value":"atlas","is_explicit":true,"severity":"low"}]}\n\n'
        "Example 3 (Negative control)\n"
        "Dialogue: User: Which database should I install?\n"
        'Output: {"claims":[]}\n\n'
        "Now extract claims from this dialogue:\n"
        f"{dialogue.strip()}"
    )


def variant_strict_schema(dialogue: str) -> str:
    return (
        "Extract only claims factually asserted or confirmed by the user. For each "
        "claim: entity is a stable lowercase snake_case subject identifier; attribute "
        "is a lowercase snake_case property; value is the exact asserted value; "
        "is_explicit is true only for an unambiguous assertion; severity is low, "
        "medium, or high. Capture user identity and technical configuration facts. "
        "Do not extract assistant content, questions, opinions, plans, hypotheticals, "
        "or social chatter. Return an empty claims list when nothing qualifies.\n\n"
        f"Dialogue:\n{dialogue}"
    )


PROMPT_VARIANTS: dict[str, PromptBuilder] = {
    # "variant_concise_rules": variant_concise_rules,
    "variant_few_shot": variant_few_shot,
    # "variant_strict_schema": variant_strict_schema,
}


def load_dataset(path: Path) -> EvalDataset:
    return EvalDataset.model_validate_json(path.read_text(encoding="utf-8"))


def format_dialogue(messages: Sequence[DialogueMessage]) -> str:
    return "\n".join(
        f"{message.role.capitalize()}: {message.content}" for message in messages
    )


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _normalize_value(value: str) -> str:
    return " ".join(value.casefold().split())


def normalize_claim(claim: ExpectedClaim | ExtractedClaim) -> tuple[str, str, str]:
    return (
        _normalize_identifier(claim.entity),
        _normalize_identifier(claim.attribute),
        _normalize_value(claim.value),
    )


def score_case(
    expected: Sequence[ExpectedClaim],
    predicted: Sequence[ExtractedClaim],
) -> dict[str, Any]:
    expected_set = {normalize_claim(claim) for claim in expected}
    predicted_set = {normalize_claim(claim) for claim in predicted}
    true_positives = expected_set & predicted_set
    false_positives = predicted_set - expected_set
    false_negatives = expected_set - predicted_set
    return {
        "exact_match": predicted_set == expected_set,
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "normalized_expected": [list(item) for item in sorted(expected_set)],
        "normalized_predicted": [list(item) for item in sorted(predicted_set)],
        "unexpected_claims": [list(item) for item in sorted(false_positives)],
        "missing_claims": [list(item) for item in sorted(false_negatives)],
    }


def _settings_for_model(settings: Settings, model: str) -> Settings:
    return settings.model_copy(
        update={
            "llm": settings.llm.model_copy(update={"fast_tier_model": model})
        }
    )


def evaluate_combination(
    model: str,
    variant_name: str,
    prompt_builder: PromptBuilder,
    cases: Sequence[EvalCase],
    settings: Settings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = RecordingSession()
    client = OllamaClient(
        settings=_settings_for_model(settings, model),
        session=session,
        fast_model=model,
    )
    records: list[dict[str, Any]] = []
    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "exact_matches": 0,
        "schema_compliant": 0,
        "invalid_json_errors": 0,
    }

    try:
        for case in cases:
            dialogue = format_dialogue(case.dialogue)
            prompt = prompt_builder(dialogue)
            session.reset_recording()
            started = perf_counter()
            parsed: ExtractedClaims | None = None
            error: dict[str, str] | None = None
            invalid_json = False
            try:
                parsed = client.structured(
                    prompt,
                    ExtractedClaims,
                    tier="fast",
                    system=(
                        "You are a deterministic claim extraction engine. Return only "
                        "JSON matching the supplied schema, without commentary."
                    ),
                    temperature=0.0,
                )
                totals["schema_compliant"] += 1
                score = score_case(case.expected_claims, parsed.claims)
            except (ValidationError, json.JSONDecodeError) as exc:
                invalid_json = True
                totals["invalid_json_errors"] += 1
                error = {"type": type(exc).__name__, "message": str(exc)}
                score = score_case(case.expected_claims, [])
                score["exact_match"] = False
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
                score = score_case(case.expected_claims, [])
                score["exact_match"] = False
            latency = perf_counter() - started

            totals["true_positives"] += score["true_positives"]
            totals["false_positives"] += score["false_positives"]
            totals["false_negatives"] += score["false_negatives"]
            totals["exact_matches"] += int(score["exact_match"])
            records.append(
                {
                    "model": model,
                    "prompt_variant": variant_name,
                    "case_id": case.id,
                    "dialogue": [message.model_dump() for message in case.dialogue],
                    "expected_claims": [
                        claim.model_dump() for claim in case.expected_claims
                    ],
                    "prompt": prompt,
                    "raw_response": session.last_payload,
                    "http_status": session.last_status,
                    "parsed_claims": (
                        [claim.model_dump(mode="json") for claim in parsed.claims]
                        if parsed is not None
                        else None
                    ),
                    "schema_compliant": parsed is not None,
                    "invalid_json": invalid_json,
                    "latency_seconds": latency,
                    "score": score,
                    "error": error,
                }
            )
    finally:
        session.close()

    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 1.0
    case_count = len(cases)
    latencies = [record["latency_seconds"] for record in records]
    summary = {
        "model": model,
        "prompt_variant": variant_name,
        "case_count": case_count,
        "accuracy_percent": (
            100.0 * totals["exact_matches"] / case_count if case_count else 0.0
        ),
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "schema_compliance_percent": (
            100.0 * totals["schema_compliant"] / case_count if case_count else 0.0
        ),
        "invalid_json_errors": totals["invalid_json_errors"],
        "invalid_json_error_rate_percent": (
            100.0 * totals["invalid_json_errors"] / case_count
            if case_count
            else 0.0
        ),
        "average_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
    }
    return summary, records


def render_markdown_table(summaries: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| Model | Prompt Variant | Accuracy (%) | F1 | False Positives | Avg Latency (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {model} | {variant} | {accuracy:.1f} | {f1:.3f} | {fp} | {latency:.3f} |".format(
                model=str(summary["model"]).replace("|", "\\|"),
                variant=str(summary["prompt_variant"]).replace("|", "\\|"),
                accuracy=summary["accuracy_percent"],
                f1=summary["f1"],
                fp=summary["false_positives"],
                latency=summary["average_latency_seconds"],
            )
        )
    return "\n".join(lines)


def parse_models(value: str) -> list[str]:
    models = list(dict.fromkeys(model.strip() for model in value.split(",") if model.strip()))
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one model")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate claim extraction prompts and Ollama fast-tier models."
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=list(DEFAULT_MODELS),
        metavar="M1,M2",
        help="Comma-separated Ollama models (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON audit output path (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Evaluation dataset path (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.dataset)
    settings = get_settings()
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for model in args.models:
        for variant_name, prompt_builder in PROMPT_VARIANTS.items():
            print(f"Evaluating {model} / {variant_name}...", file=sys.stderr)
            summary, combination_records = evaluate_combination(
                model,
                variant_name,
                prompt_builder,
                dataset.cases,
                settings,
            )
            summaries.append(summary)
            records.extend(combination_records)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_version": dataset.version,
        "models": args.models,
        "prompt_variants": list(PROMPT_VARIANTS),
        "summaries": summaries,
        "records": records,
        "failure_cases": [
            record
            for record in records
            if not record["score"]["exact_match"]
            or not record["schema_compliant"]
        ],
    }
    output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(render_markdown_table(summaries))
    print(f"\nRaw results: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
