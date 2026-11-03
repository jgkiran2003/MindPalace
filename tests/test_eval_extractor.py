from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from palace.claims.extractor import ExtractedClaim
from palace.domain.models import Severity

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval_extractor.py"
SPEC = importlib.util.spec_from_file_location("eval_extractor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
eval_extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = eval_extractor
SPEC.loader.exec_module(eval_extractor)

DEFAULT_DATASET = eval_extractor.DEFAULT_DATASET
ExpectedClaim = eval_extractor.ExpectedClaim
load_dataset = eval_extractor.load_dataset
parse_models = eval_extractor.parse_models
render_markdown_table = eval_extractor.render_markdown_table
score_case = eval_extractor.score_case

def test_eval_dataset_contains_required_ground_truth_cases() -> None:
    dataset = load_dataset(DEFAULT_DATASET)
    by_id = {case.id: case for case in dataset.cases}

    assert len(dataset.cases) >= 8
    assert set(by_id) >= {
        "direct_identity",
        "direct_tech",
        "question_identity",
        "question_tech",
        "chit_chat",
        "opinion",
        "hypothetical",
        "coreference",
    }
    identity = by_id["direct_identity"]
    assert {(claim.entity, claim.attribute, claim.value) for claim in identity.expected_claims} == {
        ("user", "name", "Bob"),
        ("user", "role", "software engineer"),
    }
    assert by_id["question_identity"].expected_claims == []
    assert by_id["question_tech"].expected_claims == []


def test_score_case_normalizes_identifiers_and_counts_errors() -> None:
    expected = [
        ExpectedClaim(entity="pi_hole", attribute="ip_address", value="192.168.1.5")
    ]
    predicted = [
        ExtractedClaim(
            entity="Pi-hole",
            attribute="IP Address",
            value="192.168.1.5",
            is_explicit=True,
            severity=Severity.LOW,
        ),
        ExtractedClaim(
            entity="pi_hole",
            attribute="purpose",
            value="ad blocking",
            is_explicit=True,
            severity=Severity.LOW,
        ),
    ]

    score = score_case(expected, predicted)

    assert score["true_positives"] == 1
    assert score["false_positives"] == 1
    assert score["false_negatives"] == 0
    assert score["exact_match"] is False


def test_cli_model_parser_and_markdown_summary() -> None:
    assert parse_models("qwen3:4b,qwen2.5:7b,qwen3:4b") == [
        "qwen3:4b",
        "qwen2.5:7b",
    ]
    table = render_markdown_table(
        [
            {
                "model": "qwen3:4b",
                "prompt_variant": "variant_few_shot",
                "accuracy_percent": 87.5,
                "f1": 0.8,
                "false_positives": 1,
                "average_latency_seconds": 0.25,
            }
        ]
    )

    assert "| Model | Prompt Variant | Accuracy (%) |" in table
    assert "| qwen3:4b | variant_few_shot | 87.5 | 0.800 | 1 | 0.250 |" in table
    assert Path(DEFAULT_DATASET).name == "eval_cases.json"
