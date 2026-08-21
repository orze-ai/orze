"""Launch-time aggregate validator coverage."""

from orze.engine.launcher import _eval_validator_rule


RULE = {
    "field_sum": "training.datasets.*.samples",
    "operator": "lte",
    "value": 36_000,
}


def test_field_sum_rejects_oversized_nested_datamix():
    cfg = {
        "training": {
            "datasets": {
                "ami": {"samples": 20_000},
                "gigaspeech": {"samples": 20_000},
            }
        }
    }

    assert "40000" in _eval_validator_rule(RULE, cfg)


def test_field_sum_accepts_scalar_dataset_spelling():
    cfg = {"training": {"datasets": {"ami": 18_000, "gigaspeech": 18_000}}}

    assert _eval_validator_rule(RULE, cfg) is None


def test_field_sum_composes_inside_any_of_scope():
    rule = {
        "any_of": [
            {"field": "strategy", "operator": "not_in", "value": ["ctc"]},
            {
                "field_sum": "training.datasets.*.samples",
                "operator": "lte",
                "value": 18_000,
            },
        ],
        "explanation": "oversized CTC mix",
    }
    cfg = {
        "strategy": "ctc",
        "training": {"datasets": {"ami": {"samples": 10_000},
                                    "gigaspeech": {"samples": 10_000}}},
    }

    assert _eval_validator_rule(rule, cfg) == "oversized CTC mix"
