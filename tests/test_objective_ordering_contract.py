"""V1-01C: public report/recovery consumers obey the declared objective.

These tests use real lifecycle rows and on-disk evidence. They deliberately do
not import a proposed comparator: a correct helper is insufficient unless the
report, its sweep winners, and recovered champion all consume the same order.
"""

import json

import pytest

from orze.engine.rebuild_state import rebuild_best_from_evidence
from orze.idea_lake import IdeaLake
from orze.reporting.leaderboard import update_report


_NO_SOURCE = object()


@pytest.fixture
def campaign(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    try:
        yield results, lake, {}
    finally:
        lake.close()


def _publish(campaign, idea_id, metrics, *, source=_NO_SOURCE):
    results, lake, ideas = campaign
    document = {"status": "COMPLETED", **metrics}
    idea_dir = results / idea_id
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        json.dumps(document), encoding="utf-8")
    if source is not _NO_SOURCE:
        (idea_dir / "evaluation.json").write_text(
            json.dumps(source), encoding="utf-8")
    lake.insert(
        idea_id, idea_id, "{}", "", status="completed",
        eval_metrics=document)
    ideas[idea_id] = {"title": idea_id}


def _config(campaign, order, *, medal=False, secondary=False):
    _, lake, _ = campaign
    columns = [{"key": "score", "label": "Score"}]
    if medal:
        columns.append({"key": "medal", "label": "Medal"})
    report = {
        "title": "Objective contract",
        "primary_metric": "score",
        "sort": order,
        "columns": columns,
    }
    if secondary:
        report["secondary_metric"] = "penalty"
        columns.append({
            "key": "penalty", "label": "Penalty",
            "source": "evaluation.json:penalty",
        })
    return {"idea_lake_db": str(lake.db_path), "report": report}


def _observe(campaign, cfg):
    results, lake, ideas = campaign
    rows = update_report(results, ideas, cfg, lake=lake)
    recovered, _ = rebuild_best_from_evidence(results, cfg, lake=lake)
    panel = json.loads(
        (results / "_leaderboard.json").read_text(encoding="utf-8"))
    return rows, recovered, [row["idea_id"] for row in panel["top"]]


def _assert_order(campaign, cfg, expected, *, main_expected=None):
    rows, recovered, panel_ids = _observe(campaign, cfg)
    actual = {
        "report": [row["id"] for row in rows],
        "recovered": recovered,
        "panel": panel_ids,
    }
    assert actual == {
        "report": expected,
        "recovered": expected[0] if expected else None,
        "panel": expected if main_expected is None else main_expected,
    }
    return rows


@pytest.mark.parametrize(
    "order,expected",
    [
        ("descending", ["idea-high", "idea-low"]),
        ("ascending", ["idea-low", "idea-high"]),
    ],
)
def test_primary_objective_overrides_conflicting_medals(campaign, order, expected):
    _publish(campaign, "idea-low", {"score": 1, "medal": "gold"})
    _publish(campaign, "idea-high", {"score": 10, "medal": "bronze"})

    _assert_order(campaign, _config(campaign, order, medal=True), expected)


def test_adding_and_removing_medal_display_column_cannot_change_champion(campaign):
    _publish(campaign, "idea-low", {"score": 1, "medal": "gold"})
    _publish(campaign, "idea-high", {"score": 10, "medal": "bronze"})
    champions = []
    for display_medal in (False, True, False):
        rows, recovered, panel = _observe(
            campaign, _config(campaign, "descending", medal=display_medal))
        champions.append((rows[0]["id"], recovered, panel[0]))

    assert champions == [("idea-high", "idea-high", "idea-high")] * 3


@pytest.mark.parametrize(
    "order,measured",
    [
        ("ascending", ["idea-negative", "idea-zero", "idea-one"]),
        ("descending", ["idea-one", "idea-zero", "idea-negative"]),
    ],
)
def test_secondary_exact_source_preserves_zero_negative_and_missing(
        campaign, order, measured):
    # A source mapping is authoritative even when the value is zero or absent.
    # The raw values would deliberately reverse the desired ordering.
    for idea_id, raw, source in (
        ("idea-negative", -2, {"penalty": -2}),
        ("idea-zero", 100, {"penalty": 0}),
        ("idea-one", 1, {"penalty": 1}),
        ("idea-a-missing", -100, {}),
        ("idea-b-null", 100, {"penalty": None}),
    ):
        _publish(
            campaign, idea_id, {"score": 5, "penalty": raw}, source=source)

    rows = _assert_order(
        campaign, _config(campaign, order, secondary=True),
        measured + ["idea-a-missing", "idea-b-null"])
    values = {row["id"]: row["values"]["penalty"] for row in rows}
    assert values["idea-zero"] == 0
    assert values["idea-negative"] == -2
    assert values["idea-a-missing"] is None
    assert values["idea-b-null"] is None


@pytest.mark.parametrize("order", ["ascending", "descending"])
def test_nonfinite_or_boolean_secondary_is_missing_not_a_competing_score(
        campaign, order):
    for idea_id, value in (
        ("idea-a-nan", float("nan")),
        ("idea-b-inf", float("inf")),
        ("idea-c-neg-inf", float("-inf")),
        ("idea-d-bool", False),
        ("idea-z-measured", -3),
    ):
        _publish(
            campaign, idea_id, {"score": 5, "penalty": 100},
            source={"penalty": value})
    cfg = _config(campaign, order, secondary=True)
    # Keep the finite primary observation eligible while explicitly permitting
    # nonfinite ancillary diagnostics. Default validation is tested separately.
    cfg["metric_validation"] = {"reject_nan": False, "reject_inf": False}

    _assert_order(campaign, cfg, [
        "idea-z-measured", "idea-a-nan", "idea-b-inf",
        "idea-c-neg-inf", "idea-d-bool",
    ])


def test_default_nonfinite_validation_is_not_weakened_by_sorting(campaign):
    _publish(
        campaign, "idea-invalid", {"score": 10},
        source={"penalty": float("nan")})
    _publish(campaign, "idea-valid", {"score": 1}, source={"penalty": 2})

    _assert_order(
        campaign, _config(campaign, "descending", secondary=True), ["idea-valid"])


@pytest.mark.parametrize(
    "order,best_secondary,other_secondary",
    [("ascending", 2, 3), ("descending", 3, 2)],
)
def test_explicit_secondary_and_stable_id_agree_with_recovery(
        campaign, order, best_secondary, other_secondary):
    # Insert winners in reverse identifier order. The secondary criterion must
    # outrank IDs, but exact ties must use the same stable ID in both directions.
    for idea_id, secondary in (
        ("idea-z-tied", best_secondary),
        ("idea-b-tied", best_secondary),
        ("idea-a-weaker", other_secondary),
    ):
        _publish(
            campaign, idea_id, {"score": 5}, source={"penalty": secondary})

    _assert_order(
        campaign, _config(campaign, order, secondary=True),
        ["idea-b-tied", "idea-z-tied", "idea-a-weaker"])


def test_undeclared_secondary_cannot_change_stable_primary_tie(campaign):
    _publish(campaign, "idea-z", {"score": 5, "penalty": 100})
    _publish(campaign, "idea-a", {"score": 5, "penalty": -100})
    cfg = _config(campaign, "descending")
    cfg["report"]["columns"].append({"key": "penalty", "label": "Diagnostic"})

    _assert_order(campaign, cfg, ["idea-a", "idea-z"])


def test_sweep_winner_uses_primary_objective_not_medal(campaign):
    _publish(campaign, "idea-family-ht-a", {"score": 1, "medal": "gold"})
    _publish(campaign, "idea-family-ht-z", {"score": 10, "medal": "bronze"})
    _publish(campaign, "idea-standalone", {"score": 5, "medal": "silver"})

    _assert_order(
        campaign, _config(campaign, "descending", medal=True),
        ["idea-family-ht-z", "idea-standalone", "idea-family-ht-a"],
        main_expected=["idea-family-ht-z", "idea-standalone"])


def test_sweep_winner_uses_exact_secondary_source(campaign):
    _publish(
        campaign, "idea-family-ht-a", {"score": 5, "penalty": 1},
        source={"penalty": 1})
    _publish(
        campaign, "idea-family-ht-z", {"score": 5, "penalty": 100},
        source={"penalty": 0})
    _publish(
        campaign, "idea-standalone", {"score": 6, "penalty": -100},
        source={"penalty": -100})
    cfg = _config(campaign, "ascending", secondary=True)

    _assert_order(
        campaign, cfg,
        ["idea-family-ht-z", "idea-family-ht-a", "idea-standalone"],
        main_expected=["idea-family-ht-z", "idea-standalone"])
    results, _, _ = campaign
    main_table = (results / "report.md").read_text(encoding="utf-8").split(
        "## Results", 1)[1].split("## Sweep Details", 1)[0]
    assert "idea-family-ht-z" in main_table
    assert "idea-family-ht-a" not in main_table
