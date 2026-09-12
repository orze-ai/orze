"""S3 actual live consumers; declarations do not infer meaning from metric names."""
from copy import deepcopy
import json

import pytest

from orze.core.config import DEFAULT_CONFIG, _validate_config
from orze.engine.champion_history import objective_scope
from orze.engine.rebuild_state import rebuild_best_from_evidence
from orze.reporting import evidence, leaderboard
from orze.research.context_builder import build_digest
from test_report_authority_compatibility import project, _publish, _oracle, _payload


def report(keys, *, declared=True, minimum=2):
    value = {"primary_metric": "score", "sort": "ascending", "min_datasets": minimum,
             "columns": [{"key": "score"}, *({"key": key} for key in keys),
                         {"key": "elapsed"}]}
    if declared:
        value["dataset_keys"] = list(keys)
    return value


def test_declared_live_qualification_is_invariant_to_metric_renaming(tmp_path):
    outcomes = []
    for name in ("loss_a", "wer_a"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps(
            {"status": "COMPLETED", "score": 0, name: -1, "loss_b": 0, "elapsed": 100}))
        cfg = {"report": report([name, "loss_b"])}
        outcomes.append(evidence.qualify_local_report_evidence(folder, cfg)[2:])
    assert outcomes[0] == (0.0, "local_evidence_verified")
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize("name", ["loss_a", "wer_a"])
def test_live_positive_gate_without_coverage_declaration_is_explicitly_refused(tmp_path, name):
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"status": "COMPLETED", "score": 0, name: 1, "part_b": 2, "elapsed": 100}))
    cfg = {"report": report([name, "part_b"], declared=False)}
    assert evidence.qualify_local_report_evidence(tmp_path, cfg)[2:] == (
        None, "dataset_coverage_not_declared")
    cfg["report"]["min_datasets"] = 0
    assert evidence.qualify_local_report_evidence(tmp_path, cfg)[2:] == (
        0.0, "local_evidence_verified")


def test_display_aggregate_and_time_cannot_fill_missing_declared_dataset(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"status": "COMPLETED", "score": -2, "part_a": 0, "elapsed": 100}))
    cfg = {"report": report(["part_a", "part_b"])}
    assert evidence.qualify_local_report_evidence(tmp_path, cfg)[2:] == (
        None, "metric_coverage_below_min:1/2")


@pytest.mark.parametrize("declaration", [None, "part_a", ["part_a", "part_a"], ["unknown"]])
def test_bad_declaration_is_refused_by_config_and_direct_live_entry(tmp_path, declaration):
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"status": "COMPLETED", "score": 0, "part_a": 1, "part_b": 2}))
    cfg = {"report": report(["part_a", "part_b"])}
    cfg["report"]["dataset_keys"] = declaration
    assert evidence.qualify_local_report_evidence(tmp_path, cfg)[2:] == (
        None, "dataset_coverage_declaration_invalid")
    check_cfg = deepcopy(DEFAULT_CONFIG)
    check_cfg["report"] = cfg["report"]
    errors, _ = _validate_config(check_cfg)
    assert any("dataset_coverage_declaration_invalid" in error for error in errors)


def test_current_report_rebuild_and_digest_use_declared_mixed_name_coverage(project):
    p = project
    p.cfg["report"] = report(["wer_a", "part_b"])
    _publish(p, score=0, wer_a=1, part_b=2)
    assert _oracle(p)[2:4] == (0.0, "local_evidence_verified")
    rows = leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [(r["id"], r["primary_val"]) for r in rows] == [("idea-native", 0.0)]
    assert _payload(p.results)["top"][0]["idea_id"] == "idea-native"
    best, value = rebuild_best_from_evidence(p.results, p.cfg, lake=p.lake)
    assert (best, value) == ("idea-native", 0.0)
    digest = build_digest(p.results, p.cfg)
    assert "idea-native" in digest
    assert "qualification: 1 accepted, 0 rejected" in digest


def test_same_columns_changed_coverage_revoke_cache_and_history_scope(project):
    p = project
    p.cfg["report"] = report(["part_a", "part_b"], minimum=1)
    p.cfg["report"]["dataset_keys"] = ["part_a"]
    _publish(p, score=0, part_a=1)
    rows = leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake)
    assert [r["id"] for r in rows] == ["idea-native"]
    cache = json.loads((p.results / "_results_cache.json").read_text())
    before_scope = objective_scope(p.cfg)
    p.cfg["report"]["dataset_keys"] = ["part_b"]
    assert objective_scope(p.cfg) != before_scope
    assert _oracle(p)[2:4] == (None, "metric_coverage_below_min:0/1")
    assert leaderboard.update_report(p.results, p.ideas, p.cfg, lake=p.lake) == []
    after = json.loads((p.results / "_results_cache.json").read_text())
    assert cache["idea-native"]["col_hash"] != after["idea-native"]["col_hash"]
    assert _payload(p.results)["top"] == []
