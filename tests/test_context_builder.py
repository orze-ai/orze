import json
import os

import orze.research.context_builder as context_builder
from orze.research.context_builder import build_digest


def _write_idea(results, idea_id, metrics, *, config=None, source=None,
                mtime=None):
    idea_dir = results / idea_id
    idea_dir.mkdir(parents=True)
    metrics_path = idea_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    if config is not None:
        import yaml
        (idea_dir / "idea_config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8")
    if source is not None:
        (idea_dir / "evaluation.json").write_text(
            json.dumps(source), encoding="utf-8")
    if mtime is not None:
        os.utime(metrics_path, (mtime, mtime))
    return idea_dir


def _source_cfg(results):
    return {
        "_project_root": str(results.parent),
        "results_dir": str(results),
        "report": {
            "primary_metric": "avg",
            "sort": "ascending",
            "min_datasets": 2,
            "columns": [
                {"key": "avg", "source": "evaluation.json:avg"},
                {"key": "wer_a", "source": "evaluation.json:wer_a"},
                {"key": "wer_b", "source": "evaluation.json:wer_b"},
            ],
        },
    }


def test_digest_ranks_exact_qualified_source_not_raw_metric(tmp_path):
    results = tmp_path / "results"
    _write_idea(
        results,
        "idea-qualified",
        {"status": "COMPLETED", "avg": -999.0},
        source={"avg": 2.0, "wer_a": 1.0, "wer_b": 3.0},
    )
    _write_idea(
        results,
        "idea-incomplete",
        {"status": "COMPLETED", "avg": -1000.0},
        source={"avg": 1.0, "wer_a": 1.0},
    )

    digest = build_digest(results, _source_cfg(results))

    assert "idea-qualified" in digest
    assert "avg=2.0" in digest
    assert "idea-incomplete" not in digest
    assert "qualification: 1 accepted, 1 rejected" in digest
    assert "leaderboard_rank_comparable: false" in digest
    assert "identity is not cryptographically proven" in digest


def test_generic_valid_half_score_is_not_hardcoded_as_sentinel(tmp_path):
    results = tmp_path / "results"
    _write_idea(
        results,
        "idea-valid-half",
        {"status": "COMPLETED", "score": 0.5},
    )
    cfg = {
        "results_dir": str(results),
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }

    digest = build_digest(results, cfg)

    assert "idea-valid-half" in digest
    assert "score=0.5" in digest


def test_cached_historic_leader_is_requalified_beyond_recent_window(tmp_path):
    results = tmp_path / "results"
    old = _write_idea(
        results,
        "idea-historic-best",
        {"status": "COMPLETED", "score": -1.0},
        mtime=1,
    )
    assert old.is_dir()
    for index in range(200):
        _write_idea(
            results,
            f"idea-recent-{index:03d}",
            {"status": "COMPLETED", "score": float(index + 10)},
            mtime=index + 10,
        )
    (results / "_leaderboard.json").write_text(json.dumps({
        "metric": "score",
        # The cached value is deliberately false. Only the ID may be trusted;
        # the current artifact must supply the value used by the digest.
        "top": [{"idea_id": "idea-historic-best", "metric_value": -9999}],
    }), encoding="utf-8")
    cfg = {
        "results_dir": str(results),
        "report": {
            "primary_metric": "score",
            "sort": "ascending",
            "columns": [{"key": "score"}],
        },
    }

    digest = build_digest(results, cfg, top_n=1)

    assert "idea-historic-best" in digest
    assert "score=-1.0" in digest
    assert "-9999" not in digest
    assert "1 independently re-qualified cached leaders" in digest


def test_recent_bound_is_applied_before_artifact_contents_are_loaded(
        tmp_path, monkeypatch):
    results = tmp_path / "results"
    for index in range(205):
        _write_idea(
            results,
            f"idea-candidate-{index:03d}",
            {"status": "COMPLETED", "score": float(index)},
            mtime=index + 1,
        )
    original = context_builder._load_metrics
    loaded = []

    def counted(idea_dir):
        loaded.append(idea_dir.name)
        return original(idea_dir)

    monkeypatch.setattr(context_builder, "_load_metrics", counted)
    context_builder._collect_recent(results)

    assert len(loaded) == context_builder.RECENT_RESULT_LIMIT
    assert "idea-candidate-000" not in loaded
    assert "idea-candidate-204" in loaded


def test_redirected_metric_artifact_cannot_enter_digest(tmp_path):
    results = tmp_path / "results"
    idea_dir = results / "idea-redirected"
    idea_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(
        json.dumps({"status": "COMPLETED", "score": 100.0}),
        encoding="utf-8",
    )
    (idea_dir / "metrics.json").symlink_to(outside)
    (results / "_leaderboard.json").write_text(json.dumps({
        "metric": "score",
        "top": [{"idea_id": "idea-redirected", "metric_value": 100.0}],
    }), encoding="utf-8")
    cfg = {
        "results_dir": str(results),
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }

    digest = build_digest(results, cfg)

    assert "idea-redirected" not in digest
    assert "## Top-10 evidence-qualified local candidates by score\n  none" in digest


def test_digest_does_not_export_error_or_config_values(tmp_path):
    results = tmp_path / "results"
    secret = "do-not-export-private-value"
    _write_idea(
        results,
        "idea-private-config",
        {"status": "COMPLETED", "score": 1.0},
        config={"model": f"https://user:{secret}@private.example/model"},
    )
    _write_idea(
        results,
        "idea-private-error",
        {"status": "FAILED", "error": f"crash token={secret}"},
    )
    cfg = {
        "results_dir": str(results),
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }

    digest = build_digest(results, cfg)

    assert secret not in digest
    assert "private.example" not in digest
    assert "token=" not in digest
    assert "[model-configured]" in digest
    assert "idea-private-error" in digest
