import yaml

from orze.reporting.leaderboard import _analyze_config_diversity, update_report


def test_diversity_reuses_unchanged_completed_configs(tmp_path, monkeypatch):
    ids = ["idea-a", "idea-b"]
    for idea_id, rank in zip(ids, (4, 8)):
        path = tmp_path / idea_id / "resolved_config.yaml"
        path.parent.mkdir()
        path.write_text(f"lora_rank: {rank}\n", encoding="utf-8")

    loads = 0
    real_safe_load = yaml.safe_load

    def counting_safe_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return real_safe_load(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", counting_safe_load)
    first = _analyze_config_diversity(tmp_path, ids)
    second = _analyze_config_diversity(tmp_path, list(reversed(ids)))

    assert first == second
    assert loads == 2


def test_diversity_invalidates_when_completed_config_changes(tmp_path, monkeypatch):
    ids = ["idea-a", "idea-b"]
    for idea_id, rank in zip(ids, (4, 8)):
        path = tmp_path / idea_id / "resolved_config.yaml"
        path.parent.mkdir()
        path.write_text(f"lora_rank: {rank}\n", encoding="utf-8")

    loads = 0
    real_safe_load = yaml.safe_load

    def counting_safe_load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return real_safe_load(*args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", counting_safe_load)
    before = _analyze_config_diversity(tmp_path, ids)
    (tmp_path / "idea-b" / "resolved_config.yaml").write_text(
        "lora_rank: 16\nextra_dimension: changed\n", encoding="utf-8"
    )
    after = _analyze_config_diversity(tmp_path, ids)

    assert before != after
    assert loads == 4


def test_unchanged_report_artifacts_are_not_rewritten(tmp_path):
    idea_dir = tmp_path / "idea-a"
    idea_dir.mkdir()
    metrics = idea_dir / "metrics.json"
    metrics.write_text(
        '{"status":"COMPLETED","score":1.0}', encoding="utf-8"
    )
    ideas = {"idea-a": {"title": "A", "priority": "high"}}
    cfg = {
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score", "label": "Score"}],
        }
    }

    update_report(tmp_path, ideas, cfg)
    outputs = [
        tmp_path / "report.md",
        tmp_path / "_leaderboard.json",
        tmp_path / "_leaderboard_views.json",
    ]
    mtimes = {path: path.stat().st_mtime_ns for path in outputs}
    update_report(tmp_path, ideas, cfg)

    assert {path: path.stat().st_mtime_ns for path in outputs} == mtimes

    metrics.write_text(
        '{"status":"COMPLETED","score":2.0}', encoding="utf-8"
    )
    update_report(tmp_path, ideas, cfg)
    assert (tmp_path / "report.md").stat().st_mtime_ns != mtimes[outputs[0]]
    assert "2.0" in (tmp_path / "report.md").read_text(encoding="utf-8")
