"""The report must preserve lifecycle and metric-basis truth."""

import json

from orze.reporting.leaderboard import update_report
from orze.idea_lake import IdeaLake


class _Lake:
    def __init__(self):
        self.records = {
            "idea-complete": {"title": "done", "status": "completed"},
            "idea-failed": {"title": "failed", "status": "failed"},
            "idea-running": {"title": "running", "status": "running"},
            "idea-queued": {"title": "queued", "status": "queued"},
            "idea-skipped": {"title": "skipped", "status": "skipped"},
            "idea-archived": {"title": "archived", "status": "archived"},
        }

    def get_all_ids(self, status=None):
        if status is None:
            return set(self.records)
        return {
            idea_id for idea_id, record in self.records.items()
            if record["status"] == status
        }

    def get(self, idea_id):
        raise AssertionError("report must use one metadata-index query")

    def get_metadata_index(self):
        return {
            idea_id: {"idea_id": idea_id, **record}
            for idea_id, record in self.records.items()
        }


def test_pipeline_status_uses_authoritative_lake_counts(tmp_path):
    cfg = {
        "report": {
            "title": "Test",
            "primary_metric": "score",
            "sort": "descending",
            "columns": [],
        }
    }

    update_report(tmp_path, {}, cfg, lake=_Lake())

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| 4 | 1 | 1 | 1 | 1 |" in report
    assert "## Queue (1 ideas)" in report
    assert "idea-queued" in report
    assert "idea-skipped" not in report.split("## Queue", 1)[1]


def test_native_report_counts_ignore_legacy_status_divergence(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-complete", "done", "{}", "", status="completed")
    lake.insert("idea-failed", "failed", "{}", "", status="failed")
    lake.conn.execute(
        "UPDATE ideas SET status = 'queued' WHERE idea_id = 'idea-complete'"
    )
    lake.conn.execute(
        "UPDATE ideas SET status = 'completed' WHERE idea_id = 'idea-failed'"
    )
    lake.conn.commit()
    cfg = {
        "report": {
            "title": "Audited",
            "primary_metric": "score",
            "sort": "descending",
            "columns": [],
        }
    }

    update_report(tmp_path, {}, cfg, lake=lake)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| 2 | 1 | 1 | 0 | 0 |" in report
    lake.close()


def test_min_datasets_counts_dataset_wer_not_aggregate_or_time(tmp_path):
    keys = [f"wer_dataset_{i}" for i in range(8)]
    columns = [
        {"key": "avg_wer", "label": "Avg"},
        *({"key": key, "label": key} for key in keys),
        {"key": "training_time", "label": "Time"},
    ]
    cfg = {
        "report": {
            "title": "Eight-set WER",
            "primary_metric": "avg_wer",
            "sort": "ascending",
            "min_datasets": 8,
            "columns": columns,
        }
    }
    ideas = {
        "idea-partial": {"title": "partial"},
        "idea-complete": {"title": "complete"},
        "idea-nonfinite": {"title": "nonfinite"},
    }
    partial = {
        "status": "COMPLETED", "avg_wer": 1.0, "training_time": 100,
        **{key: 1.0 for key in keys[:7]},
    }
    complete = {
        "status": "COMPLETED", "avg_wer": 2.0, "training_time": 100,
        **{key: 2.0 for key in keys},
    }
    nonfinite = {
        "status": "COMPLETED", "avg_wer": 0.0, "training_time": 100,
        **{key: (True if i % 2 else float("inf"))
           for i, key in enumerate(keys)},
    }
    for idea_id, metrics in (("idea-partial", partial),
                             ("idea-complete", complete),
                             ("idea-nonfinite", nonfinite)):
        idea_dir = tmp_path / idea_id
        idea_dir.mkdir()
        (idea_dir / "metrics.json").write_text(json.dumps(metrics))

    completed = update_report(tmp_path, ideas, cfg)

    assert [row["id"] for row in completed] == ["idea-complete"]
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "idea-complete" in report
    ranked = report.split("## Unranked:", 1)[0]
    assert "idea-partial" not in ranked
    assert "idea-nonfinite" not in ranked
    assert "idea-partial" in report
    assert "metric_coverage_below_min:7/8" in report


def test_audited_lifecycle_controls_ranking_candidacy(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    lake.insert("idea-db-failed", "failed", "{}", "", status="failed")
    lake.insert("idea-db-complete", "complete", "{}", "", status="completed")
    for idea_id, metrics in (
        ("idea-db-failed", {"status": "COMPLETED", "score": 100.0}),
        ("idea-db-complete", {"status": "FAILED", "score": 200.0}),
    ):
        idea_dir = tmp_path / idea_id
        idea_dir.mkdir()
        (idea_dir / "metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8")
    cfg = {
        "report": {
            "title": "Audited ranking",
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }

    completed = update_report(tmp_path, {}, cfg, lake=lake)
    leaderboard = json.loads(
        (tmp_path / "_leaderboard.json").read_text(encoding="utf-8"))
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert completed == []
    assert leaderboard["top"] == []
    assert leaderboard["evidence_qualification"]["accepted"] == 0
    assert leaderboard["evidence_qualification"]["rejected"] == {
        "local_metrics_not_completed": 1,
    }
    assert "| 2 | 1 | 1 | 0 | 0 |" in report
    assert "idea-db-failed" not in report.split("## Unranked:", 1)[0]
    assert "idea-db-complete" in report.split("## Unranked:", 1)[1]
    lake.close()


def test_report_cache_invalidates_on_lifecycle_transition(tmp_path):
    lake = IdeaLake(str(tmp_path / "ideas.db"))
    idea_id = "idea-transitioned"
    lake.insert(idea_id, "transitioned", "{}", "", status="failed")
    idea_dir = tmp_path / idea_id
    idea_dir.mkdir()
    (idea_dir / "metrics.json").write_text(
        json.dumps({"status": "COMPLETED", "score": 1.0}),
        encoding="utf-8",
    )
    cfg = {
        "report": {
            "primary_metric": "score",
            "sort": "descending",
            "columns": [{"key": "score"}],
        },
    }

    assert update_report(tmp_path, {}, cfg, lake=lake) == []
    for from_state, to_state in (
        ("FAILED", "QUEUED"),
        ("QUEUED", "CLAIMED"),
        ("CLAIMED", "IN_PROGRESS"),
        ("IN_PROGRESS", "COMPLETE"),
    ):
        assert lake.record_state_transition(
            idea_id, from_state, to_state, reason="test_transition") is True

    completed = update_report(tmp_path, {}, cfg, lake=lake)

    assert [row["id"] for row in completed] == [idea_id]
    lake.close()
