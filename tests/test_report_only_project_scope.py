"""CLI draft boundaries: preserve runtime control-root semantics and reject lists.

Four real report calls cover nested/external results and default/custom DBs.
The only report wrapper invokes the real benchmark control-path guard, records
the received config, then delegates to the real update_report implementation.
Baseline is the saved I2 draft, NOT the previously published CLI behavior.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.core.benchmark_contract import benchmark_exposure_ledger_path
from orze.idea_lake import IdeaLake
from orze.reporting import report_cli


@pytest.fixture
def safe_caller(tmp_path, monkeypatch):
    outside = tmp_path / "caller"
    outside.mkdir()
    monkeypatch.chdir(outside)
    for owner, name in (
        (extensions, "_find_pro_key"), (extensions, "get_extension"),
        (extensions, "_auto_install_pro"), (cli, "maybe_star"),
        (cli, "detect_all_gpus"), (subprocess, "Popen"), (subprocess, "run"),
    ):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError(
            f"report-only must not call {name}")))
    return outside


@pytest.mark.parametrize("results_kind", ["nested", "external"])
@pytest.mark.parametrize("database_kind", ["default", "custom-relative"])
def test_report_cli_preserves_runtime_control_root_for_nonstandard_results(
    tmp_path, monkeypatch, safe_caller, results_kind, database_kind,
):
    root = tmp_path / "project"
    root.mkdir()
    results = (
        root / "output" / "results" if results_kind == "nested"
        else tmp_path / "external" / "results")
    folder = results / "idea-control-root"
    folder.mkdir(parents=True)
    control_root = results.parent / ".orze"
    database = (
        control_root / "idea_lake.db" if database_kind == "default"
        else root / "state" / "authority.sqlite3")
    database.parent.mkdir(parents=True, exist_ok=True)
    lake = IdeaLake(database)
    try:
        lake.insert(folder.name, "Control root preserved", "seed: 4", "",
                    status="completed")
    finally:
        lake.close()
    before = database.read_bytes()
    (folder / "metrics.json").write_text(
        '{"status":"COMPLETED","score":900}', encoding="utf-8")
    (folder / "assessment.json").write_text('{"score":0}', encoding="utf-8")
    cfg = {
        "results_dir": "output/results" if results_kind == "nested" else str(results),
        "auto_seal_eval": False,
        "report": {
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "assessment.json:score"}],
        },
    }
    if database_kind == "custom-relative":
        cfg["idea_lake_db"] = "state/authority.sqlite3"
    config = root / "orze.yaml"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    seen = []
    real_update_report = report_cli.update_report

    def checked_report(results_dir, ideas, full_cfg, **kwargs):
        # Real guard: the draft's rewritten _project_root raises the existing
        # benchmark_exposure_control_directory_drift error here.
        ledger = benchmark_exposure_ledger_path(full_cfg)
        seen.append((dict(full_cfg), ledger, kwargs["lake"].db_path))
        return real_update_report(results_dir, ideas, full_cfg, **kwargs)

    monkeypatch.setattr(report_cli, "update_report", checked_report)
    monkeypatch.setattr(sys, "argv", [
        "orze", "--report-only", "-c", str(config),
    ])

    assert cli.main() == 0

    assert Path.cwd() == safe_caller
    assert len(seen) == 1
    received, ledger, observed_db = seen[0]
    assert Path(received["_project_root"]) == results.parent
    assert Path(received["_orze_dir"]) == control_root
    assert Path(received["_env_ORZE_DIR"]) == control_root
    assert Path(received["ideas_file"]) == control_root / "ideas.md"
    assert Path(received["idea_lake_db"]) == database
    assert Path(observed_db) == database
    assert ledger == control_root / "_benchmark_exposures.jsonl"
    assert database.read_bytes() == before
    payload = json.loads((results / "_leaderboard.json").read_text(encoding="utf-8"))
    assert [(row["idea_id"], row["metric_value"]) for row in payload["top"]] == [
        (folder.name, 0),
    ]
    assert not list(safe_caller.iterdir())


def test_report_cli_rejects_nonmapping_yaml_before_global_loader(
    tmp_path, monkeypatch, safe_caller, capsys,
):
    root = tmp_path / "project"
    root.mkdir()
    config = root / "orze.yaml"
    config.write_text("- valid YAML but not a project mapping\n", encoding="utf-8")
    before = set(root.rglob("*"))
    monkeypatch.setattr(sys, "argv", [
        "orze", "--report-only", "-c", str(config),
    ])

    assert cli.main() == 2

    assert "report_config_not_mapping" in capsys.readouterr().err
    assert Path.cwd() == safe_caller
    assert set(root.rglob("*")) == before
    assert not list(safe_caller.iterdir())
