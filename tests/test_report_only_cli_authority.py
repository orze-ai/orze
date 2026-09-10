"""Real report-only CLI: project-scoped reads must not bootstrap authority.

Strict routing cases tripwire credential/paid/process/GPU boundaries. Separate
storage/path cases replace only the unrelated legacy star/key probe with a
test-only sentinel, so the old constructor's DB writes can be observed without
confusing an earlier routing failure with a database mutation regression.
No config loader, parser, report function, SQLite reader or IdeaLake is mocked.
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.idea_lake import IdeaLake


def _project(tmp_path, database_kind, *, populate=True, absolute_paths=False):
    root = tmp_path / "project"
    results = root / "results"
    results.mkdir(parents=True)
    outside = tmp_path / "caller"
    outside.mkdir()
    database = root / (
        ".orze/idea_lake.db" if database_kind == "default"
        else "state/authority.sqlite3")
    inbox = root / "empty-inbox.md"
    inbox.write_text("# Empty research inbox\n", encoding="utf-8")
    config = root / "orze.yaml"
    cfg = {
        "results_dir": str(results) if absolute_paths else "results",
        "ideas_file": str(inbox) if absolute_paths else "empty-inbox.md",
        "auto_seal_eval": False,
        "roles": {},
        "report": {
            "title": "Read-only report",
            "primary_metric": "score", "sort": "ascending",
            "columns": [{"key": "score", "source": "assessment.json:score"}],
        },
    }
    if database_kind != "default":
        cfg["idea_lake_db"] = (
            str(database) if absolute_paths else str(database.relative_to(root)))
    config.write_text(json.dumps(cfg), encoding="utf-8")
    if populate:
        database.parent.mkdir(parents=True, exist_ok=True)
        lake = IdeaLake(database)
        try:
            for idea_id, status in (
                ("idea-catalog-complete", "completed"),
                ("idea-catalog-queued", "queued"),
                ("idea-catalog-failed", "failed"),
                ("idea-catalog-archived", "archived"),
            ):
                lake.insert(idea_id, idea_id, "seed: 7", "Catalog, not hot inbox",
                            status=status)
        finally:
            lake.close()
        folder = results / "idea-catalog-complete"
        folder.mkdir()
        # Raw proxy must not replace a qualified source value of zero.
        (folder / "metrics.json").write_text(
            '{"status":"COMPLETED","score":900}', encoding="utf-8")
        (folder / "assessment.json").write_text('{"score":0}', encoding="utf-8")
    return SimpleNamespace(root=root, results=results, outside=outside,
                           database=database, config=config, cfg=cfg, inbox=inbox)


def _tripwires(monkeypatch, *, bypass_legacy_key_probe=False):
    boundaries = []
    for owner, name in (
        (cli, "maybe_star"), (cli, "detect_all_gpus"),
        (extensions, "get_extension"), (extensions, "_auto_install_pro"),
        (subprocess, "Popen"), (subprocess, "run"),
    ):
        guard = Mock(side_effect=AssertionError(
            f"report-only must not call {name}"))
        monkeypatch.setattr(owner, name, guard)
        boundaries.append(guard)
    if bypass_legacy_key_probe:
        # This explicit test-only bypass isolates storage/path behavior; it
        # neither reads a key nor grants production access to paid extensions.
        monkeypatch.setattr(extensions, "_find_pro_key", lambda: "test-only-probe-bypassed")
    else:
        guard = Mock(side_effect=AssertionError(
            "report-only must not call _find_pro_key"))
        monkeypatch.setattr(extensions, "_find_pro_key", guard)
        boundaries.append(guard)
    return boundaries


def _invoke(p, monkeypatch):
    monkeypatch.chdir(p.outside)
    monkeypatch.setattr(sys, "argv", [
        "orze", "--report-only", "-c", str(p.config),
    ])
    try:
        result = cli.main()
    except SystemExit as exc:
        if not isinstance(exc.code, int):
            raise
        result = exc.code
    assert Path.cwd() == p.outside
    return result


def _database_files(p):
    """Capture DB and sidecars without opening or repairing the database."""
    return {
        path.name: path.read_bytes()
        for path in p.database.parent.glob(p.database.name + "*")
        if path.is_file()
    }


def _assert_catalog_report(p):
    text = (p.results / "report.md").read_text(encoding="utf-8")
    # ARCHIVED is catalog history, not a fourth executable pipeline item.
    assert "| 3 | 1 | 1 | 0 | 1 |" in text
    assert "## Queue (1 ideas)" in text
    assert "idea-catalog-queued" in text.split("## Queue", 1)[1]
    payload = json.loads((p.results / "_leaderboard.json").read_text(encoding="utf-8"))
    assert [(row["idea_id"], row["metric_value"]) for row in payload["top"]] == [
        ("idea-catalog-complete", 0),
    ]
    assert payload["evidence_qualification"]["accepted"] == 1
    assert not list(p.outside.iterdir())


@pytest.mark.parametrize("database_kind", ["default", "custom"])
def test_report_only_does_not_probe_paid_or_compute_paths(
    tmp_path, monkeypatch, database_kind,
):
    p = _project(tmp_path, database_kind)
    before = _database_files(p)
    boundaries = _tripwires(monkeypatch)

    assert _invoke(p, monkeypatch) in (None, 0)

    _assert_catalog_report(p)
    assert _database_files(p) == before
    for boundary in boundaries:
        boundary.assert_not_called()


@pytest.mark.parametrize("database_kind", ["default", "custom"])
def test_report_only_resolves_relative_project_paths_outside_cwd(
    tmp_path, monkeypatch, database_kind,
):
    p = _project(tmp_path, database_kind)
    before = _database_files(p)
    boundaries = _tripwires(monkeypatch, bypass_legacy_key_probe=True)

    assert _invoke(p, monkeypatch) in (None, 0)

    _assert_catalog_report(p)
    assert _database_files(p) == before
    for boundary in boundaries:
        boundary.assert_not_called()


@pytest.mark.parametrize("database_kind", ["default", "custom"])
def test_report_only_missing_database_does_not_create_authority(
    tmp_path, monkeypatch, database_kind,
):
    p = _project(tmp_path, database_kind, populate=False, absolute_paths=True)
    folder = p.results / "idea-unregistered"
    folder.mkdir()
    (folder / "metrics.json").write_text(
        '{"status":"COMPLETED","score":0}', encoding="utf-8")
    (folder / "assessment.json").write_text('{"score":0}', encoding="utf-8")
    boundaries = _tripwires(monkeypatch, bypass_legacy_key_probe=True)
    before = _database_files(p)

    # Controlled refusal or an explicitly empty unavailable report are both
    # compatible; a new return code is not the old mutation behavior under test.
    assert _invoke(p, monkeypatch) in (None, 0, 2)

    assert not p.database.exists()
    assert not p.database.parent.exists()
    assert _database_files(p) == before
    payload_path = p.results / "_leaderboard.json"
    if payload_path.exists():
        assert json.loads(payload_path.read_text(encoding="utf-8"))["top"] == []
    assert not list(p.outside.iterdir())
    for boundary in boundaries:
        boundary.assert_not_called()


def test_report_only_incompatible_database_is_not_bootstrapped(tmp_path, monkeypatch):
    p = _project(tmp_path, "custom", populate=False, absolute_paths=True)
    p.database.parent.mkdir(parents=True)
    with sqlite3.connect(p.database) as connection:
        connection.execute("CREATE TABLE unrelated (payload TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('preserve-original-schema')")
        connection.commit()
    before = _database_files(p)
    boundaries = _tripwires(monkeypatch, bypass_legacy_key_probe=True)

    assert _invoke(p, monkeypatch) in (None, 0, 2)

    # Check bytes/sidecars before any verification connection can itself affect
    # journal policy. This is the substantive bootstrap regression assertion.
    assert _database_files(p) == before
    connection = sqlite3.connect(p.database.as_uri() + "?mode=ro", uri=True)
    try:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        ).fetchall() == [("unrelated",)]
        assert connection.execute("SELECT payload FROM unrelated").fetchall() == [
            ("preserve-original-schema",),
        ]
    finally:
        connection.close()
    payload_path = p.results / "_leaderboard.json"
    if payload_path.exists():
        assert json.loads(payload_path.read_text(encoding="utf-8"))["top"] == []
    assert not list(p.outside.iterdir())
    for boundary in boundaries:
        boundary.assert_not_called()
