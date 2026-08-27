import os
import sys
from pathlib import Path

import pytest

import orze.cli as cli
import orze.extensions as extensions
from orze.core.benchmark_contract import EXPOSURE_LEDGER_FILE


@pytest.mark.parametrize("mode", ["--full", "--scratch"])
def test_reset_preserves_project_benchmark_exposure_history(
        tmp_path, monkeypatch, mode):
    orze_dir = tmp_path / ".orze"
    results_dir = tmp_path / "results"
    orze_dir.mkdir()
    results_dir.mkdir()
    ledger = orze_dir / EXPOSURE_LEDGER_FILE
    ledger.write_bytes(b'{"sealed":"history"}\n')
    (orze_dir / "disposable-state").write_text("remove me", encoding="utf-8")
    if mode == "--scratch":
        (orze_dir / "idea_lake.db").write_bytes(b"idea-history")

    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(orze_dir),
        "results_dir": str(results_dir),
        "idea_lake_db": str(orze_dir / "idea_lake.db"),
    }
    monkeypatch.setattr(cli, "load_project_config", lambda _: cfg)
    monkeypatch.setattr(extensions, "_find_pro_key", lambda: "test-key")
    monkeypatch.setattr(
        sys, "argv", ["orze", "reset", mode, "--yes", "--force"],
    )

    assert cli.main() == 0
    assert ledger.read_bytes() == b'{"sealed":"history"}\n'
    assert not (orze_dir / "disposable-state").exists()
    assert os.stat(ledger).st_mode & 0o777 == 0o600
    if mode == "--scratch":
        assert (orze_dir / "idea_lake.db").read_bytes() == b"idea-history"


def test_reset_rejects_symlinked_exposure_history(tmp_path, monkeypatch):
    orze_dir = tmp_path / ".orze"
    results_dir = tmp_path / "results"
    orze_dir.mkdir()
    results_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("do not touch", encoding="utf-8")
    (orze_dir / EXPOSURE_LEDGER_FILE).symlink_to(outside)
    cfg = {
        "_project_root": str(tmp_path),
        "_orze_dir": str(orze_dir),
        "results_dir": str(results_dir),
    }
    monkeypatch.setattr(cli, "load_project_config", lambda _: cfg)
    monkeypatch.setattr(extensions, "_find_pro_key", lambda: "test-key")
    monkeypatch.setattr(
        sys, "argv", ["orze", "reset", "--full", "--yes", "--force"],
    )

    assert cli.main() == 2
    assert outside.read_text(encoding="utf-8") == "do not touch"
