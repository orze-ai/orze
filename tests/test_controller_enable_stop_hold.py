"""Candidate cross-entry regression: partial stop request is not re-enable proof."""
from pathlib import Path

import orze.cli as cli
from orze.core.control_outcome import ControllerStopHOLD
from orze.core import fs
from orze.lifecycle import do_stop


def test_enable_cannot_erase_the_only_latch_after_partial_stop_request(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    cfg = {"results_dir": str(results)}
    original = fs.atomic_write
    writes = []

    def fail_second_marker(path, *args, **kwargs):
        writes.append(Path(path).name)
        if Path(path).name == ".orze_stop_all":
            raise OSError("controlled second-marker publication failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fs, "atomic_write", fail_second_marker)
    result = do_stop(cfg, timeout=0)
    assert result.status == "hold"
    assert writes == [".orze_disabled", ".orze_stop_all"]
    latch = results / ".orze_disabled"
    before = latch.read_bytes()
    assert not (results / ".orze_stop_all").exists()
    assert not (results / ".orze_shutdown").exists()

    monkeypatch.setattr(fs, "atomic_write", original)
    monkeypatch.setattr("sys.argv", ["orze", "--enable"])
    monkeypatch.setattr("orze.extensions._find_pro_key", lambda: "present")
    monkeypatch.setattr(cli, "load_project_config", lambda path: cfg)
    monkeypatch.setattr(cli, "_require_controller_runtime", lambda config: None)
    try:
        outcome = cli.main()
    except ControllerStopHOLD:
        outcome = 75

    assert latch.exists(), "unconfirmed stop lost its sole persistent latch"
    assert latch.read_bytes() == before
    assert outcome == 75
