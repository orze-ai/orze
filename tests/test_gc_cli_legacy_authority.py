"""Real CLI-to-GC legacy path, preserving default versus explicit DB intent."""
import json

import pytest
import yaml

from orze.agents import orze_gc


@pytest.mark.parametrize("explicit_missing_database", [False, True])
def test_cli_distinguishes_absent_default_from_explicit_missing_authority(
        tmp_path, monkeypatch, capsys, explicit_missing_database):
    project = tmp_path / "project"
    folder = project / "results" / "idea-legacy-cli"
    checkpoint = project / "checkpoints" / folder.name
    folder.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (folder / "metrics.json").write_text('{"status":"COMPLETED"}')
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"selected legacy disposable checkpoint")
    config = {"results_dir": "results", "gc": {
        "checkpoints_dir": "checkpoints", "keep_top": 0, "keep_recent": 0}}
    if explicit_missing_database:
        config["idea_lake_db"] = "missing-authority.db"
    path = project / "orze.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    caller = tmp_path / "invocation"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr("sys.argv", ["orze-gc", "-c", str(path)])

    result = orze_gc.main()
    stats = json.loads(capsys.readouterr().out)

    if explicit_missing_database:
        assert result == 2 and stats["blocked"] is True
        assert weights.read_bytes() == b"selected legacy disposable checkpoint"
    else:
        assert result == 0, "an absent unconfigured default is not a failed explicit route"
        assert stats["checkpoints"]["deleted"] == 1
        assert not checkpoint.exists(), "real configured legacy cleanup must run"
    assert not list(project.rglob("*.db")), "cleanup must never bootstrap authority"
    assert (folder / "metrics.json").read_text() == '{"status":"COMPLETED"}'
