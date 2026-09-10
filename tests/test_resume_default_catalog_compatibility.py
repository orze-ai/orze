"""Real loader/CLI compatibility; default-path equality is not provenance."""
from pathlib import Path
import sys

import pytest
import yaml

from orze import cli, extensions
from orze.core import config
from test_resume import resume_case, _write_valid_receipt
from test_stale_evaluation_completion import _files


@pytest.mark.parametrize("explicit", [False, True], ids=["implicit_missing_default", "explicit_same_missing_path"])
def test_real_resume_cli_distinguishes_default_from_explicit_missing_catalog(resume_case, monkeypatch, capsys, explicit):
    project, results, folder, checkpoint, raw_cfg, _ = resume_case
    _write_valid_receipt(resume_case)
    raw = {key: value for key, value in raw_cfg.items() if not key.startswith("_")}
    # Test both directions: users cannot manufacture or suppress the loader's
    # provenance by pre-filling an internal field in otherwise valid YAML.
    raw["_idea_lake_db_defaulted"] = explicit
    raw["roles"] = {"unused": {"enabled": False, "mode": "script", "script": "unused.py"}}
    default_db = project / ".orze" / "idea_lake.db"
    if explicit:
        raw["idea_lake_db"] = str(default_db)
    path = project / "orze.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.chdir(project)
    # The old resume CLI still has a license/star prelude. Bypass only that
    # unrelated external read rather than count it as this catalog regression.
    monkeypatch.setattr(extensions, "_find_pro_key", lambda: "test-only-no-credential-read")
    monkeypatch.setattr(config, "_load_dotenv", lambda *args: None)

    def forbidden(*args, **kwargs):
        pytest.fail("resume admission must not launch a process/provider")

    monkeypatch.setattr(cli, "maybe_star", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    loaded = config.load_project_config(str(path))
    assert Path(loaded["idea_lake_db"]) == default_db
    assert not default_db.exists()
    before = _files(folder)
    monkeypatch.setattr(sys, "argv", ["orze", "resume", folder.name,
                        "--resume-from", str(checkpoint), "-c", str(path)])

    result = cli.main()

    assert result == (2 if explicit else 0), "a loader-generated missing default is not an explicit unavailable catalog"
    assert not default_db.exists() and not default_db.parent.exists()
    if explicit:
        assert _files(folder) == before
        assert "resume_catalog_unavailable" in capsys.readouterr().out
    else:
        assert (folder / "resume_request.json").is_file()
        assert "Resume admitted" in capsys.readouterr().out
