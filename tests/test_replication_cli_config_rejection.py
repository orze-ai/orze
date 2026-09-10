"""Malformed new CLI inputs reject before opening persistent authority.

Actual argparse, configuration bytes and loader are exercised. The established
CLI fixture stubs only the new control service/external runtime boundaries;
the opener tripwire proves invalid configuration cannot reach a database.
These are new B3 draft entry-contract regressions, not old-release bugs.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from orze import cli
from orze.core import evaluation_retry_state
from test_replication_cli_boundary import project as cli_project, _args


@pytest.mark.parametrize("raw", [
    "- unexpected-project-item\n",
    "broken: [\n",
    "false\n",
    "[]\n",
], ids=["nonempty-list", "invalid-yaml", "false", "empty-list"])
def test_invalid_project_mapping_is_one_json_rejection_before_authority(
        cli_project, monkeypatch, capsys, raw):
    p = cli_project
    p.config.write_text(raw, encoding="utf-8")
    database = p.db.read_bytes()
    files = set(p.root.rglob("*"))
    opener = Mock(side_effect=AssertionError("invalid config must not open authority"))
    monkeypatch.setattr(evaluation_retry_state, "open_existing_lake", opener)
    _args(p, monkeypatch)
    code = cli.main()
    output = json.loads(capsys.readouterr().out)
    assert type(code) is int and code == 2
    assert set(output) == {"error"}
    assert isinstance(output["error"], str) and output["error"]
    assert Path.cwd() == p.outside
    assert p.db.read_bytes() == database
    assert set(p.root.rglob("*")) == files
    opener.assert_not_called()
    p.service.assert_not_called()
