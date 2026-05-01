"""Round-2 D1: ``orze ideas inject`` writes a row.
Round-2 E1: marker-driven multi-host reset works in-process."""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_inject_then_query(tmp_path):
    from orze.admin.ideas_inject import inject_idea
    from orze.idea_lake import IdeaLake

    db = tmp_path / "lake.db"
    cfg = {"idea_lake_db": str(db),
           "report": {"primary_metric": "wer"}}
    rc = inject_idea(cfg, idea_id="idea-02e83b",
                     title="manual repro",
                     priority="high",
                     status="queued")
    assert rc == 0
    lake = IdeaLake(str(db))
    row = lake.get("idea-02e83b")
    assert row is not None
    assert row["title"] == "manual repro"
    assert row["status"] == "queued"


def test_reset_role_state_marker(tmp_path):
    from orze.admin.reset_role_state import (
        write_all_hosts_marker, consume_marker_on_this_host,
    )
    results = tmp_path / "results"
    results.mkdir()
    host = socket.gethostname()
    state = results / f".orze_state_{host}.json"
    state.write_text(json.dumps({
        "iteration": 5,
        "roles": {
            "professor": {"cooldown_override": 7200,
                          "consecutive_failures": 6},
            "engineer": {"cooldown_override": 0,
                         "consecutive_failures": 0},
        },
    }), encoding="utf-8")

    marker = write_all_hosts_marker(results, role=None)
    assert marker.exists()
    cleared = consume_marker_on_this_host(results)
    assert cleared == ["professor"], cleared
    # marker should self-delete (only one host pending)
    assert not marker.exists()
    s = json.loads(state.read_text(encoding="utf-8"))
    assert s["roles"]["professor"]["consecutive_failures"] == 0
    assert "cooldown_override" not in s["roles"]["professor"]


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_inject_then_query(Path(td))
        print("test_inject_then_query OK")
    with tempfile.TemporaryDirectory() as td:
        test_reset_role_state_marker(Path(td))
        print("test_reset_role_state_marker OK")
