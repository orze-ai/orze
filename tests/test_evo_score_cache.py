from types import SimpleNamespace

from orze.engine.phases import _log_evo_score_if_changed
from orze.reporting import search_path


def test_evo_score_rebuilds_only_when_lake_inputs_change(tmp_path, monkeypatch):
    lake = tmp_path / "idea_lake.db"
    lake.write_bytes(b"db-v1")
    calls = []

    def fake_build(path, cfg):
        calls.append((path, cfg))
        return {"research_efficiency": {"score": 50.0, "grade": "C"}}

    monkeypatch.setattr(search_path, "build_from_lake", fake_build)
    owner = SimpleNamespace()
    cfg = {"report": {"sort": "ascending"}}

    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert _log_evo_score_if_changed(owner, lake, cfg) is False
    assert len(calls) == 1

    # WAL-only commits and report changes must both invalidate the cache.
    (tmp_path / "idea_lake.db-wal").write_bytes(b"commit")
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    cfg["report"]["sort"] = "descending"
    assert _log_evo_score_if_changed(owner, lake, cfg) is True
    assert len(calls) == 3
