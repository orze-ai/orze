"""New coordinator boundaries: actual SQLite, plus genuine native CPU sources.

No historical API-absence reds. SQL/commit faults are explicitly injected;
normal admission, source capture, store validation and fresh reads remain real.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from orze.core import cpu_proposal_requests as store
from orze.core import research_interfaces as api
from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine import cpu_proposals as proposals
from orze.idea_lake import IdeaLake
from test_cpu_action_sources import published
from test_cpu_domain_product import request


def decision(identity="idea-new", key="request-one", sources=None):
    return {"kind": "Propose", "request_id": key, "task_id": identity,
            "reason": "explicit next question", "domain_request": request("pass", sources=sources)}


def config(lake, results):
    return {"execution": {"version": 1, "resource": "cpu", "slots": 2, "wall_budget_seconds": 20},
            "results_dir": str(results), "idea_lake_db": str(lake.db_path),
            "action_domain": {"version": 1, "kind": "command", "config": {}}}


@pytest.fixture
def catalog(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    lake = IdeaLake(tmp_path / "lake.db")
    try:
        yield lake, results, config(lake, results)
    finally:
        lake.close()


def submit(catalog, selected=None, expected_sources=()):
    lake, results, cfg = catalog
    return proposals.propose(lake, results, cfg, selected or decision(), expected_sources=expected_sources)


class Connection:
    def __init__(self, real, commit):
        self.real, self.on_commit = real, commit

    def __getattr__(self, key):
        return getattr(self.real, key)

    def commit(self):
        self.on_commit(self.real)


def test_empty_view_has_no_schema_or_mutation(catalog):
    lake, results, cfg = catalog
    before = list(lake.conn.iterdump())
    assert proposals.recorded_proposals(lake, results) == {"results": [], "more_available": False}
    assert list(lake.conn.iterdump()) == before


def test_selected_domain_not_constructed_or_prepared_to_admit(catalog, monkeypatch):
    lake, results, cfg = catalog
    monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
    def forbidden(*args):
        raise AssertionError("proposal called Domain")
    api.register_domain("uninvoked", "uninvoked.v1", forbidden)
    cfg["action_domain"]["kind"] = "uninvoked"
    assert submit(catalog)["status"] == "inserted"
    assert lake.conn.execute("SELECT current_state,sop_type FROM idea_state").fetchall()[0][:] == ("QUEUED", "action")
    assert lake.conn.execute("SELECT name FROM sqlite_master WHERE name IN ('execution_attempts','cpu_action_reservations')").fetchall() == []
    assert not (results / "idea-new").exists()


def test_two_actual_connections_same_key_one_normal_admission(catalog):
    lake, results, cfg = catalog
    def run(_):
        peer = open_existing_lake(lake.db_path)
        try:
            return submit((peer, results, cfg))
        finally:
            peer.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(run, range(2)))
    assert replies[0] == replies[1]
    assert lake.conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 1
    assert lake.conn.execute("SELECT count(*) FROM cpu_proposal_requests").fetchone()[0] == 1
    assert lake.conn.execute("SELECT count(*) FROM idea_transitions").fetchone()[0] == 1


@pytest.mark.parametrize("changed", ["request", "domain"])
def test_same_key_conflict_does_not_rewrite(catalog, changed, monkeypatch):
    lake, results, cfg = catalog
    submit(catalog)
    before = list(lake.conn.iterdump())
    selected = decision()
    if changed == "request":
        selected["reason"] = "another question"
    else:
        monkeypatch.setattr(api, "_DOMAINS", dict(api._DOMAINS))
        factory = api._DOMAINS["command"][1]
        api._DOMAINS["command"] = ("changed.v2", factory)
    with pytest.raises(proposals.ProposalHOLD):
        submit(catalog, selected)
    assert list(lake.conn.iterdump()) == before


def test_caller_transaction_not_committed_or_rolled_back(catalog):
    lake, results, cfg = catalog
    lake.conn.execute("BEGIN IMMEDIATE")
    lake.conn.execute("UPDATE ideas SET title=title")
    with pytest.raises(proposals.ProposalHOLD, match="caller_transaction"):
        submit(catalog)
    assert lake.conn.in_transaction
    assert not lake.conn.execute("SELECT 1 FROM sqlite_master WHERE name='cpu_proposal_requests'").fetchall()
    lake.conn.rollback()


def test_actual_commit_rollback_never_acknowledged(catalog, monkeypatch):
    lake, results, cfg = catalog
    real = lake.conn
    monkeypatch.setattr(lake, "conn", Connection(real, lambda conn: conn.rollback()))
    with pytest.raises(proposals.ProposalHOLD):
        submit(catalog)
    assert real.execute("SELECT count(*) FROM ideas").fetchone()[0] == 0
    assert real.execute("SELECT name FROM sqlite_master WHERE name='cpu_proposal_requests'").fetchall() == []
    assert not real.in_transaction


def test_committed_request_corruption_is_seen_by_fresh_reader(catalog, monkeypatch):
    lake, results, cfg = catalog
    def commit(real):
        real.commit()
        real.execute("UPDATE cpu_proposal_requests SET record_json='{}'")
        real.commit()
    monkeypatch.setattr(lake, "conn", Connection(lake.conn, commit))
    with pytest.raises(proposals.ProposalHOLD):
        submit(catalog)
    assert lake.conn.execute("SELECT status FROM ideas").fetchone()[0] == "queued"
    assert lake.conn.execute("SELECT record_json FROM cpu_proposal_requests").fetchone()[0] == "{}"


def test_unknown_fresh_read_preserves_committed_intent_for_exact_replay(catalog, monkeypatch):
    lake, results, cfg = catalog
    real_connect = sqlite3.connect
    def connect(path, *args, **kwargs):
        if str(path).endswith("?mode=ro"):
            raise OSError("explicit fresh-read IO failure")
        return real_connect(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        with pytest.raises(proposals.ProposalHOLD):
            submit(catalog)
    assert lake.conn.execute("SELECT count(*) FROM cpu_proposal_requests").fetchone()[0] == 1
    before = list(lake.conn.iterdump())
    assert submit(catalog)["status"] == "inserted"
    assert list(lake.conn.iterdump()) == before


def test_after_request_trigger_cannot_modify_new_task_source(catalog):
    lake, results, cfg = catalog
    lake.conn.execute("BEGIN IMMEDIATE")
    store.ensure_schema(lake.conn)
    lake.conn.execute("CREATE TRIGGER change_target AFTER INSERT ON cpu_proposal_requests BEGIN "
                     "UPDATE ideas SET config=config||' ' WHERE idea_id=NEW.task_id; END")
    lake.conn.commit()
    with pytest.raises(proposals.ProposalHOLD):
        submit(catalog)
    assert lake.conn.execute("SELECT count(*) FROM ideas").fetchone()[0] == 0
    assert lake.conn.execute("SELECT count(*) FROM cpu_proposal_requests").fetchone()[0] == 0
    assert lake.conn.execute("SELECT count(*) FROM idea_transitions").fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["producer", "inode", "sql_after_insert"])
def test_real_confirmed_source_cannot_drift_from_selected_view(published, monkeypatch, fault):
    lake, results, records, handles = published
    selected = decision(sources=[record["artifact_id"] for record in records])
    expected = copy.deepcopy(records)
    cfg = config(lake, results)
    if fault == "producer":
        expected[0]["producer"]["generation"] += 1
    elif fault == "inode":
        original = proposals.admit_proposal_in_tx
        def admit(*args):
            from pathlib import Path
            path = Path(records[0]["path"])
            replacement = path.with_name("replacement")
            replacement.write_bytes(path.read_bytes())
            path.rename(path.with_name("prior-content"))
            replacement.rename(path)
            return original(*args)
        monkeypatch.setattr(proposals, "admit_proposal_in_tx", admit)
    else:
        lake.conn.execute("BEGIN IMMEDIATE")
        store.ensure_schema(lake.conn)
        lake.conn.execute("CREATE TRIGGER change_source AFTER INSERT ON cpu_proposal_requests BEGIN "
            "UPDATE research_artifacts SET record_json='{}'; END")
        lake.conn.commit()
    with pytest.raises(proposals.ProposalHOLD):
        submit((lake, results, cfg), selected, expected)
    assert lake.conn.execute("SELECT 1 FROM ideas WHERE idea_id='idea-new'").fetchall() == []
    assert lake.conn.execute("SELECT count(*) FROM execution_attempts").fetchone()[0] == 2
    assert lake.conn.execute("SELECT state FROM cpu_action_reservations").fetchall()[0][0] == "SETTLED"
    assert proposals.recorded_proposals(lake, results)["results"] == []
