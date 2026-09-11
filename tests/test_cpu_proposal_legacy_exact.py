"""Ordinary legacy exact-ID admission is not upgraded to new FSM history."""
from orze.core import cpu_proposal_requests as store
from test_cpu_proposals import catalog, decision, submit


def test_legacy_exact_source_without_transition_stays_exact(catalog):
    lake, results, cfg = catalog
    selected = decision()
    raw = store.canonical({"kind": "native_cpu_action", "domain_request": selected["domain_request"]})
    lake.insert(selected["task_id"], selected["domain_request"]["purpose"], raw, "",
        status="queued", kind="native_cpu_action", hypothesis=selected["reason"])
    assert lake.conn.execute("SELECT count(*) FROM idea_transitions").fetchone()[0] == 0
    before = tuple(lake.conn.execute("SELECT * FROM idea_state").fetchone())
    outcome = submit(catalog, selected)
    assert outcome["status"] == "already_present_exact"
    assert tuple(lake.conn.execute("SELECT * FROM idea_state").fetchone()) == before
    assert lake.conn.execute("SELECT count(*) FROM idea_transitions").fetchone()[0] == 0
    record = store.get_request(lake.conn, str(results), selected["request_id"])
    assert record["admission_evidence"]["transition_id"] is None
    assert record["admission_evidence"]["transition_sha256"] is None
