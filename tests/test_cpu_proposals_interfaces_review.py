"""Independent proposal boundaries using real published CPU artifact sources.

The Policy mutation and post-COMMIT peer scheduling are controlled seams;
artifact publication, captured-source verification, ordinary admission, request
storage, the peer's claim and the final committed readers remain real. The peer
only claims: this file does not execute a proposed action or use a GPU/provider.
"""
import copy
import json

from orze.core import cpu_proposal_requests as store
from orze.core import research_interfaces as api
from orze.core.evaluation_retry_state import open_existing_lake
from orze.engine import cpu_proposals as coordinator
from orze.engine.cpu_policy_evidence import recorded_evidence
from orze.engine.scheduler import claim
from test_cpu_action_sources import published
from test_cpu_domain_product import request


def _configuration(lake, results):
    return {"execution": {"version": 1, "resource": "cpu", "slots": 2,
                          "wall_budget_seconds": 20},
            "results_dir": str(results), "idea_lake_db": str(lake.db_path),
            "action_domain": {"version": 1, "kind": "command", "config": {}}}


def _decision(records):
    return {"kind": "Propose", "request_id": "review-request", "task_id": "review-task",
            "reason": "check the actually captured source",
            "domain_request": request("pass", sources=[r["artifact_id"] for r in records])}


def test_policy_callback_cannot_replace_real_captured_artifact_metadata(published, monkeypatch):
    lake, results, records, handles = published
    cfg = _configuration(lake, results)
    snapshot = {"queue": [], "active": False, "now": 1,
                "recorded_evidence": recorded_evidence(lake, results)}
    original = copy.deepcopy(snapshot)
    selected = _decision(records)
    callback_views = []
    monkeypatch.setattr(api, "_POLICIES", dict(api._POLICIES))

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, view, budget):
            callback_views.append(view)
            first = view["recorded_evidence"]["results"][0]
            first["artifact_records"][0]["content_sha256"] = "0" * 64
            first["artifact_records"][0]["producer"]["generation"] += 1
            budget["remaining"] = 1000000
            return copy.deepcopy(selected)

    api.register_policy("proposal_review_mutator", "proposal.review.v1", Policy)
    cfg["action_policy"] = {"version": 1, "kind": "proposal_review_mutator",
                            "idle": "wait", "wait_seconds": 1}
    budget_view = {"remaining": 2}
    decision = api.BoundPolicy(api.capture_interfaces(cfg)).decide(snapshot, budget_view)
    expected = api.proposal_sources(snapshot, decision)
    assert callback_views[0] != original
    assert snapshot == original
    assert budget_view == {"remaining": 2}
    assert list(expected) == records
    outcome = coordinator.propose(lake, results, cfg, decision, expected_sources=expected)
    durable = store.get_request(lake.conn, str(results), decision["request_id"])
    assert outcome["status"] == "inserted"
    assert [item["artifact"] for item in durable["source_snapshot"]["inputs"]] == records
    assert recorded_evidence(lake, results) == original["recorded_evidence"]
    assert lake.conn.execute("SELECT count(*) FROM execution_attempts").fetchone()[0] == 2
    assert not (results / decision["task_id"]).exists()


def test_legitimate_peer_claim_after_commit_is_not_a_failed_proposal(published, monkeypatch):
    lake, results, records, handles = published
    cfg = _configuration(lake, results)
    decision = _decision(records)
    actual = lake.conn
    commits, claims = [], []
    reservations = actual.execute("SELECT * FROM cpu_action_reservations ORDER BY rowid").fetchall()

    class ClaimAfterCommit:
        def __getattr__(self, name):
            return getattr(actual, name)

        def commit(self):
            actual.commit()
            commits.append("actual_commit")
            peer = open_existing_lake(lake.db_path)
            try:
                assert peer.conn is not actual
                claims.append(claim(decision["task_id"], results, None, peer, resource="cpu"))
            finally:
                peer.close()

    # Capture retains exact lake.conn identity, so install the transparent
    # scheduling seam before coordinator capture, never during verification.
    with monkeypatch.context() as patch:
        patch.setattr(lake, "conn", ClaimAfterCommit())
        outcome = coordinator.propose(lake, results, cfg, decision, expected_sources=records)
    assert commits == ["actual_commit"]
    assert claims == [True]
    assert outcome["status"] == "inserted"
    assert actual.execute("SELECT current_state,sop_type FROM idea_state WHERE idea_id=?",
                          (decision["task_id"],)).fetchone()[:] == ("CLAIMED", "action")
    claim_before = (results / decision["task_id"] / "claim.json").read_bytes()
    assert json.loads(claim_before)["resource"] == "cpu"
    assert store.get_request(actual, str(results), decision["request_id"])["outcome"] == outcome
    replay = coordinator.propose(lake, results, cfg, decision, expected_sources=records)
    assert replay == outcome
    assert (results / decision["task_id"] / "claim.json").read_bytes() == claim_before
    assert actual.execute("SELECT count(*) FROM cpu_proposal_requests").fetchone()[0] == 1
    assert actual.execute("SELECT count(*) FROM execution_attempts").fetchone()[0] == 2
    assert actual.execute("SELECT * FROM cpu_action_reservations ORDER BY rowid").fetchall() == reservations
