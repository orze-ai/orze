"""Isolated, explicitly invoked cross-channel product acceptance probe.

Not part of default tests/. Reuses the unchanged real CLI/native fixture:
one actual source worker per parameter, plus 33 metadata-only prefix entries.
Neither proposal pages nor metadata evidence hash the source artifact bytes.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_cpu_product_loop import project
from test_cpu_evidence_paging_product import (
    ANALYSIS, SOURCE, history, _capture, _database, _propose, _register,
)


@pytest.mark.parametrize("fault", ["unchanged", "effect_changed", "content_changed"])
def test_proposal_selection_rechecks_selected_evidence_without_consuming_scan(history, fault):
    history["cfg"]["action_policy"]["proposal_page_size"] = 8
    source = history["source"]
    ref = copy.deepcopy(source["producer"])
    state = {"fault": fault, "stage": 0, "mutation": None}
    before = _database(history["root"])

    class Policy:
        def __init__(self, declaration):
            pass

        def decide(self, snapshot, budget):
            _capture(history, snapshot, budget)
            stage = state["stage"]
            state["stage"] += 1
            assert state["stage"] <= 3
            page = snapshot["evidence_page"]
            view = snapshot["recorded_evidence"]
            if stage == 0:
                assert page["mode"] == "scan" and page["page"] == 1
                assert isinstance(page["next_cursor"], str)
                assert all(row["ref"] != ref for row in view["results"])
                state["scan"] = {k: copy.deepcopy(page[k]) for k in
                                 ("scan_id", "next_cursor", "page", "seen", "traversal_end")}
                decision = {"kind": "SelectEvidence", "refs": [copy.deepcopy(ref)]}
            elif stage == 1:
                assert page["mode"] == "selection"
                assert {k: page[k] for k in state["scan"]} == state["scan"]
                assert [row["ref"] for row in view["results"]] == [ref]
                assert view["results"][0]["artifact_records"] == [source]
                if fault != "unchanged":
                    path = (history["root"] / "results" / SOURCE / "_execution_effects" /
                            ref["attempt_id"] / "committed.json"
                            if fault == "effect_changed" else Path(source["path"]))
                    original = path.read_bytes()
                    (history["root"] / ("channel-original-" + path.name)).write_bytes(original)
                    replacement = b"not a committed receipt\n" if fault == "effect_changed" else b"19"
                    path.write_bytes(replacement)
                    state["mutation"] = {"path": str(path),
                        "original_sha256": hashlib.sha256(original).hexdigest(),
                        "replacement_sha256": hashlib.sha256(replacement).hexdigest()}
                decision = {"kind": "SelectProposals", "request_ids": ["absent-channel-request"]}
            else:
                assert page["mode"] == "selection"
                assert {k: page[k] for k in state["scan"]} == state["scan"]
                assert snapshot["proposal_page"]["mode"] == "selection"
                assert snapshot["proposal_page"]["missing_request_ids"] == ["absent-channel-request"]
                assert snapshot["recorded_proposals"]["results"] == []
                if fault == "effect_changed":
                    assert view["results"] == []
                    assert [row["ref"] for row in view["unavailable"]] == [ref]
                    assert view["unavailable"][0]["reason"]
                else:
                    assert [row["ref"] for row in view["results"]] == [ref]
                    assert view["results"][0]["artifact_records"] == [source]
                    assert view["unavailable"] == []
                decision = ({"kind": "Stop", "reason": "unchanged_cross_channel_evidence", "wakeup": None}
                            if fault == "unchanged" else _propose(source["artifact_id"]))
            history["trace"][-1]["decision"] = copy.deepcopy(decision)
            return decision

    _register(history, Policy)
    try:
        code = history["run"](once=False)
        history["outcomes"].append(code)
        assert code == (0 if fault == "unchanged" else 75)
        assert state["stage"] == 3
        after = _database(history["root"])
        assert after["execution_attempts"] == before["execution_attempts"]
        assert after["cpu_action_reservations"] == before["cpu_action_reservations"]
        assert [row["state"] for row in after["cpu_action_reservations"]] == ["SETTLED"]
        assert after["research_artifacts"] == before["research_artifacts"]
        assert after["research_observations"] == []
        assert after["cpu_proposal_requests"] == []
        assert len(history["processes"]) == 1
        assert not (history["root"] / "results" / ANALYSIS / "claim.json").exists()
        if fault == "unchanged":
            assert len(after["cpu_action_decisions"]) == len(before["cpu_action_decisions"]) + 1
            assert json.loads(after["cpu_action_decisions"][-1]["record_json"])["reason"] == "unchanged_cross_channel_evidence"
        else:
            assert after["cpu_action_decisions"] == before["cpu_action_decisions"]
        for entry in history["trace"]:
            assert entry["database_before_callback"]["cpu_action_reservations"] == before["cpu_action_reservations"]
            assert entry["database_before_callback"]["cpu_action_decisions"] == before["cpu_action_decisions"]
    finally:
        report = {"probe": state, "source": source, "source_ref": ref,
                  "actual_initial_source_cli_invocations": 1,
                  "followup_cli_codes": history["outcomes"],
                  "actual_native_workers": len(history["processes"]),
                  "metadata_only_prefixes": len(history["prefix_refs"]),
                  "before": before, "after": _database(history["root"]),
                  "trace": history["trace"],
                  "limits": ["Artifact content is NOT hashed by the metadata evidence page.",
                             "Content mutation may remain visible, but original Propose/capture_sources must reject.",
                             "Changed confirmed-effect bytes must become unavailable on channel switch.",
                             "Actual CLI invocations occur in the same pytest interpreter.",
                             "No extra worker, permission, budget reset, or source adoption."]}
        path = history["root"] / "proposal-channel-review.json"
        path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print("CPU_PROPOSAL_CHANNEL_REPORT=" + str(path))

