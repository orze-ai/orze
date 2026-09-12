"""Real Domain interpretation and observation registration expiry windows."""
import json
import time

import pytest

from test_native_cpu_domain_publication import domain_context, launch, finish
from orze.core import cpu_action_budget as budget
from orze.core import research_observations as observations
from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt
from orze.engine import cpu_domain_publication, native_cpu_action as native


def expire(lease):
    before = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    assert before < lease["deadline_ns"]
    while time.clock_gettime_ns(time.CLOCK_BOOTTIME) <= lease["deadline_ns"]:
        time.sleep(.005)
    print(json.dumps({"actual_expiry": {"before_ns": before,
        "after_ns": time.clock_gettime_ns(time.CLOCK_BOOTTIME), "lease": lease}}))


@pytest.mark.parametrize("boundary", ["interpret", "register"])
def test_real_domain_observation_expiry_preserves_prepared_effect_boundary(domain_context, monkeypatch, boundary):
    lake, results, scope, cfg, create, _ = domain_context
    cfg["cpu_runtime_lease"] = {"version": 1, "ttl_seconds": 1}
    measurement = {"version": 1, "observations": [{"name": "zero", "values": {"value": 0},
        "validation": {"status": "valid", "reason_code": "reported"}, "comparison_scope": "test-v1"}]}
    run, permit = create("from pathlib import Path; Path('result.json').write_text(" + repr(json.dumps(measurement)) + ")")
    handle = launch(domain_context, run, permit)
    lease = current_attempt(lake.conn, "idea-domain", "action")["binding"]["runtime_lease"]
    seen = []
    if boundary == "interpret":
        original = cpu_domain_publication.prepare

        def delayed(*args, **kwargs):
            result = original(*args, **kwargs)
            seen.append(result)
            expire(lease)
            return result

        monkeypatch.setattr(cpu_domain_publication, "prepare", delayed)
        terminal = finish(handle, results, cfg, lake, permit)
        assert terminal["outcome"] == "interrupted"
        assert terminal["reason_code"] == "cpu_runtime_lease_expired"
        assert terminal["process_tree"]["stop_requested"] is False
        assert terminal["artifact_ids"] == terminal["observation_ids"] == []
        assert budget.snapshot(lake, scope)["active_reservations"] == 0
    else:
        original = observations.register_observations

        def delayed(conn, ref, records):
            result = original(conn, ref, records)
            seen.extend(observations.observations_for_attempt(conn, ref))
            expire(lease)
            return result

        monkeypatch.setattr(observations, "register_observations", delayed)
        with pytest.raises(native.CPUActionHOLD):
            finish(handle, results, cfg, lake, permit)
        row = current_attempt(lake.conn, "idea-domain", "action")
        assert row["state"] == "RUNNING" and row["terminal"] is None
        folder = results / "idea-domain"
        assert (folder / "_attempt_effect.lock").is_dir()
        effect = folder / "_execution_effects" / handle.attempt_id
        assert (effect / "prepared.json").is_file() and not (effect / "committed.json").exists()
        assert seen[0]["values"] == {"value": 0}
        assert budget.snapshot(lake, scope)["active_reservations"] == 1
    assert len(seen) == 1
    assert observations.observations_for_attempt(lake.conn, handle.attempt_ref) == []
    assert artifacts_for_attempt(lake.conn, handle.attempt_ref) == ()
