"""B2 draft regression: later terminal SQL must not change bound inputs."""
from copy import deepcopy
import json

import pytest

from orze.core.execution_attempts import current_attempt
from orze.core.research_artifacts import artifacts_for_attempt, get_artifact
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator
from orze.engine.termination_hold import TerminationUnconfirmed
from test_observation_snapshot_contract import (
    project, artifact_project, native_case, _launch, _envelope, _observations,
)


def test_after_terminal_input_metadata_change_rolls_back_the_whole_acceptance(project):
    c = project
    ep, _, _, output = _launch(c)
    output.write_bytes(_envelope(_observations()))
    ep.process.returncode = 0
    before = c.input_records[0]
    changed = deepcopy(before)
    changed["content_sha256"] = "0" * 64
    replacement = json.dumps(changed, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).replace("'", "''")
    c.lake.conn.execute(
        "CREATE TRIGGER drift_input_after_terminal AFTER UPDATE ON execution_attempts "
        "WHEN NEW.phase='evaluation' AND NEW.state='TERMINAL' BEGIN "
        f"UPDATE research_artifacts SET record_json='{replacement}' "
        f"WHERE artifact_id='{before['artifact_id']}'; END")
    c.lake.conn.commit()
    active = {0: ep}

    with pytest.raises(TerminationUnconfirmed):
        evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)

    assert active.get(0) is ep
    assert current_attempt(c.lake.conn, c.idea, "evaluation")["state"] == "RUNNING"
    assert c.lake.get_fsm_state(c.idea) == "IN_PROGRESS"
    assert get_artifact(c.lake.conn, before["artifact_id"]) == before
    assert artifacts_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert (c.folder / "_attempt_effect.lock").is_dir()
    assert not (c.folder / "_execution_effects" / ep.attempt_id / "committed.json").exists()
