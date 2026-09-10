"""Two actual draft handoff races: hashes must bind the checked identities."""
from pathlib import Path

from orze.core.execution_attempts import current_attempt
from orze.core.research_observations import observations_for_attempt
from orze.engine import evaluator, observation_publication
from orze.engine.sealed import compute_sealed_hashes, write_sealed_manifest
from orze.engine.termination_hold import TerminationUnconfirmed
from test_observation_snapshot_contract import (
    project, artifact_project, native_case, _launch, _observations, _envelope,
)


def test_input_changed_after_hash_cannot_pair_old_digest_with_new_metadata(project, monkeypatch):
    c = project
    target = Path(c.input_records[0]["path"])
    original = observation_publication.files._snapshot_hash
    seen = []
    def hash_then_change(path, maximum):
        value = original(path, maximum)
        if Path(path) == target and not seen:
            seen.append(True)
            original_bytes = target.read_bytes()
            target.chmod(0o600)
            target.write_bytes(b"Z" * len(original_bytes))
            target.chmod(0o400)
        return value
    monkeypatch.setattr(observation_publication.files, "_snapshot_hash", hash_then_change)
    result = evaluator.launch_eval(c.idea, 0, c.results, c.cfg, lake=c.lake,
                                    source_event=c.source_event)
    assert seen == [True]
    assert result is None and c.eval_calls == [], "old input digest must not authorize changed bytes"
    assert current_attempt(c.lake.conn, c.idea, "evaluation") is None


def test_sealed_change_after_result_preparation_cannot_cross_terminal_writer(project, monkeypatch):
    c = project
    script = Path(c.cfg["eval_script"])
    c.cfg["sealed_files"] = [str(script)]
    write_sealed_manifest(c.results, compute_sealed_hashes(c.cfg["sealed_files"]))
    ep, _, _, output = _launch(c)
    output.write_bytes(_envelope(_observations()[:1]))
    ep.process.returncode = 0
    original = observation_publication.prepare_observations
    seen = []
    def prepare_then_change(*args, **kwargs):
        prepared = original(*args, **kwargs)
        seen.append(True)
        script.write_text("# changed after accepted sealed hash, before SQL writer\n")
        return prepared
    monkeypatch.setattr(observation_publication, "prepare_observations", prepare_then_change)
    active = {0: ep}
    try:
        events = evaluator.check_active_evals(active, c.results, c.cfg, lake=c.lake)
    except TerminationUnconfirmed:
        events = []
    assert seen == [True]
    assert events == [] and active.get(0) is ep
    assert observations_for_attempt(c.lake.conn, ep.attempt_ref) == []
    assert current_attempt(c.lake.conn, c.idea, "evaluation")["state"] == "RUNNING"
