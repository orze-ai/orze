"""Campaign inputs keep their declared types through the actual Lake consumer."""
from examples.research_comparison.scheduling_campaign import _admit
from orze.core.integrity import hash_config
from orze.core.research_interfaces import parse_domain_task
from orze.idea_lake import IdeaLake


def test_campaign_admission_preserves_numeric_and_literal_input_identity(tmp_path):
    inputs = {"small": 1e-5, "large": 1e20, "literal": "1e-05",
              "integer": 1, "fraction": 1.0, "flag": False, "text_flag": "false"}
    request = {"version": 1, "purpose": "Check typed inputs", "inputs": inputs,
               "timeout_seconds": 5, "input_artifact_ids": [], "payload": {},
               "outputs": {"evaluation": {"path": "evaluation.json", "max_bytes": 1024}}}
    run = {"root": str(tmp_path), "snapshots": [], "admissions": []}
    _admit(run, "typed-inputs", "idea-typed-inputs", request)
    lake = IdeaLake(str(tmp_path / "lake.db"))
    try:
        row = lake.get("idea-typed-inputs")
    finally:
        lake.close()
    actual = parse_domain_task(row["config"])
    assert actual == request
    assert {key: type(value) for key, value in actual["inputs"].items()} == {
        key: type(value) for key, value in inputs.items()}
    assert row["config_hash"] == hash_config({"kind": "native_cpu_action", "domain_request": request})
    assert run["admissions"][0]["raw_config"] == row["config"]
