"""Legacy metadata repair through real ingress, actual CPU CLI and fresh replay."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prefix_product", ROOT / "docs/evidence/checks/2026-09-15-sidecar-prefix-product.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sha = helper.sha


def main(project):
    from orze.core.proposal_admission import admit_proposal
    from orze.engine import idea_ingress
    from orze.idea_lake import IdeaLake
    with patch.dict(sys.modules, {"orze_pro": None}):
        from orze.engine.orchestrator import Orze
    project.mkdir(parents=True, exist_ok=False)
    before_sources = helper.fingerprint()
    source = project / "ideas.md"
    source.write_text("# Ideas\n")
    cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
           "results_dir": str(project / "results"), "ideas_file": str(source),
           "idea_lake_db": str(project / "lake.db"), "min_disk_gb": 0,
           "notifications": {"enabled": False},
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .01}}
    (project / "orze.yaml").write_text(yaml.safe_dump(cfg))
    invocations = [helper.invoke(project, "initialize")]
    action = {"version": 1, "adapter": "command", "purpose": "verify native admission after legacy identity preparation",
              "inputs": {}, "command": [sys.executable, "-c",
                 "import json; from pathlib import Path; Path('answer.json').write_text(json.dumps(sum([2,3,5])))"],
              "timeout_seconds": 4, "outputs": {"answer": {"path": "answer.json", "max_bytes": 128}}}
    tail = yaml.safe_dump({"kind": "native_cpu_action", "action": action})
    engine = Orze.__new__(Orze)
    engine.cfg, engine.results_dir, engine.active_roles = cfg, Path(cfg["results_dir"]), {}
    engine.failure_counts = {}
    engine.lake = IdeaLake(cfg["idea_lake_db"])
    cfg["_orze_dir"] = str(project / ".orze")
    try:
        for index in range(1500):
            historical = copy.deepcopy(action)
            historical["inputs"] = {"fixture_seed": index}
            config = yaml.safe_dump({"kind": "native_cpu_action", "action": historical})
            engine.lake.conn.execute(
                "INSERT INTO ideas(idea_id,title,config,raw_markdown,status,kind) VALUES (?,?,?,'','completed','native_cpu_action')",
                (f"legacy-{index:04d}", "Imported completed metadata only", config),
            )
        engine.lake.conn.commit()
        missing_before = engine.lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL").fetchone()[0]
        candidate = engine.lake.prepare_proposal("idea-tail", title="Authored CPU action", config_yaml=tail,
                                                raw_markdown="", kind="native_cpu_action")
        unprepared = admit_proposal(engine.lake, candidate)
        assert unprepared["status"] == "rejected" and unprepared["reason"] == "proposal_dedup_capacity"
        assert engine.lake.get("idea-tail") is None
        source.write_text("# Ideas\n" + "\n".join(
            f"## {key}: Authored CPU action\n```yaml\n{tail}```\n"
            for key in ("idea-tail", "idea-tail-duplicate")))
        raw, inserted = idea_ingress.ingest_ideas_source(engine, cfg)
        assert list(raw) == ["idea-tail", "idea-tail-duplicate"] and inserted == ["idea-tail"]
        assert engine.lake.get("idea-tail-duplicate") is None
        missing_after = engine.lake.conn.execute("SELECT COUNT(*) FROM ideas WHERE config_hash IS NULL OR config_source_sha256 IS NULL").fetchone()[0]
        assert missing_before == 1500 and missing_after == 0
    finally:
        engine.lake.close()
    invocations.append(helper.invoke(project, "execute"))
    before = helper.helper.rows(project / "lake.db")
    assert len(before["ideas"]) == 1501
    attempt, = before["execution_attempts"]
    reservation, = before["cpu_action_reservations"]
    artifact, = before["research_artifacts"]
    terminal = json.loads(attempt["terminal_json"])
    assert attempt["task_id"] == "idea-tail" and attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
    assert terminal["process_tree"]["event"] == "TREE_CLOSED" and terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
    assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(attempt["terminal_json"].encode())
    record = json.loads(artifact["record_json"])
    assert json.loads(Path(record["path"]).read_text()) == 10
    assert sha(Path(record["path"]).read_bytes()) == record["content_sha256"]
    authored = sha(source.read_bytes())
    invocations.append(helper.invoke(project, "replay"))
    after = helper.helper.rows(project / "lake.db")
    assert before == after and sha(source.read_bytes()) == authored
    assert before_sources == helper.fingerprint()
    report = {"passed": True, "project": str(project), "script_sha256": sha(Path(__file__).read_bytes()),
              "helper_sha256": sha(Path(helper.__file__).read_bytes()), "source_files": before_sources,
              "invocations": invocations, "unprepared": unprepared, "missing_before": missing_before,
              "missing_after": missing_after, "ingress": {"ids": list(raw), "inserted": inserted},
              "source_sha256_after_execution": authored, "before": before, "after": after,
              "limits": ["1500 imported completed metadata rows, not historical executed work.",
                         "One actual native CPU worker, three fresh Core CLI invocations, no provider or GPU.",
                         "Real ingress called on a real Lake; legacy preparation preserves current writer dedup authority.",
                         "No production, scientific speedup, or real-model acceptance."]}
    (project / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "legacy_missing": missing_before, "native_cpu_workers": 1, "fresh_cli_invocations": 3}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
