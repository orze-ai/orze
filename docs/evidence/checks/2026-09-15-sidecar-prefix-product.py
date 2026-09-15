"""Real paged ingress API followed by actual CPU CLI execution and fresh replay.

Historical completed rows are metadata fixtures. Native CPU actions at the end
of each source traversal are real; no model, GPU, existing service or deployment.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("prior_product", ROOT / "docs/evidence/checks/2026-09-15-ingress-product.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sha = helper.sha


def fingerprint():
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (ROOT / "src").rglob("*.py")}


def invoke(project, name):
    command = [sys.executable, "-c", "import sys; sys.modules['orze_pro']=None; from orze.cli import main; raise SystemExit(main())",
               "-c", str(project / "orze.yaml"), "--once"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    with (project / (name + ".stdout.log")).open("xb") as stdout, (project / (name + ".stderr.log")).open("xb") as stderr:
        done = subprocess.run(command, cwd=project, env=env, stdout=stdout, stderr=stderr, timeout=60)
    assert done.returncode == 0
    return {"command": command, "exit_code": done.returncode}


def main(root):
    from orze.engine import idea_ingress
    from orze.idea_lake import IdeaLake
    with patch.dict(sys.modules, {"orze_pro": None}):
        from orze.engine.orchestrator import Orze
    root.mkdir(parents=True, exist_ok=False)
    before_sources = fingerprint()
    report = {"source_files": before_sources, "script_sha256": sha(Path(__file__).read_bytes()),
              "helper_sha256": sha(Path(helper.__file__).read_bytes()), "variants": [],
              "limits": ["299 imported completed metadata rows per project; not executed historical research.",
                         "Three direct calls to real ingress on a real Lake, followed by actual Core CLI CPU execution.",
                         "One bounded metadata prefix layout and one partially traversed single-file negative control.",
                         "Fresh local test projects only; no provider or production acceptance."]}
    for layout in ("many", "one"):
        project = root / layout
        project.mkdir()
        source = project / "ideas.md"
        source.write_text("# Ideas\n")
        cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
               "results_dir": str(project / "results"), "ideas_file": str(source),
               "idea_lake_db": str(project / "lake.db"), "min_disk_gb": 0,
               "notifications": {"enabled": False},
               "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .01}}
        (project / "orze.yaml").write_text(yaml.safe_dump(cfg))
        invocations = [invoke(project, "initialize")]
        side = project / "ideas.d"
        side.mkdir()
        engine = Orze.__new__(Orze)
        engine.cfg, engine.results_dir, engine.active_roles = cfg, Path(cfg["results_dir"]), {}
        engine.failure_counts = {}
        engine.lake = IdeaLake(cfg["idea_lake_db"])
        cfg["_orze_dir"] = str(project / ".orze")
        action = {"version": 1, "adapter": "command", "purpose": "verify a sidecar action beyond two inspection pages",
                  "inputs": {}, "command": [sys.executable, "-c",
                    "import json; from pathlib import Path; Path('answer.json').write_text(json.dumps(sorted([5,1,3])))"],
                  "timeout_seconds": 4, "outputs": {"answer": {"path": "answer.json", "max_bytes": 128}}}
        blocks = []
        try:
            for index in range(299):
                key = f"idea-history-{index:04d}"
                engine.lake.insert(key, "Stored metadata", f"seed: {index}\n", "", status="completed")
                blocks.append(f"## {key}: Conflicting authored title\n```yaml\nseed: {index}\n```\n")
            for key in ("idea-tail", "idea-tail-duplicate"):
                blocks.append(f"## {key}: Authored CPU action\n```yaml\n" + yaml.safe_dump(
                    {"kind": "native_cpu_action", "action": action}) + "```\n")
            if layout == "many":
                for index, block in enumerate(blocks):
                    (side / f"{index:04d}.md").write_text(block)
            else:
                (side / "all.md").write_text("\n".join(blocks))
            authored = {str(p.relative_to(project)): sha(p.read_bytes()) for p in [source, *side.glob("*.md")]}
            trace = []
            for _ in range(3):
                raw, inserted = idea_ingress.ingest_ideas_source(engine, cfg)
                trace.append({"ids": list(raw), "inserted": inserted, "offset": engine._idea_ingress_cursor[2],
                              "prefix_files": len(engine._idea_sidecar_prefix.files)})
            assert [r["inserted"] for r in trace] == [[], [], ["idea-tail"]]
            assert [len(r["ids"]) for r in trace] == [128, 128, 45]
            assert trace[-1]["offset"] == 0
            assert engine.lake.get("idea-tail-duplicate") is None
        finally:
            engine.lake.close()
        (project / "ingress.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        invocations.append(invoke(project, "execute"))
        before = helper.rows(project / "lake.db")
        assert len(before["ideas"]) == 300
        attempt, = before["execution_attempts"]
        reservation, = before["cpu_action_reservations"]
        artifact, = before["research_artifacts"]
        terminal = json.loads(attempt["terminal_json"])
        assert attempt["task_id"] == "idea-tail" and attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
        assert terminal["process_tree"]["event"] == "TREE_CLOSED" and terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
        assert reservation["state"] == "SETTLED" and reservation["terminal_sha256"] == sha(attempt["terminal_json"].encode())
        record = json.loads(artifact["record_json"])
        assert json.loads(Path(record["path"]).read_text()) == [1, 3, 5]
        assert sha(Path(record["path"]).read_bytes()) == record["content_sha256"]
        invocations.append(invoke(project, "replay"))
        after = helper.rows(project / "lake.db")
        assert before == after
        assert authored == {p: sha((project / p).read_bytes()) for p in authored}
        report["variants"].append({"project": str(project), "layout": layout, "source_files": authored,
            "invocations": invocations, "ingress": trace, "before": before, "after": after})
        (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert before_sources == fingerprint()
    report["passed"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "native_cpu_workers": 2, "fresh_cli_invocations": 6}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
