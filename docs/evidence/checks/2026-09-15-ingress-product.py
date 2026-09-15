"""Two fresh CLI invocations, real sidecar CPU action, then exact replay.

Private temporary project only; no provider, GPU, existing service or package
installation. Duplicate sidecar remains authored input, never a second worker.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[3]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rows(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return {name: [dict(r) for r in conn.execute("SELECT * FROM " + name)]
                for name in ("ideas", "execution_attempts", "cpu_action_reservations", "research_artifacts")}


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    side = root / "ideas.d"
    side.mkdir()
    (root / "ideas.md").write_text("# Ideas\n")
    action = {"version": 1, "adapter": "command", "purpose": "verify sidecar ingress CPU execution",
              "inputs": {}, "command": [sys.executable, "-c",
                  "import json; from pathlib import Path; Path('answer.json').write_text(json.dumps(sorted([5,1,3])))"],
              "timeout_seconds": 4, "outputs": {"answer": {"path": "answer.json", "max_bytes": 128}}}
    source = "".join(f"## idea-{name}: Sidecar CPU {name}\n```yaml\n" + yaml.safe_dump(
        {"kind": "native_cpu_action", "action": action}) + "```\n" for name in ("first", "duplicate"))
    (side / "actions.md").write_text(source)
    cfg = {"execution": {"version": 2, "resource": "cpu", "slots": 1, "wall_budget_seconds": None},
           "results_dir": str(root / "results"), "ideas_file": str(root / "ideas.md"),
           "idea_lake_db": str(root / "lake.db"), "min_disk_gb": 0,
           "action_policy": {"version": 1, "kind": "queue", "idle": "wait", "wait_seconds": .01}}
    config = root / "orze.yaml"
    config.write_text(yaml.safe_dump(cfg))
    before = {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in (REPO / "src").rglob("*.py")}
    output = {"source_sha256": sha(source.encode()), "source_files": before, "invocations": []}
    for index in (1, 2):
        command = [sys.executable, "-c",
                   "import sys; sys.modules['orze_pro']=None; from orze.cli import main; raise SystemExit(main())",
                   "-c", str(config), "--once"]
        env = dict(os.environ, PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        with (root / f"cli-{index}.stdout.log").open("wb") as stdout, (root / f"cli-{index}.stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr, timeout=30)
        assert result.returncode == 0, (root / f"cli-{index}.stderr.log").read_text()
        snapshot = rows(root / "lake.db")
        output["invocations"].append({"command": command, "exit_code": result.returncode, "database": snapshot})
        (root / "report.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
        assert result.returncode == 0
        assert len(snapshot["ideas"]) == len(snapshot["execution_attempts"]) == len(snapshot["cpu_action_reservations"]) == 1
        attempt, = snapshot["execution_attempts"]
        reservation, = snapshot["cpu_action_reservations"]
        terminal = json.loads(attempt["terminal_json"])
        assert attempt["state"] == "TERMINAL" and terminal["outcome"] == "completed"
        assert terminal["process_tree"]["event"] == "TREE_CLOSED"
        assert terminal["process_tree"]["wait_proof"] == "ECHILD_WALL"
        assert reservation["state"] == "SETTLED"
        assert reservation["terminal_sha256"] == sha(attempt["terminal_json"].encode())
        artifact, = snapshot["research_artifacts"]
        record = json.loads(artifact["record_json"])
        assert json.loads(Path(record["path"]).read_text()) == [1, 3, 5]
        assert (side / "actions.md").read_text() == source
        assert (root / "ideas.md").read_text() == "# Ideas\n"
    assert output["invocations"][0]["database"] == output["invocations"][1]["database"]
    assert before == {str(p.relative_to(REPO)): sha(p.read_bytes()) for p in (REPO / "src").rglob("*.py")}
    output["passed"] = True
    (root / "report.json").write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"passed": True, "actual_workers": 1, "fresh_cli_invocations": 2, "root": str(root)}))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
