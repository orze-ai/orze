"""Read-only fixed-epoch V1-06F evidence verification; never runs workers."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import sys

if not __debug__:
    raise SystemExit("Evidence verification requires assertions; do not use -O.")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--worktree-only", action="store_true")
args = parser.parse_args()
root = Path.cwd()
repos = {"core": root, "pro": root.parent / "orze-pro"}
path = root / "docs/evidence/2026-09-11-v1-06f-terminal-recovery.json"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def eq(left, right):
    assert encoded(left) == encoded(right), (left, right)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(repo, *words):
    return subprocess.check_output(["git", "-C", str(repos[repo]), *words])


def manifest(result):
    eq(result["exit_code"], 0)
    lines = result["output"].splitlines()
    values = {}
    for line in lines:
        assert re.fullmatch("[0-9a-f]{64}", line[:64]) and line[64:66] == "  "
        assert line[66:] not in values
        values[line[66:]] = line[:64]
    return values


def footer(chunks, passed, skipped=0, failed=0):
    assert chunks
    eq(chunks[-1]["exit_code"], int(bool(failed)))
    assert all("exit_code" not in item for item in chunks[:-1])
    output = "".join(item["output"] for item in chunks)
    assert not re.search(r"(?i)output truncated|warning:[^\n]*truncat", output)
    tail = output.strip().splitlines()[-1]
    prefix = str(failed) + " failed" if failed else str(passed) + " passed"
    if skipped:
        prefix += ", " + str(skipped) + " skipped"
    assert re.fullmatch(prefix + r"(?:, \d+ warnings?)? in [0-9.]+s(?: \([^\n]+\))?", tail), tail
    return output


e = json.loads(path.read_bytes())
PINNED_COMMANDS = json.loads("{\"baseline\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_cpu_terminal_settlement_recovery.py --tb=short\",\"candidate\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_cpu_terminal_settlement_recovery.py tests/test_cpu_terminal_recovery_guards.py tests/test_cpu_terminal_recovery_barrier.py tests/test_cpu_terminal_recovery_controls.py --tb=short\",\"domain_first\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_cpu_terminal_recovery_domain.py --tb=short\",\"target\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests:. CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_cpu_terminal_settlement_recovery.py tests/test_cpu_terminal_recovery_guards.py tests/test_cpu_terminal_recovery_barrier.py tests/test_cpu_terminal_recovery_controls.py tests/test_cpu_terminal_recovery_domain.py --tb=short\",\"core\":\"PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python3 -m pytest -q tests --tb=short -rs --basetemp=/tmp/orze-v106f-core-full.208t7H\",\"pro\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../orze/src CUDA_VISIBLE_DEVICES= python3 -c 'from unittest.mock import patch; import pytest; p=patch(\\\"orze_pro._gate.require_license\\\"); p.start(); raise SystemExit(pytest.main([\\\"-q\\\", \\\"tests\\\", \\\"--tb=short\\\"]))'\",\"paired\":\"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../orze-pro/src CUDA_VISIBLE_DEVICES= python3 -c 'from unittest.mock import patch; import pytest; p=patch(\\\"orze_pro._gate.require_license\\\"); p.start(); raise SystemExit(pytest.main([\\\"-q\\\",\\\"tests/test_evolution.py\\\",\\\"tests/test_sweeper.py\\\",\\\"tests/test_skill_registry.py\\\",\\\"tests/test_fsm_runner_registry.py\\\",\\\"tests/test_skill_composition_manifest.py\\\",\\\"--tb=short\\\",\\\"-rs\\\"]))'\"}")
PINNED_CANDIDATE_SNAPSHOTS = json.loads("[{\"source\":\"src/orze/core/cpu_action_budget.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-src-orze-core-cpu-action-budget.py\",\"read_chunk\":\"3d86b6\",\"sha256\":\"b862ba337cbde23b89016fc4a9f0cc0a28dafd192817df8d9d4b5ff4d1c2cf14\"},{\"source\":\"src/orze/engine/cpu_phase.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-src-orze-engine-cpu-phase.py\",\"read_chunk\":\"5f09ca\",\"sha256\":\"f27f9f096d429c276dbcc7e24b1fdb6e2f849d46b0a3a7e55a712fe383dcddf8\"},{\"source\":\"tests/cpu_terminal_recovery_helpers.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-tests-cpu-terminal-recovery-helpers.py\",\"read_chunk\":\"3a1262\",\"sha256\":\"6222c9d58bee79fcfafcefd8d634101b1d2f690267dd48e1fcecad2cc37d020b\"},{\"source\":\"tests/test_cpu_terminal_settlement_recovery.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-tests-test-cpu-terminal-settlement-recovery.py\",\"read_chunk\":\"09568e\",\"sha256\":\"406a62346e0877a45cdbfbc9b7f7285bc47afe37c5d9dfe879934ced29172c33\"},{\"source\":\"tests/test_cpu_terminal_recovery_guards.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-tests-test-cpu-terminal-recovery-guards.py\",\"read_chunk\":\"e1ad65\",\"sha256\":\"324ac530f37b82b0409fab2486acfb4a01d75b12ee9d2ec7a7f60ed8f9b1fd1c\"},{\"source\":\"tests/test_cpu_terminal_recovery_barrier.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-tests-test-cpu-terminal-recovery-barrier.py\",\"read_chunk\":\"a5ef12\",\"sha256\":\"4cf56de07db9b0db30df9a1e22b4e89360e6eb660373936937b8e79a6aaa858c\"},{\"source\":\"tests/test_cpu_terminal_recovery_controls.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-tests-test-cpu-terminal-recovery-controls.py\",\"read_chunk\":\"f57f5d\",\"sha256\":\"d75dff5461d95fc7af60ba096633d0e330bbe3083cf58ddc021f2c478c943c13\"},{\"source\":\"docs/evidence/checks/v1-06f-runs-check.py\",\"archive\":\"docs/evidence/snapshots/v1-06f-candidate1-docs-evidence-checks-v1-06f-runs-check.py\",\"read_chunk\":\"f0f900\",\"sha256\":\"f8467952f89a35ab869dda2d203d92334913c7c55f691a2372b64229c41872d7\"}]")
for repo, commit in e["code_commits"].items():
    assert type(commit) is str and (re.fullmatch(r"[0-9a-f]{40}", commit)
                                   or (args.worktree_only and commit == "UNCOMMITTED")), (repo, commit)
eq(e["schema"], 1)
eq(e["slice"], "V1-06F")
eq(e["implemented_verified"], True)
eq(e["whole_v1_complete"], False)
eq(e["research_benefit_proven"], False)
eq(e["baseline_commits"], {"core": "55c2ed321dc759b6b96d721e0b3b423151bbc49b",
                            "pro": "af8e3c7f4c83a1883d11ef4756efd4c4a8ee48cc"})
support = {}
for item in e["supporting_files"]:
    name = item["path"]
    assert name not in support and not Path(name).is_absolute() and ".." not in Path(name).parts
    raw = (root / name).read_bytes()
    eq(sha(raw), item["sha256"])
    eq(len(raw), item["bytes"])
    support[name] = raw
    if not args.worktree_only:
        eq(sha(git("core", "show", e["code_commits"]["core"] + ":" + name)), item["sha256"])


def document(name):
    assert name in support, name
    return json.loads(support[name])


names = {
    "baseline": "docs/evidence/2026-09-11-v1-06f-baseline-history.json",
    "candidate": "docs/evidence/2026-09-11-v1-06f-candidate1-history.json",
    "domain_first": "docs/evidence/2026-09-11-v1-06f-domain-first-history.json",
    "target": "docs/evidence/2026-09-11-v1-06f-target-history.json",
}
histories = {key: document(name) for key, name in names.items()}
for key, count, fail in (("baseline", 757, 2), ("candidate", 760, 0),
                         ("domain_first", 761, 0), ("target", 761, 0)):
    item = histories[key]
    eq(item["before"]["output"], item["after"]["output"])
    eq(manifest(item["before"]), manifest(item["after"]))
    eq(len(manifest(item["before"])), count)
    eq(item["command"], PINNED_COMMANDS[key])
    footer(item["invocations"], {"baseline": 0, "candidate": 39, "domain_first": 2, "target": 41}[key], failed=fail)
    subprocess.run(["bash", "-n"], input=item["command"].encode(), check=True)
eq(histories["domain_first"]["status"], "initial_domain_pass_log_preserved_but_raw_reports_unavailable")
assert "reports" not in histories["domain_first"]
eq(len(histories["domain_first"]["exported_markers"]), 2)
eq(len(histories["domain_first"]["readback_failures"]), 2)
assert all(ref["exit_code"] == 1 for ref in histories["domain_first"]["readback_failures"])

files = {}
for repo, count in (("core", 761), ("pro", 223)):
    before, after = e["full_epoch"]["before"][repo], e["full_epoch"]["after"][repo]
    eq(before["output"], after["output"])
    eq(manifest(before), manifest(after))
    entries = manifest(before)
    eq(len(entries), count)
    actual = sorted(set(p.decode() for p in git(repo, "ls-files", "-c", "-o", "--exclude-standard",
                    "-z", "--", "src", "tests", "examples", "pyproject.toml").split(b"\0") if p))
    eq(actual, sorted(entries))
    if not args.worktree_only:
        git(repo, "merge-base", "--is-ancestor", e["code_commits"][repo], "HEAD")
    for name, digest in entries.items():
        eq(sha((repos[repo] / name).read_bytes()), digest)
        if not args.worktree_only:
            eq(sha(git(repo, "show", e["code_commits"][repo] + ":" + name)), digest)
    files[repo] = entries
eq(files["core"], manifest(histories["target"]["after"]))
eq(e["full_epoch"]["before"]["extra"]["output"], e["full_epoch"]["after"]["extra"]["output"])
eq(manifest(e["full_epoch"]["before"]["extra"]), manifest(e["full_epoch"]["after"]["extra"]))
extra = manifest(e["full_epoch"]["before"]["extra"])
eq(extra, {"docs/evidence/checks/v1-07b-domain-check.py":
           "7907844dc7cb2362efc75f56415a900b9e1ed729f1cecb8aa8c4311d3c842c3e"})
for name, digest in extra.items():
    eq(sha((root / name).read_bytes()), digest)

baseline = git("core", "ls-tree", "-r", "--name-only", e["baseline_commits"]["core"],
               "--", "src", "tests", "examples", "pyproject.toml").decode().splitlines()
eq(len(baseline), 755)
original_manifest = manifest(histories["baseline"]["before"])
for name in baseline:
    eq(original_manifest[name], sha(git("core", "show", e["baseline_commits"]["core"] + ":" + name)))
changes = [name for name in baseline
           if sha(git("core", "show", e["baseline_commits"]["core"] + ":" + name)) != files["core"][name]]
eq(changes, ["src/orze/core/cpu_action_budget.py", "src/orze/engine/cpu_phase.py"])
eq(sum(name.startswith("tests/") for name in baseline), 465)
new = sorted(set(files["core"]) - set(baseline))
eq(new, sorted(e["new_files"]))
eq(len(new), 6)
for name in files["pro"]:
    eq(sha(git("pro", "show", e["baseline_commits"]["pro"] + ":" + name)), files["pro"][name])
original = "docs/evidence/snapshots/v1-06f-baseline-product-tests.py"
eq(support[original].decode(), (root / "tests/test_cpu_terminal_settlement_recovery.py").read_text())
for name, src in (("cpu-action-budget", "src/orze/core/cpu_action_budget.py"),
                  ("cpu-phase", "src/orze/engine/cpu_phase.py"),
                  ("product-tests", "tests/test_cpu_terminal_settlement_recovery.py"),
                  ("recovery-helpers", "tests/cpu_terminal_recovery_helpers.py")):
    eq(sha(support["docs/evidence/snapshots/v1-06f-baseline-" + name + ".py"]), original_manifest[src])
for src, archived in (
    ("src/orze/core/cpu_action_budget.py", "docs/evidence/snapshots/v1-06f-baseline-cpu-action-budget.py"),
    ("src/orze/engine/cpu_phase.py", "docs/evidence/snapshots/v1-06f-baseline-cpu-phase.py"),
):
    eq(support[archived].decode(), git("core", "show", e["baseline_commits"]["core"] + ":" + src).decode())
eq(histories["candidate"]["snapshots"], PINNED_CANDIDATE_SNAPSHOTS)
for item in histories["candidate"]["snapshots"]:
    eq(sha(support[item["archive"]]), item["sha256"])
    if item["source"] != "docs/evidence/checks/v1-06f-runs-check.py":
        eq(item["sha256"], manifest(histories["candidate"]["before"])[item["source"]])
candidate_files = manifest(histories["candidate"]["after"])
eq(sorted(set(files["core"]) - set(candidate_files)), ["tests/test_cpu_terminal_recovery_domain.py"])
eq([name for name in sorted(candidate_files) if candidate_files[name] != files["core"][name]],
   ["tests/cpu_terminal_recovery_helpers.py"])
eq(manifest(histories["domain_first"]["before"]), files["core"])

collection = histories["target"]["collection"]
eq(collection["exit_code"], 0)
nodes = [line for line in collection["output"].splitlines() if line.startswith("tests/") and "::" in line]
eq(len(nodes), 41)
eq(len(set(nodes)), 41)
expected_cases = {
    "tests/test_cpu_terminal_settlement_recovery.py": 2,
    "tests/test_cpu_terminal_recovery_guards.py": 20,
    "tests/test_cpu_terminal_recovery_barrier.py": 12,
    "tests/test_cpu_terminal_recovery_controls.py": 5,
    "tests/test_cpu_terminal_recovery_domain.py": 2,
}
eq(dict(Counter(n.split("::")[0] for n in nodes)), expected_cases)
eq(sum(sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse((root / name).read_bytes())))
       for name in expected_cases), 137)
eq(sum(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(
    (root / "tests/cpu_terminal_recovery_helpers.py").read_bytes()))), 17)

full_outputs = {}
for key, passed, skipped in (("core", 4236, 7), ("pro", 974, 0), ("paired", 60, 0)):
    eq(e["full_epoch"]["commands"][key], PINNED_COMMANDS[key])
    full_outputs[key] = footer(e["full_epoch"]["chunks"][key], passed, skipped)
    subprocess.run(["bash", "-n"], input=e["full_epoch"]["commands"][key].encode(), check=True)


def markers(output, prefix):
    matches = [json.loads(line.split(prefix, 1)[1]) for line in output.splitlines() if prefix in line]
    eq(len(matches), 1)
    return matches[0]


def verify_reports(items, exported):
    eq([{k: v for k, v in ref.items() if k != "archive"} for ref in items], exported)
    for ref in items:
        raw = support[ref["archive"]]
        eq(sha(raw), ref["sha256"])
        eq(len(raw), ref["bytes"])


for key in ("baseline", "candidate", "target"):
    item = histories[key]
    output = "".join(r["output"] for r in item["invocations"])
    verify_reports(item["reports"], markers(output, "CPU_RECOVERY_REPORTS="))
eq(len(histories["baseline"]["reports"]), 2)
eq(len(histories["candidate"]["reports"]), 34)
eq(histories["domain_first"]["exported_markers"], markers(
   "".join(item["output"] for item in histories["domain_first"]["invocations"]), "CPU_RECOVERY_REPORTS="))
verify_reports(e["full_reports"], markers(full_outputs["core"], "CPU_RECOVERY_REPORTS="))
eq(len(e["full_reports"]), 36)
archive_checker = runpy.run_path(str(root / "docs/evidence/checks/v1-06f-runs-check.py"), run_name="f_archive_checker")
epochs = []
for label, refs in (("target", histories["target"]["reports"]), ("full", e["full_reports"])):
    eq(sorted(r["test"] for r in refs), sorted(n for n in nodes if n.split("::")[0] != "tests/test_cpu_terminal_recovery_controls.py"))
    births, native_count, cli_count = [], 0, 0
    native_births, nonces, project_roots = [], [], []
    for ref in refs:
        report = json.loads(support[ref["archive"]])
        project_roots.append(report["root"])
        by_label = {s["label"]: s for s in report["snapshots"]}
        if "domain_events" in report:
            native_bindings = [x["binding"] for x in report["domain_events"] if x["event"] == "native_ready"]
        else:
            first = report["calls"][0]
            eq(first["label"], "first_controller_crash")
            eq(first["exit_code"], 86)
            original_shot = by_label[first["after"]]
            original_attempt = original_shot["database"]["execution_attempts"][0]
            archive_checker["native"](original_shot, original_attempt, report["root"])
            crash = first["metadata"][1]
            eq(crash["transaction_open"], False)
            eq(crash["effect_guard_absent"], True)
            eq(crash["effect_confirmed"], True)
            eq(crash["deliberate_exit_code"], 86)
            eq(crash["terminal"], json.loads(original_attempt["terminal_json"]))
            eq(crash["ref"], {k: original_attempt[k] for k in ("task_id", "phase", "generation", "attempt_id")})
            native_bindings = [json.loads(original_attempt["binding_json"])["supervision"]]
            if len(report["snapshots"][-1]["worker_events"]) == 2:
                second = next(a for a in report["snapshots"][-1]["database"]["execution_attempts"] if a["task_id"] == "idea-second")
                native_bindings.append(json.loads(second["binding_json"])["supervision"])
        for binding in native_bindings:
            native_births.append((binding["worker"]["pid"], binding["worker"]["start_ticks"]))
            nonces.append(binding["nonce_sha256"])
        labels = [s["label"] for s in report["snapshots"]]
        eq(len(labels), len(set(labels)))
        for shot in report["snapshots"]:
            for entry in shot["files"].values():
                eq(sha(entry["utf8"].encode()), entry["sha256"])
                eq(len(entry["utf8"].encode()), entry["bytes"])
        end = 0
        for call in report["calls"]:
            cli_count += 1
            bound = call["controller_binding"]
            born = bound["worker"]
            assert all(type(born[k]) is int and born[k] > 0 for k in ("pid", "start_ticks"))
            births.append((born["pid"], born["start_ticks"]))
            closed = call["controller_closure"]
            archive_checker["closure"](closed, bound, call["exit_code"])
            eq(bound["identity"], {"scope": report["root"], "terminal_recovery_controller": call["label"]})
            assert call["started_monotonic"] >= end
            assert call["finished_monotonic"] > call["started_monotonic"]
            end = call["finished_monotonic"]
            eq(closed["binding"], bound)
            eq(closed["worker_returncode"], call["exit_code"])
            eq(closed["event"], "TREE_CLOSED")
            eq(closed["wait_proof"], "ECHILD_WALL")
            eq(closed["forced_cleanup"], False)
            eq(closed["stop_requested"], False)
            eq(call["metadata"], [json.loads(line.split("=", 1)[1]) for line in call["stdout"].splitlines()
                                  if line.startswith("CPU_RECOVERY_META=")])
            eq(len(call["metadata"]), 2)
            is_crash = "--crash-before-settle" in call["command"]
            eq([m["event"] for m in call["metadata"]], ["start", "crash_before_settle" if is_crash else "finish"])
            eq(call["metadata"][1]["deliberate_exit_code" if is_crash else "exit_code"], call["exit_code"])
            for marker in call["metadata"]:
                eq({k: marker[k] for k in ("pid", "start_ticks")}, born)
                assert call["started_monotonic"] <= marker["monotonic"] <= end
        native_count += (sum(x["event"] == "native_ready" for x in report["domain_events"])
                         if "domain_events" in report else len(report["snapshots"][-1]["worker_events"]))
    eq((len(births), len(set(births)), native_count, cli_count), (71, 71, 40, 71))
    eq((len(set(native_births)), len(native_births), len(set(nonces)), len(set(project_roots))), (40, 40, 40, 36))
    epochs.append((set(births), set(native_births), set(nonces), set(project_roots)))
    eq(e["measurements"][label]["native_actions"], native_count)
    eq(e["measurements"][label]["fresh_cli_invocations"], cli_count)
    for filename, checker in (("test_cpu_terminal_settlement_recovery.py", "v1-06f-runs-check.py"),
                               ("test_cpu_terminal_recovery_domain.py", "v1-06f-domain-check.py")):
        chosen = [ref["archive"] for ref in refs if ref["test"].split("::")[0].endswith(filename)]
        command = [sys.executable, "docs/evidence/checks/" + checker]
        if filename == "test_cpu_terminal_settlement_recovery.py":
            command += ["--mode", "green"]
        subprocess.run(command + chosen, check=True, stdout=subprocess.DEVNULL)
assert all(not left.intersection(right) for left, right in zip(*epochs))

subprocess.run([sys.executable, "docs/evidence/checks/v1-06f-runs-check.py", "--mode", "baseline",
                *[r["archive"] for r in histories["baseline"]["reports"]]], check=True, stdout=subprocess.DEVNULL)
eq(len(e["known_full_reports"]), 4)
eq(len(e["holdout_full_reports"]), 6)
verify_reports(e["known_full_reports"], markers(full_outputs["core"], "ACCEPTANCE_REPORTS="))
verify_reports(e["holdout_full_reports"], markers(full_outputs["core"], "HOLDOUT_REPORTS="))
# Historical A/B runner checkers validate newly generated full-suite reports;
# their old whole-worktree inventory checkers are intentionally not invoked.
for key, checker in (("known_full_reports", "v1-07a-runs-check.py"),
                     ("holdout_full_reports", "v1-07b-runs-check.py")):
    for ref in e[key]:
        eq(sha(support[ref["archive"]]), ref["sha256"])
        eq(len(support[ref["archive"]]), ref["bytes"])
    if key == "known_full_reports":
        for ref in e[key]:
            eq(ref["archive"], "docs/evidence/runs/v1-06f-known-full-" + ref["name"] + ".json")
        arguments = ["--prefix", "v1-06f-known-full"]
    else:
        arguments = [r["archive"] for r in e[key]]
    subprocess.run([sys.executable, "docs/evidence/checks/" + checker, *arguments],
                   check=True, stdout=subprocess.DEVNULL)
required = set(names.values()) | {
    "docs/cpu-terminal-recovery.md", "docs/plans/2026-09-11-v1-06f-terminal-recovery.zh-CN.md",
    *["docs/evidence/checks/" + name for name in ("v1-06f-check.py", "v1-06f-runs-check.py",
      "v1-06f-domain-check.py", "v1-07a-runs-check.py", "v1-07b-runs-check.py", "v1-07b-domain-check.py")],
    *["docs/evidence/2026-09-11-v1-06f-" + name + ".json" for name in
      ("budget-author", "guards-review", "recovery-mechanisms-review", "domain-review")],
    *["docs/evidence/snapshots/v1-06f-baseline-" + name + ".py" for name in
      ("cpu-action-budget", "cpu-phase", "product-tests", "recovery-helpers")],
}
for key in ("baseline", "candidate", "target"):
    required.update(ref["archive"] for ref in histories[key]["reports"])
required.update(ref["archive"] for ref in histories["candidate"]["snapshots"])
for key in ("full_reports", "known_full_reports", "holdout_full_reports"):
    required.update(ref["archive"] for ref in e[key])
eq(sorted(support), sorted(required))
author = document("docs/evidence/2026-09-11-v1-06f-budget-author.json")
eq([item["path"] for item in author["owned_sources"]], changes)
for item in author["owned_sources"]:
    eq(item["sha256"], files["core"][item["path"]])
    eq(item["baseline_sha256"], original_manifest[item["path"]])
    eq(sha(support[item["baseline_snapshot"]]), item["baseline_sha256"])
old_runs = [item for item in author["author_executions"] if "chunks" in item]
eq(len(old_runs), 1)
old_run = old_runs[0]
eq(footer(old_run["chunks"], 37), old_run["full_output"])
eq(old_run["passed"], 37)
eq(old_run["failed"], 0)
eq(old_run["command"], "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES= python3 -m pytest -q -p no:cacheprovider tests/test_cpu_action_budget.py tests/test_cpu_action_shared_idle.py tests/test_cpu_action_reserved_bound.py tests/test_native_cpu_action.py tests/test_cpu_product_loop.py --tb=short")
guards = document("docs/evidence/2026-09-11-v1-06f-guards-review.json")["owned_tests"]
eq(guards["sha256"], files["core"][guards["path"]])
eq((guards["cases"], guards["literal_asserts"]), (20, 21))
guard_raw = (root / guards["path"]).read_bytes()
eq(sha(guard_raw.split(b"\n\ndef test_recovery_refuses_unbound_replication_request_field", 1)[0]),
   guards["growth"]["initial_18_sha256"])
mechanisms = document("docs/evidence/2026-09-11-v1-06f-recovery-mechanisms-review.json")
for name, item in mechanisms["tests"].items():
    eq(item["sha256"], files["core"][name])
    assertions = [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse((root / name).read_bytes())) if isinstance(n, ast.Assert)]
    eq(len(assertions), item["literal_asserts"])
    eq(sha(encoded(assertions)), item["assert_ast_sha256"])
    eq(item["cases"], expected_cases[name])
domain_review = document("docs/evidence/2026-09-11-v1-06f-domain-review.json")
for mapping in (mechanisms["related_current_files"], domain_review["fixed_sources"]):
    for name, item in mapping.items():
        raw = (root / name).read_bytes()
        eq(sha(raw), item["sha256"])
        eq(len(raw), item["bytes"])
for item in domain_review["supporting_references"]:
    eq(sha(support[item["path"]]), item["sha256"])
print(json.dumps({"result": "verified", "fixed_commit_checked": not args.worktree_only,
    "source_files": 984, "extra_imported_checkers": 1, "old_tests_byte_exact": 465,
    "new_cases": 41, "new_test_asserts": 137, "helper_asserts": 17,
    "support_files": len(support), "whole_v1_complete": False,
    "record_consistency_not_independent_execution_signature": True}, sort_keys=True))
