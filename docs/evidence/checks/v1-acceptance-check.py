"""Original V1 acceptance index consistency; not scientific/semantic attestation."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

if not __debug__:
    raise SystemExit("Do not use -O: this verifier requires assertions.")
ROOT = Path(__file__).resolve().parents[3]
REPOS = {"core": ROOT, "pro": ROOT.parent / "orze-pro"}
PLAN = "docs/plans/2026-09-10-autoresearch-v1.zh-CN.md"
PLAN_SHA = "b9b3c0db98c18e443c4468914073305bb88e32f4137cc9967cb7eac0374d5f3d"
S2 = "docs/evidence/2026-09-11-s2-claim-reader.json"
S2_SHA = "fbca4393ef2feae4f262ed1c2afa86d93f6f2ae2c8598683f173c37396ddc83f"
S2_CHECK = "docs/evidence/checks/s2-check.py"
S2_CHECK_SHA = "b9b8c58d95bbd8902cc6adf6310d3ce0e384e7a9fbe88c449eb5f3468f652c6e"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def blob(repo, commit, path):
    assert re.fullmatch("[0-9a-f]{40}", commit)
    return subprocess.check_output(["git", "-C", str(REPOS[repo]), "show", commit + ":" + path])

def main():
    d = json.loads((ROOT / "docs/evidence/2026-09-11-autoresearch-v1-acceptance.json").read_bytes())
    assert d["schema"] == 1 and d["scope"] == "original_V1_00_to_07_minimal_mechanisms"
    assert d["accepted"] is True
    assert all(d[k] is False for k in ("research_benefit_proven", "live_gpu_provider_validated",
        "production_deployed", "new_unseen_holdout", "universal_automatic_recovery"))
    assert sha(Path(__file__).read_bytes()) == d["checker_sha256"]
    assert sha((ROOT / PLAN).read_bytes()) == PLAN_SHA == d["original_plan"]["sha256"]
    assert sha(blob("core", d["original_plan"]["freeze_commit"], PLAN)) == PLAN_SHA
    assert sha((ROOT / S2).read_bytes()) == S2_SHA == d["s2"]["sha256"]
    assert sha(blob("core", d["s2"]["evidence_commit"], S2)) == S2_SHA
    assert sha((ROOT / S2_CHECK).read_bytes()) == S2_CHECK_SHA
    s2 = json.loads((ROOT / S2).read_bytes())
    assert d["code_commits"] == s2["code_commits"]
    subprocess.run([sys.executable, str(ROOT / S2_CHECK)], check=True)
    for p, digest in d["document_sha256"].items():
        assert sha((ROOT / p).read_bytes()) == digest
    assert set(d["document_sha256"]) == {
        "docs/plans/2026-09-11-autoresearch-v1-acceptance.zh-CN.md",
        "docs/plans/2026-09-10-autoresearch-v1-status.zh-CN.md"}
    suites = {n: list(ET.fromstring((ROOT / s2["junit"][n]["archive"]).read_bytes()).iter("testcase"))
              for n in REPOS}
    assert [r["id"] for r in d["rows"]] == ["V1-%02d" % i for i in range(8)]
    selectors, cases, history = [], 0, 0
    for row in d["rows"]:
        assert row["condition"] and row["history"]
        assert bool(row["nodes"]) == (row["id"] != "V1-00")
        for node in row["nodes"]:
            repo, module, test = node["repo"], node["module"], node["test"]
            assert node["path"] == "tests/" + module + ".py"
            raw = (REPOS[repo] / node["path"]).read_bytes()
            assert sha(raw) == node["sha256"]
            assert blob(repo, d["code_commits"][repo], node["path"]) == raw
            assert sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == test
                       for n in ast.walk(ast.parse(raw))) == 1
            selected = [c for c in suites[repo] if module in c.attrib["classname"].split(".")
                        and c.attrib["name"].split("[")[0] == test]
            assert selected and [dict(c.attrib) for c in selected] == node["cases"]
            assert all(not any(c.find(k) is not None for k in ("failure", "error", "skipped"))
                       for c in selected)
            selectors.append((repo, module, test))
            cases += len(selected)
        for h in row["history"]:
            assert h["fixed_commit"] == d["code_commits"][h["repo"]]
            raw = (REPOS[h["repo"]] / h["path"]).read_bytes()
            assert sha(raw) == h["sha256"] and len(raw) == h["bytes"]
            assert blob(h["repo"], h["fixed_commit"], h["path"]) == raw
            history += 1
    assert len(selectors) == len(set(selectors)) == 41 and cases == 81 and history == 23
    print(json.dumps({"result": "verified", "original_rows": 8, "representative_selectors": 41,
        "matched_passed_cases": 81, "history_references": 23,
        "fixed_source_and_s2_records_checked": True, "research_benefit_proven": False,
        "record_consistency_not_semantic_or_execution_attestation": True}, sort_keys=True))

if __name__ == "__main__":
    main()
