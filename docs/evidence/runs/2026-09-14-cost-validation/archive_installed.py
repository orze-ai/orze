"""Archive only the completed scratch installed run and its four test projects."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET

if not __debug__:
    raise SystemExit("archive verification requires assertions enabled")
SOURCE, OUT = (Path(x).resolve() for x in sys.argv[1:3])
assert not OUT.exists()
OUT.mkdir(parents=True)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
report = json.loads((SOURCE / "report.json").read_bytes())
assert report["exit_code"] == 0 and report["source_tests_prepost_exact"]
files = {}


def copy(source, name):
    target = OUT / name
    raw = source.read_bytes()
    shutil.copyfile(source, target)
    assert target.read_bytes() == raw
    files[name] = {"original": str(source), "sha256": sha(raw), "bytes": len(raw)}


copy(SOURCE / "report.json", "report.json")
copy(SOURCE / "junit.xml", "junit.xml")
cases = list(ET.parse(SOURCE / "junit.xml").iter("testcase"))
assert len(cases) == 35 and all(len(c) == 0 for c in cases)
projects = {}
for event in report["events"]:
    if event["when"] != "teardown":
        continue
    for line in event["stdout"].splitlines():
        if not line.startswith("CPU_PROPOSAL_PAGING_REPORT="):
            continue
        path = Path(line.split("=", 1)[1]).resolve()
        assert path.is_relative_to(SOURCE / "pytest")
        assert path not in projects
        payload = json.loads(path.read_bytes())
        assert payload["actual_cli_invocation_in_same_pytest_interpreter"]
        assert payload["no_proposal_receipt_is_claimed_as_worker_or_execution_authority"]
        assert all(item["event"] == "TREE_CLOSED" and item["wait_proof"] == "ECHILD_WALL"
                   for item in payload["actual_closure_receipts"])
        copy(path, path.parent.name + "-report.json")
        projects[path] = {"nodeid": event["nodeid"], "actual_native_workers": payload["actual_native_workers"],
                          "proposal_coordinator_outcomes": len(payload["admissions"]),
                          "metadata_evidence_prefixes": payload["metadata_evidence_prefixes"],
                          "closure_receipts": payload["actual_closure_receipts"]}
assert len(projects) == 4 and sum(p["actual_native_workers"] for p in projects.values()) == 1
members = {}
archive = OUT / "product-projects.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for report_path in projects:
        project = report_path.parent
        for path in sorted(project.rglob("*")):
            assert not path.is_symlink(), path
            if path.is_file():
                name = str(path.relative_to(project.parent))
                raw = path.read_bytes()
                members[name] = {"sha256": sha(raw), "bytes": len(raw)}
                tar.add(path, arcname=name, recursive=False)
with tarfile.open(archive, "r:gz") as tar:
    entries = tar.getmembers()
    assert len(entries) == len(members) and {e.name for e in entries} == set(members)
    for entry in entries:
        assert entry.isfile()
        raw = tar.extractfile(entry).read()
        assert sha(raw) == members[entry.name]["sha256"] and len(raw) == members[entry.name]["bytes"]
result = {"schema": 1, "status": "passed", "original": str(SOURCE), "files": files,
          "passed_tests": len(cases), "core_commit": report["core_commit"],
          "wheel_sha256": report["wheel_sha256"], "loaded_orze_modules": len(report["loaded_orze_modules"]),
          "source_test_files": len(report["source_test_before"]), "source_tests_prepost_exact": True,
          "projects": {str(k): v for k, v in projects.items()},
          "archive": {"path": archive.name, "sha256": sha(archive.read_bytes()), "members": members},
          "limits": report["limits"]}
(OUT / "index.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print(json.dumps({"index": str(OUT / "index.json"), "sha256": sha((OUT / "index.json").read_bytes()),
                  "passed_tests": len(cases), "actual_native_workers": 1, "archive_members": len(members)}))
