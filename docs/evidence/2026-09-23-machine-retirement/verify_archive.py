"""Verify archived bytes without models, GPUs, imports of candidates, or writes."""
import hashlib
import json
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
index = json.loads(Path(__file__).with_name("archive-index.json").read_text())
tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
errors = []
total_files = 0
total_bytes = 0
for group in index["groups"]:
    entries = []
    for name in sorted(n for n in tracked if n.startswith(group["path"] + "/")):
        path = repo / name
        if not path.is_file():
            errors.append("missing: " + name)
            continue
        blob = path.read_bytes()
        entries.append([name, len(blob), hashlib.sha256(blob).hexdigest()])
    digest = hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    size = sum(entry[1] for entry in entries)
    if (len(entries), size, digest) != (group["files"], group["bytes"], group["tree_sha256"]):
        errors.append("archive changed: " + group["path"])
    total_files += len(entries)
    total_bytes += size
for name, digest in index["code_files"].items():
    path = repo / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        errors.append("code changed since handoff: " + name)
print(json.dumps({"valid": not errors, "archive_groups": len(index["groups"]),
                  "archived_files": total_files, "archived_bytes": total_bytes,
                  "code_files": len(index["code_files"]), "errors": errors,
                  "scope": "Byte integrity only; does not re-prove scientific validity, GPU execution or external data availability."}, indent=2))
raise SystemExit(bool(errors))
