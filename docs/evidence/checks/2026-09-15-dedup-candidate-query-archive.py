"""Preserve the isolated profile, repeated measurements and actual CLI project."""
import hashlib
import json
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/evidence/runs/2026-09-15-dedup-candidate-query"
ROOTS = {'benchmark_v1': '/tmp/orze-dedup-candidate-query-benchmark-20260915-v1', 'benchmark_v2': '/tmp/orze-dedup-candidate-query-benchmark-20260915-v2', 'benchmark_v3': '/tmp/orze-dedup-candidate-query-benchmark-20260915-v3', 'benchmark_v4': '/tmp/orze-dedup-candidate-query-benchmark-20260915-v4', 'differential_v1': '/tmp/orze-dedup-candidate-query-differential-20260915-v1', 'differential_v2': '/tmp/orze-dedup-candidate-query-differential-20260915-v2', 'product': '/tmp/orze-dedup-candidate-query-product-20260915'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "archives.json").exists()
    records = {}
    for name, path in ROOTS.items():
        root = Path(path)
        assert (root / "report.json").is_file(), name
        target = OUT / (name + ".tar.gz")
        members = {}
        with tarfile.open(target, "x:gz") as tar:
            for item in sorted(root.rglob("*")):
                assert not item.is_symlink(), item
                if not item.is_file() or "__pycache__" in item.parts or item.suffix == ".pyc":
                    continue
                relative = item.relative_to(root).as_posix()
                members[relative] = sha(item)
                tar.add(item, arcname=relative, recursive=False)
        records[name] = {"root": path, "archive": target.name, "sha256": sha(target), "members": members}
    (OUT / "archives.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"members": len(r["members"]), "sha256": r["sha256"]} for name, r in records.items()}))


if __name__ == "__main__":
    main()
