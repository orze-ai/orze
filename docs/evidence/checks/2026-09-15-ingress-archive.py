"""Archive completed isolated runs once; preserve failed inputs as well."""
import hashlib
import json
from pathlib import Path
import tarfile

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs/evidence/runs/2026-09-15-ingress-bounds"
ROOTS = {
    "benchmark_v1": "/tmp/orze-ingress-bounds-20260915-v1",
    "benchmark_v2": "/tmp/orze-ingress-bounds-20260915-v2",
    "product_failed_v1": "/tmp/orze-ingress-product-20260915-v1",
    "product": "/tmp/orze-ingress-product-20260915-v2",
    "duplicate_control": "/tmp/orze-ingress-duplicate-control-20260915",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "archives.json").exists()
    for name, path in ROOTS.items():
        if name != "product_failed_v1":
            assert (Path(path) / "report.json").is_file(), name
    records = {}
    for name, path in ROOTS.items():
        root = Path(path)
        target = OUT / (name + ".tar.gz")
        assert not target.exists()
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
