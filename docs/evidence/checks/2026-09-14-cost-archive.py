"""Mechanical byte-exact archive of three owned temporary diagnostics; no execution."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "docs/evidence/runs/2026-09-14-cost-equivalence"
INPUTS = {
    "metadata-equivalence": Path("/tmp/orze-budget-equivalence-q0qhx9zr"),
    "scale": Path("/tmp/orze-cost-paired-scale-04nl32e4"),
    "real-regression": Path("/tmp/orze-costr-h0wsdkyn"),
}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def inventory(root):
    rows = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        rel = path.relative_to(root).as_posix()
        if stat.S_ISLNK(mode):
            target = str(path.readlink())
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("external symlink: " + rel)
            rows.append({"path": rel, "type": "symlink", "target": target})
        elif stat.S_ISREG(mode):
            data = path.read_bytes()
            rows.append({"path": rel, "type": "file", "bytes": len(data), "sha256": sha(data)})
        elif not stat.S_ISDIR(mode):
            raise ValueError("not an inert generated file: " + rel)
    return rows

def main():
    result = {"schema": 1, "archives": [], "limits": [
        "Only three named owned private /tmp fixture trees are archived; no user projects or account paths.",
        "Sources copied before runs are Core-only; no Pro implementation, authorization, environment dump or dependency binaries.",
        "Symlinks are retained as link metadata without dereferencing, and must resolve within the original private fixture.",
        "Synthetic metadata ledger artifacts do not prove native process, TREE, effect or SETTLED authority.",
        "The real-regression directory separately contains unmodified native/recovery fixtures, their real DBs and exported reports.",
    ]}
    for label, original in INPUTS.items():
        before = inventory(original)
        out = DEST / label
        out.mkdir(parents=True, exist_ok=False)
        archive = out / "raw.tar.gz"
        with tarfile.open(archive, "w:gz", dereference=False) as tar:
            tar.add(original, arcname="raw")
        with tarfile.open(archive, "r:gz") as tar:
            actual = []
            for member in tar.getmembers():
                rel = member.name.removeprefix("raw/") if member.name != "raw" else ""
                if member.isfile():
                    raw = tar.extractfile(member).read()
                    actual.append({"path": rel, "type": "file", "bytes": len(raw), "sha256": sha(raw)})
                elif member.issym():
                    actual.append({"path": rel, "type": "symlink", "target": member.linkname})
                elif not member.isdir():
                    raise ValueError("unexpected tar member " + member.name)
        assert sorted(actual, key=lambda row: row["path"]) == before
        assert inventory(original) == before
        copies = []
        for path in sorted(original.iterdir()):
            if path.is_file() and not path.is_symlink():
                target = out / path.name
                shutil.copyfile(path, target)
                assert path.read_bytes() == target.read_bytes()
                copies.append({"original": str(path), "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": sha(target.read_bytes())})
        reports = sorted(original.rglob("budget-equivalence.json")) + sorted(original.rglob("recovery-report.json"))
        if label == "scale":
            reports = [original / "data/report.json"]
        if reports:
            (out / "reports").mkdir()
        for i, path in enumerate(reports):
            target = out / "reports" / (str(i).zfill(2) + "-" + path.name)
            shutil.copyfile(path, target)
            assert path.read_bytes() == target.read_bytes()
            copies.append({"original": str(path), "path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": sha(target.read_bytes())})
        item = {"label": label, "original": str(original), "archive": str(archive.relative_to(ROOT)), "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive.read_bytes()), "entries": before, "copies": copies, "all_members_byte_exact": True}
        (out / "index.json").write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
        result["archives"].append({"label": label, "index": str((out / "index.json").relative_to(ROOT)), "index_sha256": sha((out / "index.json").read_bytes()), "archive_sha256": item["archive_sha256"], "files": sum(row["type"] == "file" for row in before), "symlinks": sum(row["type"] == "symlink" for row in before), "report_copies": len(reports), "archive_bytes": item["archive_bytes"]})
    (DEST / "archive-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
