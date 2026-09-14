"""Archive only this task's generated synthetic fixtures and package evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

out = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
groups = {
    "benchmark-v1": Path("/tmp/orze-budget-normalization-20260914-v1"),
    "benchmark-v2": Path("/tmp/orze-budget-normalization-20260914-v2"),
    "installed-product": Path("/tmp/onz-installed-1"),
}
index = {}
for name, root in groups.items():
    report = root / "report.json"
    assert report.is_file()
    if name.startswith("benchmark"):
        assert len(json.loads(report.read_text())["rows"]) == 3
    else:
        assert json.loads(report.read_text())["exit_code"] == 0
    files = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    archive = out / (name + ".tar.gz")
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tar:
        for relative in sorted(files):
            tar.add(root / relative, arcname=relative, recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        actual = {m.name: {"bytes": m.size, "sha256": sha(tar.extractfile(m).read())}
                  for m in tar.getmembers()}
    assert actual == files
    shutil.copyfile(report, out / (name + ".json"))
    index[name] = {"root": str(root), "archive": archive.name,
                   "sha256": sha(archive.read_bytes()), "members": files}
package = Path("/tmp/orze-normalization-package-20260914")
target = out / "package"
target.mkdir()
selected = [p for p in package.iterdir() if p.is_file()]
selected += list((package / "wheels").glob("*.whl"))
for p in selected:
    shutil.copyfile(p, target / p.name)
index["package"] = {"root": str(package), "files": {
    str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
    for p in target.iterdir()}}
(out / "archive-index.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
print(json.dumps({name: len(value.get("members", value.get("files", {}))) for name, value in index.items()}))
