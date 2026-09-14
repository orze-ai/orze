"""Complete the staged-wheel installation without system ensurepip or network."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

root = Path(sys.argv[1]).resolve()
venv = root / "verified-core-only-venv"
assert not venv.exists()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
bootstrap, = Path("/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP/bootstrap").glob("pip-*.whl")
wheel, = (root / "wheels").glob("orze-*.whl")
deps = Path("/hot-data/fsx/workspace/erik/orze-history-cost-package-2026-09-14.KBo6Xh/wheelhouse")
python = venv / "bin/python"
loader = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
commands = [
    [sys.executable, "-m", "venv", "--without-pip", str(venv)],
    [str(python), "-I", "-B", "-c", loader, str(bootstrap), "--isolated", "--disable-pip-version-check",
     "install", "--no-index", "--no-compile", str(bootstrap)],
    [str(python), "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "install", "--no-index",
     "--no-compile", "--find-links", str(deps), str(wheel), "--report", str(root / "verified-installed.json")],
    [str(python), "-I", "-B", "-m", "pip", "check"],
    [str(python), "-I", "-B", "-m", "orze.cli", "--help"],
]
records = []
for index, command in enumerate(commands):
    path = root / f"verified-offline-{index}.log"
    with path.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    records.append({"command": command, "exit_code": result.returncode,
                    "log": str(path), "log_sha256": sha(path.read_bytes())})
    assert result.returncode == 0, records[-1]
site, = (venv / "lib").glob("python*/site-packages")
assert not (site / "orze_pro").exists()
manifest = json.loads((root / "source-manifest.json").read_text())
with zipfile.ZipFile(wheel) as archive:
    files = {name: archive.read(name) for name in archive.namelist() if name.startswith("orze/") and not name.endswith("/")}
    assert {name for name in files if name.endswith(".py")} == {
        name[4:] for name in manifest if name.startswith("src/orze/") and name.endswith(".py")}
    for name, raw in files.items():
        assert sha(raw) == manifest["src/" + name]
        assert (site / name).read_bytes() == raw
record, = site.glob("orze-*.dist-info/RECORD")
record_rows = list(csv.reader(io.StringIO(record.read_text())))
for relative, digest, size in record_rows:
    if relative == str(record.relative_to(site)):
        assert digest == size == ""
        continue
    path = (site / relative).resolve()
    assert path.is_relative_to(venv)
    raw = path.read_bytes()
    assert str(len(raw)) == size
    assert digest == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
report = {"passed": True, "commands": records, "wheel": str(wheel), "wheel_sha256": sha(wheel.read_bytes()),
          "source_manifest_sha256": sha((root / "source-manifest.json").read_bytes()), "installed_site": str(site),
          "package_files": {name: sha(raw) for name, raw in files.items()}, "installed_record_rows": len(record_rows),
          "initial_failure": "System Python ensurepip unavailable; original 1.log retained; --without-pip plus existing offline wheel used.",
          "bootstrap_sha256": sha(bootstrap.read_bytes()), "limits": "Staged working source; fixed Git identity is checked by the final verifier after commit. No production deployment."}
(root / "build-install.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("passed", "wheel_sha256", "installed_record_rows")}))
