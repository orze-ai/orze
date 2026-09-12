"""Offline fixed Core+Pro package preflight. No worker or licensed Pro execution."""
import ast
import base64
import csv
import email
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
PRO_REPO = Path("/hot-data/fsx/workspace/erik/orze-implementation-2026-09-10.SyXgzC/orze-pro")
CORE_REPO = Path("/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/orze")
OLD = Path("/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP")
CURRENT = Path("/hot-data/fsx/workspace/erik/orze-unlimited-validation-2026-09-12.lUanmfzX")
PRO_COMMIT = "81f8a52bdca288c8f39ed213cd08887467d52036"
CORE_COMMIT = "6007f6efe49964f14317cd620f0b177615c35da8"
CORE_SHA = "9898fd3038fb0a723a7fbbb19aa4b9c027032e89a8b8499606f42c70ee6de43d"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")

def read_archive(path):
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        assert all(not Path(m.name).is_absolute() and ".." not in Path(m.name).parts for m in members)
        return {m.name: tar.extractfile(m).read() for m in members if m.isfile()}

def record_rows(raw):
    rows = list(csv.reader(io.StringIO(raw.decode())))
    assert all(len(r) == 3 for r in rows)
    assert len({r[0] for r in rows}) == len(rows)
    return rows

def verify_wheel(wheel, blobs, package, patterns):
    with zipfile.ZipFile(wheel) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        files = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
    expected = {n[4:] for n in blobs if n.startswith("src/" + package + "/") and n.endswith(".py")}
    package_dir = ROOT / ("pro-source/src/" + package) if package == "orze_pro" else CURRENT / ("source/src/" + package)
    declared = {package + "/" + str(p.relative_to(package_dir)) for pattern in patterns
                for p in package_dir.glob(pattern) if p.is_file()}
    expected |= declared
    payload = {n: raw for n, raw in files.items() if n.startswith(package + "/")}
    assert set(payload) == expected, (set(payload) - expected, expected - set(payload))
    for n, raw in payload.items():
        assert raw == blobs["src/" + n], n
    record, = [n for n in files if n.endswith(".dist-info/RECORD")]
    rows = record_rows(files[record])
    assert {r[0] for r in rows} == set(files)
    for n, checksum, size in rows:
        if n == record:
            assert checksum == size == ""
        else:
            assert checksum == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(files[n]).digest()).decode().rstrip("=")
            assert int(size) == len(files[n])
    metadata, = [n for n in files if n.endswith(".dist-info/METADATA")]
    fields = email.message_from_bytes(files[metadata])
    return {"wheel": str(wheel), "sha256": sha(wheel.read_bytes()), "bytes": wheel.stat().st_size,
            "name": fields["Name"], "version": fields["Version"],
            "requires_dist": fields.get_all("Requires-Dist", []), "record": record,
            "record_rows": len(rows), "record_verified": True,
            "package_files": {n: sha(raw) for n, raw in sorted(payload.items())},
            "python_files": sum(n.endswith(".py") for n in payload),
            "resource_files": sum(not n.endswith(".py") for n in payload),
            "package_data_patterns": patterns}, files

def main():
    logs = ROOT / "logs"
    logs.mkdir()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    commands = []
    report = {"schema": 1, "core_commit": CORE_COMMIT, "pro_commit": PRO_COMMIT,
              "status": "in_progress", "offline_dependencies": True, "workers_started": 0,
              "pro_runtime_imported": False, "license_runtime_validated": False,
              "production_switched": False, "root": str(ROOT),
              "build_tools": {n: importlib.metadata.version(n) for n in ("build", "setuptools", "wheel")}}
    def run(label, argv, cwd=ROOT):
        started = time.monotonic()
        result = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, capture_output=True, timeout=600)
        item = {"label": label, "argv": list(map(str, argv)), "cwd": str(cwd),
                "exit_code": result.returncode, "wall_seconds": time.monotonic() - started}
        for name, raw in (("stdout", result.stdout), ("stderr", result.stderr)):
            path = logs / (label + "." + name)
            path.write_bytes(raw)
            item[name] = {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}
        commands.append(item)
        save(ROOT / "commands.json", commands)
        print(json.dumps({"label": label, "exit_code": result.returncode}), flush=True)
        if result.returncode:
            raise RuntimeError("command failed; raw logs retained: " + label)
        return result
    try:
        for name, repo, commit in (("pro", PRO_REPO, PRO_COMMIT), ("core", CORE_REPO, CORE_COMMIT)):
            actual = run(name + "-commit", ["git", "-C", repo, "rev-parse", commit + "^{commit}"]).stdout.decode().strip()
            assert actual == commit
            argv = ["git", "-C", repo, "archive", "--format=tar", "--output=" + str(ROOT / (name + "-source.tar")), commit]
            if name == "core":
                argv.extend(["src/orze", "pyproject.toml"])
            run(name + "-archive", argv)
        source = ROOT / "pro-source"
        source.mkdir()
        run("extract-pro", ["tar", "-xf", ROOT / "pro-source.tar", "-C", source])
        pro_blobs = read_archive(ROOT / "pro-source.tar")
        core_blobs = read_archive(ROOT / "core-source.tar")
        wheels = ROOT / "wheelhouse"
        wheels.mkdir()
        run("build-pro-wheel", [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", wheels, source])
        pro_wheel, = wheels.glob("orze_pro-*.whl")
        core_source_wheel, = (CURRENT / "wheelhouse").glob("orze-*.whl")
        assert sha(core_source_wheel.read_bytes()) == CORE_SHA
        core_wheel = wheels / core_source_wheel.name
        shutil.copyfile(core_source_wheel, core_wheel)
        payloads = {}
        report["packages"] = {}
        for name, wheel, blobs in (("core", core_wheel, core_blobs), ("pro", pro_wheel, pro_blobs)):
            section = blobs["pyproject.toml"].decode().split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
            patterns = ast.literal_eval(section.split("=", 1)[1].strip())
            data, files = verify_wheel(wheel, blobs, "orze" if name == "core" else "orze_pro", patterns)
            report["packages"][name] = data
            payloads[name] = files
        dependencies = json.loads((OLD / "dependency-lock.json").read_bytes())
        report["original_dependency_lock_sha256"] = sha((OLD / "dependency-lock.json").read_bytes())
        new_dependencies = []
        for d in dependencies:
            if d["name"] in ("orze", "orze-pro"):
                p = report["packages"]["core" if d["name"] == "orze" else "pro"]
                new_dependencies.append({**d, "version": p["version"], "sha256": p["sha256"],
                    "source_url": Path(p["wheel"]).as_uri(), "source_commit": CORE_COMMIT if d["name"] == "orze" else PRO_COMMIT})
            else:
                matches = [p for p in (OLD / "wheelhouse").glob("*.whl") if sha(p.read_bytes()) == d["sha256"]]
                assert len(matches) == 1
                shutil.copyfile(matches[0], wheels / matches[0].name)
                new_dependencies.append({**d, "offline_wheel": str(wheels / matches[0].name)})
        assert len(new_dependencies) == 25
        save(ROOT / "dependency-lock.json", new_dependencies)
        original_lock = (OLD / "paired.lock").read_text()
        assert sha(original_lock.encode()) == "e35ba7e3a3d23a9ef7db18cca286ba83a80a76578ec2b6fa10e4a3c9b486e69f"
        lock = original_lock
        for d in new_dependencies:
            if d["name"] in ("orze", "orze-pro"):
                line = d["name"] + "==" + d["version"] + " --hash=sha256:" + d["sha256"]
                lock, changed = re.subn("^" + re.escape(d["name"]) + r"==[^\n]+$", line, lock, flags=re.MULTILINE)
                assert changed == 1
        (ROOT / "paired.lock").write_text(lock)
        assert [l for l in original_lock.splitlines() if not l.startswith(("orze==", "orze-pro=="))] == [
            l for l in lock.splitlines() if not l.startswith(("orze==", "orze-pro=="))]
        target = ROOT / "paired-venv"
        run("create-venv", [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", target])
        python = target / "bin/python"
        bootstrap, = (OLD / "bootstrap").glob("pip-*.whl")
        loader = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('pip',run_name='__main__')"
        run("bootstrap-pip", [python, "-I", "-B", "-c", loader, bootstrap, "--isolated", "--disable-pip-version-check",
            "--no-input", "install", "--no-index", "--no-compile", bootstrap, "--report", ROOT / "bootstrap-install.json"])
        pip = [python, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input"]
        run("install-offline-paired", pip + ["install", "--no-index", "--no-compile", "--find-links", wheels,
            "--require-hashes", "-r", ROOT / "paired.lock", "--report", ROOT / "installed.json"])
        run("pip-check", pip + ["check"])
        run("pip-freeze", pip + ["freeze", "--all"])
        run("core-help-no-pro-chain", [python, "-I", "-B", ROOT / "help_probe.py"])
        sites, = (target / "lib").glob("python*/site-packages")
        installed_records = {}
        for name, files in payloads.items():
            record_name = report["packages"][name]["record"]
            for relative, raw in files.items():
                if relative != record_name:
                    assert (sites / relative).read_bytes() == raw, relative
            rows = record_rows((sites / record_name).read_bytes())
            for relative, checksum, size in rows:
                path = (sites / relative).resolve()
                assert path.is_relative_to(target)
                if relative == record_name:
                    assert checksum == size == ""
                else:
                    assert checksum.startswith("sha256=")
                    raw = path.read_bytes()
                    assert int(size) == len(raw)
                    assert checksum[7:] == base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
            installed_records[name] = {"path": str(sites / record_name), "rows": len(rows),
                "sha256": sha((sites / record_name).read_bytes()), "all_recorded_entries_verified": True}
        help_stdout = (logs / "core-help-no-pro-chain.stdout").read_text()
        assert 'HELP_ISOLATION={"core_help_exit": 0' in help_stdout
        report.update(status="passed", paired_venv=str(target), installed_records=installed_records,
            exact_original_dependency_wheels=23, pip_check_passed=True, core_help_passed=True,
            paired_lock_sha256=sha((ROOT / "paired.lock").read_bytes()),
            dependency_lock_sha256=sha((ROOT / "dependency-lock.json").read_bytes()),
            help_isolation="Read-only Python audit guard rejects Pro imports, credential opens, child launches, and network; no product function or license result replaced; zero forbidden events observed.",
            license_read_or_patched=False,
            source_archives={n: {"sha256": sha((ROOT / (n + "-source.tar")).read_bytes()),
                                 "bytes": (ROOT / (n + "-source.tar")).stat().st_size} for n in ("core", "pro")},
            limitations=["Installed metadata and byte integrity only; Pro execution and genuine authorization remain unverified.",
                         "No CPU or GPU worker was launched; no production target was changed.",
                         "Passing installation does not resolve the known CephFS role-release incompatibility, long AF_UNIX paths, or renameat2 support.",
                         "No current or original Core-only/formal environment, source worktree, account, or license was modified."])
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        report["commands"] = commands
        save(ROOT / "review.json", report)
        print("PAIRED_PACKAGE_REPORT=" + str(ROOT / "review.json"), flush=True)

if __name__ == "__main__":
    main()

