"""Actual installed-Core payload under an explicit external pytest harness."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = Path("/hot-data/fsx/workspace/erik/orze-production-validation-2026-09-12.UgS3uV/orze")
PACKAGE = Path("/hot-data/fsx/workspace/erik/orze-long-history-package-2026-09-14.dxa7jzhz")
SITE = PACKAGE / "core-only-venv/lib/python3.10/site-packages"
WHEEL = PACKAGE / "wheelhouse/orze-4.6.2-py3-none-any.whl"
COMMIT = "26643ed7d6fa42e1d72939c61eb6fc8d249ffa66"
EXPECTED_WHEEL = "0501beda118b0300f89a756ce5636c21b326876c00f83cb8be175c943687fc50"
sha = lambda raw: hashlib.sha256(raw).hexdigest()
assert sha(WHEEL.read_bytes()) == EXPECTED_WHEEL
sys.path.insert(0, str(SITE))
sys.path.insert(1, str(REPO / "tests"))
assert str(REPO / "src") not in sys.path

import pytest
import orze

loaded = {}
events = []
paths = []
git_hashes = {}
with zipfile.ZipFile(WHEEL) as z:
    wheel_hashes = {name: sha(z.read(name)) for name in z.namelist()
                    if name.startswith("orze/") and not name.endswith("/")}


def audit(label):
    paths.append({"label": label, "sys_path": list(sys.path)})
    assert str(REPO / "src") not in sys.path
    for name, module in tuple(sys.modules.items()):
        if name != "orze" and not name.startswith("orze."):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            for directory in getattr(module, "__path__", ()):
                assert Path(directory).resolve().is_relative_to(SITE / "orze")
            continue
        path = Path(path).resolve()
        assert path.is_relative_to(SITE / "orze"), (name, str(path))
        member = path.relative_to(SITE).as_posix()
        actual = sha(path.read_bytes())
        assert actual == wheel_hashes[member], (name, member)
        if member not in git_hashes:
            raw = subprocess.check_output(["git", "show", COMMIT + ":src/" + member], cwd=REPO)
            git_hashes[member] = sha(raw)
        assert actual == git_hashes[member], (name, "Git mismatch")
        loaded[name] = {"file": str(path), "wheel_member": member, "sha256": actual,
                        "git_source": "src/" + member}


class Audit:
    def pytest_sessionstart(self, session):
        audit("sessionstart")

    def pytest_runtest_logreport(self, report):
        events.append({"nodeid": report.nodeid, "when": report.when,
                       "outcome": report.outcome, "stdout": report.capstdout,
                       "stderr": report.capstderr})
        audit(report.nodeid + ":" + report.when)

    def pytest_sessionfinish(self, session, exitstatus):
        audit("sessionfinish")


def hashes():
    names = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "--", "src", "tests"],
        cwd=REPO, text=True).splitlines()
    return {p: sha((REPO / p).read_bytes()) for p in sorted(set(names))
            if (REPO / p).is_file()}


audit("before_pytest")
before = hashes()
args = ["-q", "--confcutdir", str(REPO / "tests"), "-o", "pythonpath=" + str(SITE),
        str(REPO / "tests/test_cpu_proposal_paging_product_review.py"),
        "--basetemp", str(ROOT / "pytest"), "--junitxml", str(ROOT / "junit.xml")]
started = time.monotonic()
code = pytest.main(args, plugins=[Audit()])
audit("after_pytest")
after = hashes()
assert before == after
report = {
    "schema": 1, "classification": "installed Core payload under external global pytest harness",
    "argv": [sys.executable, "-B", str(Path(__file__).resolve())],
    "pytest_arguments": args, "pytest_module": pytest.__file__,
    "actual_interpreter": sys.executable, "python_version": sys.version,
    "installed_core_only_site": str(SITE), "orze_file": orze.__file__,
    "core_commit": COMMIT, "wheel": str(WHEEL), "wheel_sha256": EXPECTED_WHEEL,
    "loaded_orze_modules": loaded, "sys_path_observations": paths,
    "events": events, "source_test_before": before, "source_test_after": after,
    "source_tests_prepost_exact": before == after, "exit_code": int(code),
    "wall_seconds": time.monotonic() - started,
    "safe_environment_fields": {k: os.environ.get(k) for k in
        ("PATH", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH")},
    "limits": [
        "Not a pure isolated-venv interpreter: the existing global pytest is only a test harness.",
        "Every observed loaded orze.* implementation comes from the fresh installed Core-only site and is compared with both wheel bytes and fixed Git blobs.",
        "No local src import is allowed. Explicit -o overrides pyproject pythonpath, and --confcutdir=tests excludes only the root conftest which unconditionally inserts local src; the unchanged tests/conftest remains active.",
        "No package installation, environment mutation, real license, provider, GPU or production work is performed.",
        "Actual fixture CLI invocations share the pytest interpreter; original tests are unchanged."
    ]
}
(ROOT / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(json.dumps({"report": str(ROOT / "report.json"), "report_sha256": sha((ROOT / "report.json").read_bytes()),
                  "exit_code": int(code), "loaded_orze_modules": len(loaded),
                  "unique_loaded_orze_files": len(git_hashes), "source_test_count": len(before),
                  "source_tests_prepost_exact": before == after, "wheel_sha256": EXPECTED_WHEEL,
                  "orze_file": orze.__file__}, sort_keys=True))
raise SystemExit(code)

