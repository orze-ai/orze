"""Run unchanged product tests against installed Core under external pytest.

Arguments: REPO PACKAGE_ROOT FIXED_CORE_COMMIT OUTPUT.
No install, provider, licensing or production service actions are performed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from run_frozen import fingerprint

if not __debug__:
    raise SystemExit("installed validation requires assertions enabled")

REPO, PACKAGE = (Path(x).resolve() for x in sys.argv[1:3])
COMMIT, OUT = sys.argv[3], Path(sys.argv[4]).resolve()
assert not OUT.exists()
OUT.mkdir(parents=True)
sha = lambda raw: hashlib.sha256(raw).hexdigest()
package_report = json.loads((PACKAGE / "review.json").read_bytes())
assert package_report["status"] == "passed"
core = package_report["packages"]["core"]
assert core["commit"] == COMMIT
WHEEL = Path(core["wheel"]).resolve()
EXPECTED_WHEEL = core["sha256"]
assert sha(WHEEL.read_bytes()) == EXPECTED_WHEEL
SITE, = (PACKAGE / "core-only-venv/lib").glob("python*/site-packages")
assert not (SITE / "orze_pro").exists()
sys.path.insert(0, str(SITE))
sys.path.insert(1, str(REPO / "tests"))
assert str(REPO / "src") not in sys.path

import pytest
import orze

loaded, git_hashes, events, paths = {}, {}, [], []
with zipfile.ZipFile(WHEEL) as archive:
    wheel_hashes = {name: sha(archive.read(name)) for name in archive.namelist()
                    if name.startswith("orze/") and not name.endswith("/")}


def audit(label):
    paths.append({"label": label, "sys_path": list(sys.path)})
    assert str(REPO / "src") not in sys.path
    assert not any(name == "orze_pro" or name.startswith("orze_pro.") for name in sys.modules)
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
            git_hashes[member] = sha(subprocess.check_output(
                ["git", "show", COMMIT + ":src/" + member], cwd=REPO))
        assert actual == git_hashes[member], (name, "Git mismatch")
        loaded[name] = {"file": str(path), "wheel_member": member, "sha256": actual}


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


audit("before_pytest")
before = fingerprint(REPO)
self_sha = sha(Path(__file__).read_bytes())
test_files = ["test_cpu_proposal_paging_product_review.py", "test_cpu_budget_scan.py",
              "test_idea_ingress_cost.py",
              "test_idea_ingress_contract.py::test_stale_id_snapshot_cannot_replace_a_concurrent_winner"]
args = ["-q", "--confcutdir", str(REPO / "tests"), "-o", "pythonpath=" + str(SITE),
        *[str(REPO / "tests" / name) for name in test_files],
        "--tb=short", "-rs", "-p", "no:cacheprovider", "--basetemp", str(OUT / "pytest"),
        "--junitxml", str(OUT / "junit.xml")]
started = time.monotonic()
code = pytest.main(args, plugins=[Audit()])
audit("after_pytest")
after = fingerprint(REPO)
assert before == after and self_sha == sha(Path(__file__).read_bytes())
report = {"schema": 1, "classification": "installed Core payload under external global pytest harness",
          "argv": [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
          "pytest_arguments": args, "pytest_module": pytest.__file__,
          "actual_interpreter": sys.executable, "python_version": sys.version,
          "installed_core_only_site": str(SITE), "orze_file": orze.__file__,
          "core_commit": COMMIT, "wheel": str(WHEEL), "wheel_sha256": EXPECTED_WHEEL,
          "loaded_orze_modules": loaded, "sys_path_observations": paths,
          "events": events, "source_test_before": before, "source_test_after": after,
          "source_tests_prepost_exact": before == after, "exit_code": int(code),
          "launcher_sha256": self_sha, "wall_seconds": time.monotonic() - started,
          "safe_environment_fields": {k: os.environ.get(k) for k in
              ("PATH", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH")},
          "limits": ["Not a pure isolated-venv interpreter; existing global pytest is the external test harness.",
                     "Every observed loaded orze.* implementation must match fresh installed wheel and fixed Git bytes.",
                     "Root conftest is excluded because it inserts local src; unchanged tests/conftest stays active.",
                     "No provider, GPU, Pro import, license or production service work.",
                     "Metadata-only budget/ingress cases do not become worker evidence by running from a wheel."]}
(OUT / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(json.dumps({"report": str(OUT / "report.json"), "exit_code": int(code),
                  "loaded_orze_modules": len(loaded), "source_test_files": len(before),
                  "source_tests_prepost_exact": before == after, "wheel_sha256": EXPECTED_WHEEL,
                  "report_sha256": sha((OUT / "report.json").read_bytes())}, sort_keys=True))
raise SystemExit(code)
