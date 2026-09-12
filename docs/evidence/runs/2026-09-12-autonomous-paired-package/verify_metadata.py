"""Second byte/RECORD verification and textual metadata export; no product import."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
CURRENT = Path("/hot-data/fsx/workspace/erik/orze-unlimited-validation-2026-09-12.lUanmfzX")
OLD = Path("/hot-data/fsx/workspace/erik/orze-release-candidate-6BUecCLP")

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    report = json.loads((ROOT / "review.json").read_bytes())
    assert report["status"] == "passed" and report["workers_started"] == 0
    assert len(report["commands"]) == 12 and all(c["exit_code"] == 0 for c in report["commands"])
    metadata_root = ROOT / "metadata"
    metadata_root.mkdir()
    sites, = (ROOT / "paired-venv/lib").glob("python*/site-packages")
    results = {}
    metadata_files = {}
    for name in ("core", "pro"):
        item = report["packages"][name]
        wheel = Path(item["wheel"])
        assert sha(wheel.read_bytes()) == item["sha256"]
        with tarfile.open(ROOT / (name + "-source.tar")) as tar:
            blobs = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        package = "orze" if name == "core" else "orze_pro"
        with zipfile.ZipFile(wheel) as z:
            members = {n: z.read(n) for n in z.namelist() if not n.endswith("/")}
        assert len(members) == len(set(members))
        package_members = {n for n in members if n.startswith(package + "/")}
        assert package_members == set(item["package_files"])
        assert {n for n in package_members if n.endswith(".py")} == {
            n[4:] for n in blobs if n.startswith("src/" + package + "/") and n.endswith(".py")}
        for relative in package_members:
            raw = members[relative]
            assert raw == blobs["src/" + relative] == (sites / relative).read_bytes()
            assert sha(raw) == item["package_files"][relative]
        checks = {}
        for kind, raw_record in (("wheel", members[item["record"]]),
                                 ("installed", (sites / item["record"]).read_bytes())):
            rows = list(csv.reader(io.StringIO(raw_record.decode())))
            assert len({r[0] for r in rows}) == len(rows)
            if kind == "wheel":
                assert {r[0] for r in rows} == set(members)
            for relative, hashed, size in rows:
                if relative == item["record"]:
                    assert hashed == size == ""
                else:
                    if kind == "wheel":
                        raw = members[relative]
                    else:
                        path = (sites / relative).resolve()
                        assert path.is_relative_to(ROOT / "paired-venv")
                        raw = path.read_bytes()
                    assert hashed == "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
                    assert size == str(len(raw))
            checks[kind] = {"rows": len(rows), "sha256": sha(raw_record)}
        assert checks["installed"]["sha256"] == report["installed_records"][name]["sha256"]
        directory = metadata_root / name
        directory.mkdir()
        (directory / "pyproject.toml").write_bytes(blobs["pyproject.toml"])
        distinfo = item["record"].rsplit("/", 1)[0]
        for relative, raw in members.items():
            if relative.startswith(distinfo + "/"):
                target = directory / "wheel" / relative[len(distinfo) + 1:]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                assert target.read_bytes() == raw
        for path in sorted((sites / distinfo).rglob("*")):
            if path.is_file():
                target = directory / "installed" / path.relative_to(sites / distinfo)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                assert target.read_bytes() == path.read_bytes()
        results[name] = {"commit": report[name + "_commit"], "wheel_sha256": item["sha256"],
            "git_wheel_installed_package_files": len(package_members),
            "python_files": item["python_files"], "resource_files": item["resource_files"],
            "record_checks": checks}
    for path in metadata_root.rglob("*"):
        if path.is_file():
            metadata_files[str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
    dependencies = json.loads((ROOT / "dependency-lock.json").read_bytes())
    original_dependencies = json.loads((OLD / "dependency-lock.json").read_bytes())
    expected = {d["name"]: d for d in original_dependencies if d["name"] not in ("orze", "orze-pro")}
    actual = {d["name"]: d for d in dependencies if d["name"] not in ("orze", "orze-pro")}
    assert set(actual) == set(expected) and len(actual) == 23
    for name, d in actual.items():
        assert all(d[k] == expected[name][k] for k in ("version", "sha256", "source_url"))
        assert sha(Path(d["offline_wheel"]).read_bytes()) == d["sha256"]
    original_core = json.loads((CURRENT / "installation-final/review.json").read_bytes())
    original_sites, = (CURRENT / "core-only-venv/lib").glob("python*/site-packages")
    for relative, digest in original_core["package_files"].items():
        assert sha((original_sites / relative).read_bytes()) == digest
    assert sha((CURRENT / "wheelhouse/orze-4.6.2-py3-none-any.whl").read_bytes()) == original_core["wheel_sha256"]
    assert sha((OLD / "wheelhouse/orze_pro-0.13.1-py3-none-any.whl").read_bytes()) == "f27d1ca0675ac2a76ea11f9ae2e42cd5dfead9246f31854bf652bf8eb1ea8b11"
    help_text = (ROOT / "logs/core-help-no-pro-chain.stdout").read_text()
    isolation = json.loads(next(line.split("=", 1)[1] for line in help_text.splitlines() if line.startswith("HELP_ISOLATION=")))
    assert isolation == {"core_help_exit": 0, "forbidden_events": [], "pro_imported": False,
                         "guard_replaces_no_product_function": True}
    result = {"passed": True, "packages": results, "metadata_files": metadata_files,
              "report_sha256": sha((ROOT / "review.json").read_bytes()),
              "unchanged_original_locked_dependency_wheels": 23,
              "original_6007_core_only_package_payload_unchanged": True,
              "original_da2_pro_wheel_unchanged": True, "help_isolation": isolation,
              "new_workers_started": 0, "product_imports_in_this_checker": 0,
              "scope": "Re-read wheel/Git/installed payload and all RECORD hashes; export only exact textual metadata.",
              "limitations": ["No licensed Pro runtime validation or production switch.",
                             "No demonstration of CephFS role-release support or wider filesystem compatibility."]}
    path = ROOT / "verification.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"passed": True, "packages": results, "metadata_files": len(metadata_files),
                      "report": str(path), "sha256": sha(path.read_bytes())}, sort_keys=True))

if __name__ == "__main__":
    main()

