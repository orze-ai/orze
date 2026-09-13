"""Validation-only recorder; never a product runner or production deployment.

Usage: python run_frozen.py REPOSITORY OUTPUT_DIRECTORY -- PYTHON -m pytest ...
The caller supplies a new private output directory and all pytest arguments.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def fingerprint(repo):
    paths = {p for folder in ("src", "tests", "examples")
             for p in (repo / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.suffix not in {".pyc", ".pyo"}}
    paths.update(p for p in (repo / "pyproject.toml", repo / "setup.cfg") if p.is_file())
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    repo, output = map(lambda value: Path(value).resolve(), sys.argv[1:3])
    assert sys.argv[3] == "--" and not output.exists()
    command = sys.argv[4:]
    output.mkdir(parents=True)
    before = fingerprint(repo)
    start = time.time()
    (output / "before.json").write_text(json.dumps(before, sort_keys=True, indent=2) + "\n")
    with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr)
    after = fingerprint(repo)
    report = {"repository": str(repo), "command": command,
              "environment": {k: os.environ.get(k) for k in
                              ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")},
              "started_unix": start, "finished_unix": time.time(), "exit_code": result.returncode,
              "frozen": before == after, "before": before, "after": after,
              "files": {p.name: {"bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "run.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("repository", "exit_code", "frozen", "started_unix", "finished_unix")}), flush=True)
    raise SystemExit(result.returncode if before == after else 99)


if __name__ == "__main__":
    main()
