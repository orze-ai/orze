"""Validation recorder, not a product runner. Outputs with Pro inventories stay private.

Usage: python run_frozen.py REPO OUTPUT [--peer REPO] -- COMMAND ...
No source/test changes, environment dumping, license or provider access by this recorder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(repo):
    paths = {p for folder in ("src", "tests", "examples")
             for p in (repo / folder).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts
             and p.suffix not in {".pyc", ".pyo"}}
    paths.update(p for p in (repo / "pyproject.toml", repo / "setup.cfg", repo / "requirements.txt")
                 if p.is_file())
    return {str(p.relative_to(repo)): sha(p.read_bytes()) for p in sorted(paths)}


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def main():
    split = sys.argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--peer", type=Path)
    args = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1:]
    assert command
    repo, output = args.repository.resolve(), args.output.resolve()
    roots = {"primary": repo}
    if args.peer:
        roots["peer"] = args.peer.resolve()
    assert not output.exists()
    output.mkdir(parents=True)
    before = {key: fingerprint(root) for key, root in roots.items()}
    save(output / "before.json", before)
    start = time.time()
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        result = subprocess.run(command, cwd=repo, stdout=stdout, stderr=stderr)
    after = {key: fingerprint(root) for key, root in roots.items()}
    save(output / "after.json", after)
    report = {"repositories": {key: str(root) for key, root in roots.items()},
              "command": command,
              "environment": {k: os.environ.get(k) for k in
                              ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")},
              "started_unix": start, "finished_unix": time.time(), "exit_code": result.returncode,
              "frozen": before == after, "before": before, "after": after,
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                        for p in sorted(output.iterdir()) if p.is_file()}}
    save(output / "run.json", report)
    print(json.dumps({key: report[key] for key in
                      ("repositories", "exit_code", "frozen", "started_unix", "finished_unix")}), flush=True)
    raise SystemExit(result.returncode if before == after else 99)


if __name__ == "__main__":
    main()
