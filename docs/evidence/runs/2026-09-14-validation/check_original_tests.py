"""Read-only byte equality audit of every test tracked at a named baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def main():
    repo, baseline = Path(sys.argv[1]).resolve(), sys.argv[2]
    commit = git(repo, "rev-parse", baseline + "^{commit}").decode().strip()
    names = git(repo, "ls-tree", "-rz", "--name-only", commit, "--", "tests").split(b"\0")
    results = {}
    for name in filter(None, names):
        relative = name.decode()
        original = git(repo, "show", commit + ":" + relative)
        current = repo / relative
        raw = current.read_bytes() if current.is_file() else None
        results[relative] = {"original_sha256": hashlib.sha256(original).hexdigest(),
                             "current_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                             "exact": original == raw}
    assert results
    report = {"baseline": commit, "repository": str(repo), "files": results,
              "original_test_files": len(results),
              "all_exact": all(row["exact"] for row in results.values())}
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_exact"] else 1)


if __name__ == "__main__":
    main()
