"""Create-only archives of terminal repair benchmarks and actual CPU project."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("archive_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-archive.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-identity-repair-staging"
helper.ROOTS = {
    "benchmark_v1": "/tmp/orze-identity-repair-benchmark-v1",
    "benchmark_v2": "/tmp/orze-identity-repair-benchmark-v2",
    "memory": "/tmp/orze-identity-repair-memory-v1",
    "product": "/tmp/orze-identity-repair-product-v1",
}

if __name__ == "__main__":
    helper.main()
