"""Create-only archives of terminal source, cost and worker fixtures."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("archive_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-archive.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-sidecar-prefix"
helper.ROOTS = {
    "benchmark_v1": "/tmp/orze-sidecar-prefix-benchmark-v1",
    "benchmark_v2": "/tmp/orze-sidecar-prefix-benchmark-v2",
    "benchmark_v3": "/tmp/orze-sidecar-prefix-benchmark-v3",
    "one_file_v1": "/tmp/orze-sidecar-prefix-one-file-v1",
    "one_file_v2": "/tmp/orze-sidecar-prefix-one-file-v2",
    "one_file_v3": "/tmp/orze-sidecar-prefix-one-file-v3",
    "memory": "/tmp/orze-sidecar-prefix-memory-v1",
    "product": "/tmp/orze-sidecar-prefix-product-v1",
}

if __name__ == "__main__":
    helper.main()
