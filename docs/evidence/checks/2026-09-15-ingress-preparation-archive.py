"""Create-only archive of completed query experiments, differential and CPU run."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("archive_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-archive.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-ingress-preparation"
helper.ROOTS = {
    "benchmark_v1": "/tmp/orze-ingress-preparation-benchmark-v1",
    "benchmark_v2": "/tmp/orze-ingress-preparation-benchmark-v2",
    "differential": "/tmp/orze-ingress-preparation-differential-v1",
    "product": "/tmp/orze-ingress-preparation-product-v1",
    "prototype": str(ROOT.parent / "ingress-preparation-prototype"),
}

if __name__ == "__main__":
    helper.main()
