"""Create-only archive of completed query experiments, differential and CPU run."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("archive_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-archive.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-status-merge-queries"
helper.ROOTS = {
    "source_v1": "/tmp/orze-status-merge-source-v1",
    "source_v2": "/tmp/orze-status-merge-source-v2",
    "source_memory": "/tmp/orze-status-merge-source-memory-v1",
    "source_allocator": "/tmp/orze-status-merge-source-allocator-v1",
    "admission_v1": "/tmp/orze-status-merge-admission-v1",
    "admission_v2": "/tmp/orze-status-merge-admission-v2",
    "differential": "/tmp/orze-status-merge-differential-v1",
    "product": "/tmp/orze-status-merge-product-v1",
    "prototype": str(ROOT.parent / "identity-merge-prototype"),
}

if __name__ == "__main__":
    helper.main()
