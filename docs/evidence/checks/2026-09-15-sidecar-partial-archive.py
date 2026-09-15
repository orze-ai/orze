"""Create-only archive of completed query experiments, differential and CPU run."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("archive_helper", ROOT / "docs/evidence/checks/2026-09-15-ingress-archive.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-sidecar-partial-prefix"
helper.ROOTS = {
    **{name.replace("-", "_"): "/tmp/orze-sidecar-partial-" + name for name in (
        "many-v1", "one-v1", "many-v2", "one-v2", "many-memory", "one-memory")},
    "differential": "/tmp/orze-sidecar-partial-differential-v1",
    "product": "/tmp/orze-sidecar-partial-product-v1",
    "prototype": str(ROOT.parent / "sidecar-partial-prefix-prototype"),
    "prototype_probe": "/tmp/orze-sidecar-partial-probe-v1",
}

if __name__ == "__main__":
    helper.main()
