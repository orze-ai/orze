"""Reuse the full mixed-metadata differential with this slice's committed baseline."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("differential", ROOT / "docs/evidence/checks/2026-09-15-dedup-candidate-query-differential.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT = ROOT / "docs/evidence/runs/2026-09-15-status-merge-queries"
helper.BASE = "f9d69e24c3a31569cb8235d617366b5d5a8894f5"

if __name__ == "__main__":
    helper.run(Path(sys.argv[1]).resolve())
