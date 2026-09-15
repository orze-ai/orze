"""Execute the new payload-read contract against the preserved original ingress."""
import importlib.util
from pathlib import Path
import pytest
from orze.engine import idea_ingress

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "docs/evidence/runs/2026-09-15-sidecar-prefix/baseline/idea_ingress.py"
spec = importlib.util.spec_from_file_location("original_ingress", path)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
# Keep the test's fresh-read counter attached to the original implementation.
original._read_source = lambda path: idea_ingress._read_source(path)
idea_ingress.ingest_ideas_source = original.ingest_ideas_source
raise SystemExit(pytest.main(["-q", "tests/test_sidecar_prefix_continuation.py::test_continuation_does_not_reread_preceding_sidecar_payloads",
    "--tb=short", "-p", "no:cacheprovider", "--basetemp=/tmp/osp-br1",
    "--junitxml=docs/evidence/runs/2026-09-15-sidecar-prefix/baseline-regression/junit.xml"]))
