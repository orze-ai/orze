"""Run the bounded source-read requirement against the preserved old method."""
import importlib.util
from pathlib import Path
import pytest
from orze.idea_lake import IdeaLake

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "docs/evidence/runs/2026-09-15-identity-repair-staging/baseline/idea_lake.py"
spec = importlib.util.spec_from_file_location("old_identity_lake", path)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
IdeaLake._repair_admitted_config_hashes = old.IdeaLake._repair_admitted_config_hashes
raise SystemExit(pytest.main(["-q", "tests/test_config_identity_staging.py::test_source_rows_are_bounded_but_main_write_is_one_atomic_transaction",
    "--tb=short", "-p", "no:cacheprovider", "--basetemp=/tmp/oir-br1",
    "--junitxml=docs/evidence/runs/2026-09-15-identity-repair-staging/baseline-regression/junit.xml"]))
