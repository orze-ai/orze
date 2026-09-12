"""The existing wheel data contract requires a real operations guide."""
from pathlib import Path


def test_declared_operations_guide_exists_in_package_source():
    package = Path(__file__).resolve().parents[1] / "src" / "orze"
    assert (package / "SKILL.md").is_file(), "declared orze/SKILL.md is missing from package source"
