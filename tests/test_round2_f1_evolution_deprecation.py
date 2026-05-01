"""Round-2 F1: top-level ``evolution:`` migrates with DeprecationWarning."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_evolution_migrates(tmp_path):
    cfg_path = tmp_path / "orze.yaml"
    cfg_path.write_text(
        "train_script: train.py\n"
        "evolution:\n"
        "  enabled: true\n"
        "  max_attempts_per_plateau: 3\n",
        encoding="utf-8",
    )
    (tmp_path / "train.py").write_text("# stub", encoding="utf-8")

    from orze.core.config import load_project_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_project_config(str(cfg_path))

    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)
           and "evolution" in str(w.message).lower()]
    assert dep, f"expected DeprecationWarning, got: {[str(w.message) for w in caught]}"

    roles = cfg.get("roles") or {}
    ce = roles.get("code_evolution") or {}
    assert ce.get("enabled") is True
    assert ce.get("max_attempts_per_plateau") == 3


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_evolution_migrates(Path(td))
        print("test_evolution_migrates OK")
