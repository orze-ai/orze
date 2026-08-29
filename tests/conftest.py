"""Shared fixtures for orze tests."""
import contextlib
import os
import shutil
import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _forbid_live_gpu_ownership_queries(monkeypatch):
    """Unit tests use process-scoped leases and never inspect the host fleet."""
    from orze.core.gpu_lease import gpu_execution_lease as real_gpu_lease

    monkeypatch.setattr(
        "orze.core.gpu_lease.assert_gpu_scope_idle",
        lambda gpu_ids: {
            "physical_scope": sorted(gpu_ids),
            "compute_processes": 0,
            "accelerator_access": "none",
            "accelerator_compute_access": "none",
        },
    )

    @contextlib.contextmanager
    def isolated_gpu_lease(gpu, *, require_idle=False):
        if gpu is None or gpu < 0:
            mapped_gpu = gpu
        else:
            # Separate concurrently sharded pytest processes while retaining
            # the real lease/file-descriptor semantics under test by launchers.
            mapped_gpu = 800_000 + (os.getpid() % 10_000) * 100 + gpu
        with real_gpu_lease(
                mapped_gpu, require_idle=require_idle) as lease_fds:
            yield lease_fds

    monkeypatch.setattr(
        "orze.engine.launcher.gpu_execution_lease", isolated_gpu_lease,
    )
    monkeypatch.setattr(
        "orze.engine.evaluator.gpu_execution_lease", isolated_gpu_lease,
    )


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal orze project directory and chdir into it."""
    orig_dir = os.getcwd()
    os.chdir(tmp_path)

    # Minimal orze.yaml
    (tmp_path / "orze.yaml").write_text(textwrap.dedent("""\
        train_script: train.py
        ideas_file: ideas.md
        results_dir: results
        python: python3
    """))

    # Minimal ideas.md
    (tmp_path / "ideas.md").write_text(textwrap.dedent("""\
        # Ideas

        ## idea-0001: Baseline
        - **Priority**: high

        ```yaml
        learning_rate: 0.001
        epochs: 5
        ```
    """))

    # configs directory
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("seed: 42\nnoise: 0.1\n")

    # results directory
    (tmp_path / "results").mkdir()

    yield tmp_path

    os.chdir(orig_dir)


@pytest.fixture
def write_train_script(tmp_project):
    """Write the baseline train.py into the tmp_project."""
    from orze.cli_demo import BASELINE_TRAIN_PY

    (tmp_project / "train.py").write_text(BASELINE_TRAIN_PY.strip() + "\n")
    return tmp_project
