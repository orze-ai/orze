"""Lifecycle process discovery must identify only the Orze daemon."""

import re

from orze.lifecycle import _ORZE_PAT


def test_daemon_pattern_matches_python_module_process():
    command = "/opt/venv/bin/python3 -m orze.cli -c orze.yaml --gpus 0,1"

    assert re.search(_ORZE_PAT, command)


def test_daemon_pattern_rejects_shell_wrapper_command_text():
    command = (
        "/usr/bin/bash -lc export PYTHONPATH=/src; "
        "python -m orze.cli start -c orze.yaml"
    )

    assert re.search(_ORZE_PAT, command) is None
