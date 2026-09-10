"""Independent real-filesystem boundaries for the new interruption split."""
import os
from pathlib import Path

import pytest

from orze.engine import resume
from test_interruption_prepared_publication import (
    api, prepare, publish, regular_case, resume_case, terminal,
)


def test_publication_readback_cannot_block_on_a_replaced_fifo(api, regular_case, monkeypatch):
    prepared = prepare(api, regular_case)
    output = regular_case[2] / "interruption.json"
    real_atomic_write = resume.atomic_write
    real_open = os.open

    def replace_with_fifo(path, text):
        result = real_atomic_write(path, text)
        if Path(path) == output:
            output.unlink()
            os.mkfifo(output)
        return result

    def avoid_hanging_test(path, flags, *args, **kwargs):
        if Path(path) == output and not flags & os.O_NONBLOCK:
            pytest.fail("readback would block indefinitely opening a writerless FIFO")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(resume, "atomic_write", replace_with_fifo)
    monkeypatch.setattr(os, "open", avoid_hanging_test)
    with pytest.raises(resume.ResumeValidationError):
        publish(api, regular_case, prepared)
    assert not terminal(regular_case).exists()


def test_nonmapping_idea_config_preserves_declared_script_fallback(api, regular_case):
    _, _, idea_dir, _, _, tp = regular_case
    tp.train_script = None
    (idea_dir / "idea_config.yaml").write_text("- a nonmapping legacy config\n")
    prepared = prepare(api, regular_case)
    payload = publish(api, regular_case, prepared)
    assert payload["resume_eligible"] is True
    assert payload["resume_reason"] == "verified"
    assert payload["checkpoint"]["kind"] == "file"
    assert terminal(regular_case).exists()
