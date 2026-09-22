import threading

import pytest

from orze.research.execution import feature_tensor, prepare_batch
from orze.research.source import parse_python_proposal


def test_ready_proposal_can_execute_before_other_preparation_finishes():
    slow_started = threading.Event()
    release = threading.Event()
    settled = threading.Event()

    def prepare(item):
        if item == "slow":
            slow_started.set()
            assert release.wait(5)
            settled.set()
        else:
            assert slow_started.wait(5)
        return item.upper()

    try:
        with prepare_batch(["slow", "fast"], prepare, workers=2) as ready:
            assert next(ready) == ("fast", "FAST")
            assert not settled.is_set()
            release.set()  # early delivery still joins the other paid callback
    finally:
        release.set()
    assert settled.is_set()


def test_prepare_failure_still_settles_other_started_work():
    settled = threading.Event()

    def prepare(item):
        if item == "fail":
            raise RuntimeError("provider failed")
        settled.set()
        return item

    with pytest.raises(RuntimeError, match="provider failed"):
        with prepare_batch(["fail", "other"], prepare, workers=2) as ready:
            list(ready)
    assert settled.is_set()
    with pytest.raises(ValueError, match="bound"):
        with prepare_batch([1, 2], prepare, workers=1):
            pass


def test_tensor_and_readonly_memmap_are_supported(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    path = tmp_path / "features.npy"
    np.save(path, np.arange(8, dtype=np.float16).reshape(2, 4))
    stored = np.load(path, mmap_mode="r")
    tensor = feature_tensor(stored, device="cpu", dtype=torch.bfloat16)
    assert tensor.tolist() == stored.tolist()
    tensor[0, 0] = 99
    assert stored[0, 0] == 0
    assert feature_tensor(tensor, device="cpu", dtype=tensor.dtype) is tensor
    trainable = torch.ones(2, requires_grad=True)
    feature_tensor(trainable, device="cpu", dtype=torch.float64).sum().backward()
    assert trainable.grad.tolist() == [1, 1]


METHOD = '"""Test a decoding hypothesis."""\ndef build(api): pass\ndef train(api, model): pass\ndef predict(api, model, features): pass'


@pytest.mark.parametrize("text", [METHOD, "```python\n" + METHOD + "\n```"])
def test_plain_python_without_executing_it(text):
    proposal = parse_python_proposal(text, complete=True)
    assert proposal == {"kind": "method", "source": METHOD}
    assert parse_python_proposal("raise RuntimeError('must not execute')\ndef analyze(api): pass", complete=True)["kind"] == "analysis"


@pytest.mark.parametrize("text,complete", [
    (METHOD, False), ("```python\n" + METHOD, True),
    (METHOD + "\ndef analyze(api): pass", True),
    (METHOD + "\ndef build(api): pass", True),
    ("def build(api): pass", True), ("", True),
    ("```python\n" + METHOD + "\n```\nExplanation", True),
    ("def analyze(api):\n", True),
])
def test_invalid_or_incomplete_source_is_rejected(text, complete):
    with pytest.raises((ValueError, SyntaxError)):
        parse_python_proposal(text, complete=complete)
