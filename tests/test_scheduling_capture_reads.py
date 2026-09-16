"""A tiny pinned capture must not allocate the entire maximum file budget."""
import hashlib
import tracemalloc

import pytest

from examples.research_comparison import scheduling


def test_small_capture_has_small_python_allocation(tmp_path):
    path=tmp_path/'small.json';raw=b'{"value":1}'
    path.write_bytes(raw)
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        result=scheduling.read_capture(path,hashlib.sha256(raw).hexdigest())
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert result=={'value':1}
    assert peak < 1024*1024, 'A tiny file should stay below a generous 1 MiB reader allocation budget'


@pytest.mark.parametrize('mutation',['grow','shrink'])
def test_size_change_during_read_is_unavailable(tmp_path,monkeypatch,mutation):
    path=tmp_path/'changing.json';raw=b'{"value":1}'
    path.write_bytes(raw);fdopen=scheduling.os.fdopen
    class ChangingReader:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,count):
            path.write_bytes(raw+b' ' if mutation=='grow' else b'{}')
            return self.stream.read(count)
    monkeypatch.setattr(scheduling.os,'fdopen',lambda *a,**k:ChangingReader(fdopen(*a,**k)))
    with pytest.raises(ValueError,match='changed or digest mismatch'):
        scheduling.read_capture(path,hashlib.sha256(raw).hexdigest())


def test_file_size_limit_is_still_enforced(tmp_path,monkeypatch):
    path=tmp_path/'large.json';raw=b'{"padding":"'+b'a'*128+b'"}'
    path.write_bytes(raw);monkeypatch.setattr(scheduling,'MAX_CAPTURE_BYTES',128)
    with pytest.raises(ValueError,match='bounded regular file'):
        scheduling.read_capture(path,hashlib.sha256(raw).hexdigest())
