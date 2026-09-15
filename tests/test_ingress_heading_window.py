"""Keep primary heading scan temporary memory below a fixed fixture budget."""
import json
import os
import subprocess
import sys


def test_fixed_20000_primary_heading_peak_stays_below_five_mib():
    # Input and returned block spans stay fully resident. This checks only the
    # scan of this fixed <4 MiB source, not source reading, admission or RSS.
    script = r'''
import gc, json, tracemalloc
from orze.engine.idea_ingress import _blocks
text = '# Retain this preamble\n' + ''.join(
    f'## idea-{i:06d}: Proposal {i}\n```yaml\nx: {i}\n```\n'
    + ('## Unknown heading\nKeep this text.\n' if i % 7 == 0 else '')
    for i in range(20000))
assert len(text.encode()) < 4 * 1024 * 1024
gc.collect();tracemalloc.start()
blocks = _blocks(text)
current,peak = tracemalloc.get_traced_memory();tracemalloc.stop()
assert len(blocks) == 22858
assert sum(key is not None for key,_,_ in blocks) == 20000
assert blocks[0][0] == 'idea-000000' and blocks[-2][0] == 'idea-019999'
assert blocks[-1][0] is None and blocks[-1][2] == len(text)
print(json.dumps({'peak':peak,'current':current,'blocks':len(blocks),'source_bytes':len(text.encode())}))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, check=True, env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    measured = json.loads(result.stdout)
    assert measured['peak'] < 5 * 1024 * 1024, measured
