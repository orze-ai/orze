"""Owned workload bootstrap; checks the artifact before any research activity."""
import argparse
from pathlib import Path
import sys
import time

from .campaign import _bind, _sha, _write, describe
from .protocol import digest, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    before, after, status, capture_sha, partial_sha, error = None, None, 'failed', None, None, None
    try:
        request = read_json(args.request.read_bytes())
        if digest(request) != args.request_sha256:
            raise ValueError('campaign request changed before admission')
        _bind(request['protocol'], request['slot']['run_id'], request['inputs'], request['runtime'])
        before = describe()
        if before != request['runtime']:
            raise ValueError('campaign runtime differs from the admitted artifact')
        from .scheduling_campaign import run
        capture = run(request, args.output)
        _write(args.output / 'capture.json', capture)
        capture_sha = _sha((args.output / 'capture.json').read_bytes())
        after = describe()
        if before != after:
            raise ValueError('campaign runtime changed during execution')
        status = 'completed'
    except Exception as exc:
        error = type(exc).__name__
        if before is not None:
            try:
                after = describe()
            except Exception:
                pass
        partial = args.output / 'project/partial-capture.json'
        if partial.is_file() and not partial.is_symlink():
            partial_sha = _sha(partial.read_bytes())
    _write(args.output / 'worker.json', {'schema': 1, 'request_sha256': args.request_sha256,
        'before': before, 'after': after, 'status': status, 'error': error,
        'started_monotonic': start, 'finished_monotonic': time.monotonic(),
        'capture_sha256': capture_sha, 'partial_capture_sha256': partial_sha})
    if error:
        print('campaign workload failed: ' + error, file=sys.stderr)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
