"""Preserve this pilot's original outputs; never copy credentials or Pro sources."""
import hashlib
import json
from pathlib import Path
import tarfile
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'orze/docs/evidence/runs/2026-09-16-real-memory-pilot'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    secrets = [v.encode() for v in dotenv_values('/hot-data/fsx/workspace/erik/auto-research-1.7b/.env').values()
               if v and len(v) >= 12]
    sources = [ROOT / name for name in ('real-memory-pilot-20260916', 'real-memory-pilot-20260916-v2',
               'real-memory-pilot-20260916-fixed', 'real-memory-pilot-20260916-admission')]
    script_names = ('real_memory_pilot_v1.py', 'real_memory_pilot_v2.py', 'real_memory_haiku_v1.py',
        'real_memory_after_fix_v1.py', 'real_memory_after_admission_v1.py',
        'diagnose_memory_response_v1.py', 'diagnose_memory_response_v2.py',
        'analyze_real_memory_v1.py', 'analyze_real_memory_v2.py', 'analyze_real_memory_v3.py',
        'analyze_real_memory_v4.py', 'package_real_memory_v1.py')
    logs = ('real-memory-timeout-tests-v1.log', 'memory-fenced-response-tests-v1.log',
        'campaign-duplicates-before.log', 'campaign-duplicates-before-v2.log', 'campaign-duplicates-after.log',
        'duplicate-diagnosis-v1.json', 'real-memory-after-fix-launch.log', 'real-memory-after-admission-launch.log')
    files = [p for source in sources for p in sorted(source.rglob('*')) if p.is_file()]
    files += [ROOT / name for name in (*script_names, *logs)]
    for path in files:
        assert path.is_file() and not path.is_symlink(), path.name
        assert not any(x in path.parts for x in ('.env', '.git', '__pycache__', 'orze_pro')), path.name
        raw = path.read_bytes()
        assert not any(secret in raw for secret in secrets), 'credential detected in ' + str(path.relative_to(ROOT))
    DEST.mkdir(parents=True, exist_ok=False)
    archives = []
    for source in sources:
        destination = DEST / 'raw' / (source.name + '.tar.gz')
        destination.parent.mkdir(exist_ok=True)
        with tarfile.open(destination, 'x:gz') as archive:
            archive.add(source, arcname=source.name)
        entries = []
        with tarfile.open(destination, 'r:gz') as archive:
            for member in archive:
                if not member.isfile():
                    assert member.isdir(), member.name
                    continue
                raw = archive.extractfile(member).read()
                assert raw == (ROOT / member.name).read_bytes(), member.name
                entries.append({'path': member.name, 'bytes': len(raw), 'sha256': sha(raw)})
        write(destination.with_suffix('.manifest.json'), entries)
        archives.append({'path': str(destination.relative_to(DEST)), 'files': len(entries),
                         'bytes': destination.stat().st_size, 'sha256': sha(destination.read_bytes())})
    for name in script_names:
        path = DEST / 'tools' / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for name in logs:
        path = DEST / 'checks' / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    for source, name in ((sources[1]/'independent-summary-v2.json', 'baseline-summary.json'),
                         (sources[2]/'independent-summary-v3.json', 'parser-fix-summary.json'),
                         (sources[3]/'independent-summary-v4.json', 'summary.json')):
        (DEST / name).write_bytes(source.read_bytes())
    write(DEST/'archives.json', archives)
    write(DEST/'files.json', {str(p.relative_to(DEST)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
                              for p in sorted(DEST.rglob('*')) if p.is_file()})
    print(json.dumps({'archives': archives, 'credential_scan_files': len(files), 'destination': str(DEST)}))


if __name__ == '__main__':
    main()
