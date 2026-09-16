"""Durable task retirement for GC bulk I/O outside the short effect guard.

Pending records deny fresh task effects. A record never grants process control,
authorizes recovery, or permits a different process to resume partial deletion.
Completed records are immutable; history uses the existing GC limit of 1024.
"""
import hashlib
import itertools
from pathlib import Path
import secrets


_TREE = '_gc_retirements'
_LIMIT = 1024


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _identity(path):
    from orze.engine.attempt_effect_receipts import _directory
    info = _directory(path)
    return info.st_dev, info.st_ino, info.st_mode


def _records(folder):
    from orze.engine.attempt_effect_receipts import _directory, _read, _decode
    root = Path(folder) / _TREE
    if not root.exists() and not root.is_symlink():
        return []
    before = _stamp(_directory(root))
    records = list(itertools.islice(root.iterdir(), _LIMIT + 1))
    if len(records) > _LIMIT:
        raise ValueError('gc_retirement_history_limit')
    witnesses = []
    for record in records:
        if len(record.name) != 32 or any(c not in '0123456789abcdef' for c in record.name):
            raise ValueError('gc_retirement_name_invalid')
        directory = _stamp(_directory(record))
        names = {p.name for p in itertools.islice(record.iterdir(), 3)}
        if names != {'intent.json', 'completed.json'}:
            raise ValueError('gc_retirement_pending')
        raw = _read(record / 'intent.json')
        intent = _decode(raw)
        if (set(intent) != {'version', 'task', 'token', 'journal', 'plan_sha256'}
                or type(intent['version']) is not int or intent['version'] != 1
                or intent['task'] != str(Path(folder).absolute()) or intent['token'] != record.name
                or type(intent['journal']) is not str or not Path(intent['journal']).is_absolute()
                or type(intent['plan_sha256']) is not str or len(intent['plan_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in intent['plan_sha256'])):
            raise ValueError('gc_retirement_intent_invalid')
        done = _read(record / 'completed.json')
        decoded = _decode(done)
        if type(decoded.get('version')) is not int or decoded != {'version': 1, 'intent_sha256': hashlib.sha256(raw).hexdigest()}:
            raise ValueError('gc_retirement_completion_invalid')
        witnesses.append((record, directory, raw, done))
    for record, directory, raw, done in witnesses:
        if (_stamp(_directory(record)) != directory or _read(record / 'intent.json') != raw
                or _read(record / 'completed.json') != done):
            raise ValueError('gc_retirement_history_changed')
    if _stamp(_directory(root)) != before:
        raise ValueError('gc_retirement_history_changed')
    return records


def require_quiet(folder):
    from orze.engine.attempt_effect_lock import AttemptEffectInDoubt
    try:
        _records(folder)
    except Exception as exc:
        raise AttemptEffectInDoubt('gc_retirement_unconfirmed') from exc


class Retirement:
    def __init__(self, folder, record, raw):
        self.folder, self.record, self.raw = folder, record, raw
        self.identities = {p: _identity(p) for p in (folder, record.parent, record)}

    def check(self):
        from orze.engine.attempt_effect_receipts import _read
        if (any(_identity(path) != expected for path, expected in self.identities.items())
                or _read(self.record / 'intent.json') != self.raw
                or {p.name for p in itertools.islice(self.record.iterdir(), 3)} != {'intent.json'}):
            raise ValueError('gc_retirement_owner_changed')

    def complete(self):
        from orze.engine.attempt_effect_receipts import _encoded, _publish
        self.check()
        _publish(self.record / 'completed.json', _encoded({
            'version': 1, 'intent_sha256': hashlib.sha256(self.raw).hexdigest()}))
        require_quiet(self.folder)


def prepare(lease, journal, plan_sha256):
    from orze.engine.attempt_effect_lock import require_effect_lease
    from orze.engine.attempt_effect_receipts import _encoded, _publish, _sync
    folder = lease.idea_dir
    require_effect_lease(lease, folder)
    records = _records(folder)
    if len(records) >= _LIMIT:
        raise ValueError('gc_retirement_history_limit')
    root = folder / _TREE
    root.mkdir(mode=0o700, exist_ok=True)
    _identity(root)
    _sync(folder)
    token = secrets.token_hex(16)
    record = root / token
    record.mkdir(mode=0o700)
    _sync(root)
    raw = _encoded({'version': 1, 'task': str(folder), 'token': token,
                    'journal': str(journal), 'plan_sha256': plan_sha256})
    _publish(record / 'intent.json', raw)
    result = Retirement(folder, record, raw)
    result.check()
    return result
