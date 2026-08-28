import concurrent.futures

from orze.core.fs import atomic_create, atomic_write
from orze.engine.scheduler import claim


def test_atomic_create_has_one_complete_winner(tmp_path):
    path = tmp_path / "winner.json"
    payloads = [f'{{"worker": {index}, "padding": "{index * "x"}"}}\n'
                for index in range(16)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(
            lambda payload: atomic_create(path, payload), payloads
        ))

    assert sum(results) == 1
    assert path.read_text() in payloads
    assert list(tmp_path.glob(".*.tmp")) == []


def test_concurrent_claim_has_exactly_one_winner(tmp_path):
    results_dir = tmp_path / "results"
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(claim, "idea-race", results_dir, 4)
            for _ in range(16)
        ]
    assert sum(bool(future.result()) for future in futures) == 1
    assert (results_dir / "idea-race" / "claim.json").is_file()


def test_concurrent_atomic_writes_never_share_temporary_name(tmp_path):
    path = tmp_path / "latest.json"
    payloads = [f'{{"writer": {index}}}\n' for index in range(32)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda payload: atomic_write(path, payload), payloads))

    assert path.read_text() in payloads
    assert list(tmp_path.glob("*.tmp")) == []
