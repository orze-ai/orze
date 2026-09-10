"""Replacing pathlib glob must not broaden directory-only selections."""
from orze.engine.scheduler import run_cleanup


def test_recursive_directory_only_patterns_do_not_become_file_deletion(tmp_path):
    results = tmp_path / "results"
    folder = results / "idea-glob-compatibility"
    nested = folder / "nested"
    nested.mkdir(parents=True)
    flat = folder / "scratch.tmp"
    deep = nested / "scratch.tmp"
    flat.write_bytes(b"not selected by a directory-only glob")
    deep.write_bytes(b"also not selected")
    patterns = ["**", "nested/**", "**/**"]
    assert not any(path.is_file() for pattern in patterns for path in folder.glob(pattern))

    run_cleanup(results, {"cleanup": {"patterns": patterns}})

    assert flat.is_file() and deep.is_file(), "new glob matcher broadened deletion selection"
    assert flat.read_bytes() == b"not selected by a directory-only glob"
    assert deep.read_bytes() == b"also not selected"
