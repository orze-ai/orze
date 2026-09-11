"""New request-publication mechanisms; no shutdown ACK is manufactured."""
import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import orze.core.fs as fs
import orze.lifecycle as lifecycle
from orze.core.control_outcome import StopOutcome


@pytest.fixture
def publication(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    state = SimpleNamespace(
        results=results, cfg={"results_dir": str(results)},
        syncs=[], selected_fds=set(), writes=0, injected=0, fault=None,
    )
    real_open, real_close = os.open, os.close
    real_write, real_fsync = os.write, os.fsync
    real_fstat = os.fstat
    root_identity = (results.stat().st_dev, results.stat().st_ino)

    def open_fd(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        path = Path(path)
        if (path.parent == results and path.name.startswith(".orze_stop_all.")
                and path.name.endswith(".tmp")):
            state.selected_fds.add(fd)
        return fd

    def close_fd(fd):
        state.selected_fds.discard(fd)
        return real_close(fd)

    def write(fd, data):
        if state.fault == "partial_second_marker" and fd in state.selected_fds:
            state.writes += 1
            if state.writes == 1:
                return real_write(fd, data[:1])
            state.injected += 1
            raise OSError(errno.EIO, "synthetic partial marker write")
        return real_write(fd, data)

    def sync(fd):
        info = real_fstat(fd)
        identity = (info.st_dev, info.st_ino)
        if (state.fault == "directory_after_second_publish"
                and identity == root_identity
                and (results / ".orze_stop_all").exists()):
            state.injected += 1
            raise OSError(errno.EIO, "synthetic request-directory fsync failure")
        real_fsync(fd)
        state.syncs.append((identity, stat.S_ISDIR(info.st_mode)))

    os_double = SimpleNamespace(**vars(os))
    os_double.open, os_double.close = open_fd, close_fd
    os_double.write, os_double.fsync = write, sync
    for name in ("kill", "killpg", "getpgid", "listdir", "execv"):
        setattr(os_double, name,
                lambda *args, **kwargs: pytest.fail("no process operation"))
    monkeypatch.setattr(fs, "os", os_double)
    monkeypatch.setattr(lifecycle, "os", os_double)
    process_double = SimpleNamespace(
        run=lambda *args, **kwargs: pytest.fail("no process discovery"),
        Popen=lambda *args, **kwargs: pytest.fail("no controller launch"),
    )
    monkeypatch.setattr(lifecycle, "subprocess", process_double)
    return state


def test_normal_request_syncs_both_markers_but_is_not_confirmed(publication):
    outcome = lifecycle.do_stop(publication.cfg)

    assert type(outcome) is StopOutcome
    assert outcome.status == "requested"
    markers = [publication.results / name
               for name in (".orze_disabled", ".orze_stop_all")]
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in markers}
    assert identities <= {identity for identity, directory in publication.syncs
                          if not directory}
    root_info = publication.results.stat()
    assert ((root_info.st_dev, root_info.st_ino), True) in publication.syncs
    assert markers[0].read_text() == "Controller stop requested; closure unconfirmed"
    assert markers[1].read_bytes() == b"kill"


def test_partial_second_marker_write_is_hold_not_requested(publication):
    publication.fault = "partial_second_marker"

    outcome = lifecycle.do_stop(publication.cfg)

    assert publication.writes == 2
    assert publication.injected == 1
    assert outcome.status == "hold"
    assert (publication.results / ".orze_disabled").is_file()
    assert not (publication.results / ".orze_stop_all").exists()


def test_directory_sync_failure_after_publish_is_hold(publication):
    publication.fault = "directory_after_second_publish"

    outcome = lifecycle.do_stop(publication.cfg)

    assert publication.injected == 1
    assert outcome.status == "hold"
    assert (publication.results / ".orze_disabled").is_file()
    assert (publication.results / ".orze_stop_all").read_bytes() == b"kill"


def test_changed_readback_cannot_confirm_request(publication, monkeypatch):
    original = fs.atomic_write
    changed = []

    def publish_then_change(path, content):
        original(path, content)
        if Path(path).name == ".orze_stop_all":
            Path(path).write_bytes(b"different request bytes")
            changed.append(str(path))

    monkeypatch.setattr(fs, "atomic_write", publish_then_change)
    outcome = lifecycle.do_stop(publication.cfg)

    assert changed == [str(publication.results / ".orze_stop_all")]
    assert outcome.status == "hold"
    assert (publication.results / ".orze_disabled").is_file()
    assert (publication.results / ".orze_stop_all").read_bytes() == b"different request bytes"


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_linked_marker_refused_without_changing_target(publication, link_kind):
    target = publication.results.parent / "unrelated-control-file"
    target.write_bytes(b"operator-owned bytes")
    marker = publication.results / ".orze_stop_all"
    if link_kind == "symlink":
        marker.symlink_to(target)
    else:
        os.link(target, marker)
    target_info, marker_info = target.stat(), marker.lstat()

    outcome = lifecycle.do_stop(publication.cfg)

    assert outcome.status == "hold"
    assert target.read_bytes() == b"operator-owned bytes"
    assert (target.stat().st_dev, target.stat().st_ino) == (
        target_info.st_dev, target_info.st_ino)
    assert (marker.lstat().st_dev, marker.lstat().st_ino) == (
        marker_info.st_dev, marker_info.st_ino)
    assert (publication.results / ".orze_disabled").is_file()
