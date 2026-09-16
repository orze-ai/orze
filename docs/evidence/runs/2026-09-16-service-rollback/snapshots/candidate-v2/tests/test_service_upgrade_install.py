"""Explicit upgrade installation through owned manager substitutes only."""
import copy
import json
from pathlib import Path
import sys

import pytest

from orze.service import host, install, runtime_contract, scoped, upgrade
from test_service_scoped import projects, property_map


def arguments(source, destination):
    return ['--source-service-config', str(source), '--service-config', str(destination),
            '--request-id', 'install-upgrade', '--backup', '/fixture/pinned-backup',
            '--manifest-sha256', 'a' * 64, '--install']


@pytest.mark.parametrize('fault', ['none', 'prepare', 'contract', 'timer', 'unknown-stop', 'collision'])
def test_explicit_install_preserves_prepared_transition_and_selected_units(projects, tmp_path, monkeypatch, fault):
    old, unrelated = projects
    source = Path(old['service_config_file'])
    source.write_text(json.dumps(old))
    original = source.read_bytes()
    destination = source.with_name('upgraded.json')
    units = tmp_path / 'units'
    units.mkdir()
    other_units = scoped.render_units(unrelated)
    for name, raw in other_units.items():
        (units / name).write_text(raw)
    prepared = {**old, 'service_config_file': str(destination)}
    main, timer, watchdog = scoped.unit_names(prepared)
    collision = units / main
    if fault == 'collision':
        collision.write_text('existing installation')
    events = []

    def prepare(*args):
        events.append(('prepare', *map(str, args)))
        assert Path.cwd() == Path(old['workdir'])
        if fault == 'prepare':
            raise RuntimeError('injected unconfirmed source')
        host._create(destination, prepared)
        return {'kind': 'prepared_upgrade', 'service_config': str(destination)}

    def checked(*args):
        events.append(args)
        if fault in ('timer', 'unknown-stop') and args == ('enable', '--now', timer):
            raise RuntimeError('injected timer uncertainty')

    effective = property_map(prepared)
    if fault == 'contract':
        effective[main]['ExecStart'] = property_map(unrelated)[scoped.unit_names(unrelated)[0]]['ExecStart']

    monkeypatch.setattr(upgrade, 'prepare', prepare)
    monkeypatch.setattr(install, '_SYSTEMD_DIR', units)
    monkeypatch.setattr(scoped, '_checked', checked)
    monkeypatch.setattr(scoped, '_wait_ready', lambda svc: events.append(('ready', svc['service_config_file'])))
    monkeypatch.setattr(runtime_contract, '_systemd_properties', lambda unit, **kwargs: copy.deepcopy(effective[unit]))
    monkeypatch.setattr(host, 'request', lambda path, op: events.append((op, path)) or
                        {'kind': 'unknown' if fault == 'unknown-stop' else 'stopped'})
    cwd = Path.cwd()
    assert upgrade.main(arguments(source, destination)) == (0 if fault == 'none' else 75)
    assert Path.cwd() == cwd and source.read_bytes() == original
    assert all((units / name).read_text() == raw for name, raw in other_units.items())
    assert destination.exists() == (fault != 'prepare')
    if destination.exists():
        assert json.loads(destination.read_bytes()) == prepared
    if fault == 'prepare':
        assert len(events) == 1
    elif fault == 'collision':
        assert len(events) == 1 and collision.read_text() == 'existing installation'
    else:
        assert all((units / name).read_text() == raw for name, raw in scoped.render_units(prepared).items())
        if fault == 'contract':
            assert events[1:] == [('daemon-reload',)]
        elif fault == 'none':
            assert events[1:] == [('daemon-reload',), ('enable', '--now', main),
                                  ('ready', str(destination)), ('enable', '--now', timer)]
        elif fault == 'timer':
            assert events[-3:] == [('stop', str(destination)), ('disable', '--now', timer), ('disable', '--now', main)]
        else:
            assert events[-1] == ('stop', str(destination)) and not any(e[0] == 'disable' for e in events)
    (tmp_path / 'install-result.json').write_text(json.dumps({'fault': fault, 'events': events,
        'source': str(source), 'destination': str(destination), 'units': sorted(p.name for p in units.iterdir())}, indent=2))


def test_install_rejects_process_method_before_preparation(projects, monkeypatch):
    svc = {**projects[0], 'method': 'process'}
    source = Path(svc['service_config_file'])
    source.write_text(json.dumps(svc))
    target = source.with_name('not-created.json')
    monkeypatch.setattr(upgrade, 'prepare', lambda *args: pytest.fail('process method reached preparation'))
    assert upgrade.main(arguments(source, target)) == 75 and not target.exists()


@pytest.mark.parametrize('requested', [False, True])
def test_cli_forwards_optional_install(monkeypatch, requested):
    from orze import cli
    expected = arguments('/source.json', '/destination.json')
    if not requested:
        expected.pop()
    calls = []
    monkeypatch.setattr(upgrade, 'main', lambda argv: calls.append(argv) or 75)
    monkeypatch.setattr(sys, 'argv', ['orze', 'service', 'upgrade', *expected])
    assert cli.main() == 75 and calls == [expected]
