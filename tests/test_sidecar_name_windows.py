"""Actual ingress checks for bounded name reuse and unchanged fresh-source rules."""
from pathlib import Path
import pytest
from orze.engine import idea_ingress, sidecar_prefix
from test_idea_ingress_contract import engine, _block
from test_sidecar_prefix_continuation import populate


def test_unchanged_namespace_reuses_one_window_for_three_pages(engine, monkeypatch):
    instance,cfg,source=engine
    side=populate(source)
    original=sidecar_prefix.os.scandir
    scans=[]
    def counted(path):
        if Path(path)==side:scans.append(path)
        return original(path)
    monkeypatch.setattr(sidecar_prefix.os,'scandir',counted)
    all_ids=[]
    for _ in range(3):
        raw,inserted=idea_ingress.ingest_ideas_source(instance,cfg)
        assert raw and len(raw)<=128
        all_ids.extend(inserted)
    assert all_ids==[f'idea-side-{i:04d}' for i in range(300)]
    assert instance._idea_ingress_cursor[2]==0
    assert len(scans)==1,'one unchanged 300-name window was repeatedly enumerated'


@pytest.mark.parametrize('change',['rewrite','rename_add','remove_add'])
def test_cached_future_name_still_reads_current_payload_and_namespace(engine, change):
    instance,cfg,source=engine
    side=populate(source)
    assert len(idea_ingress.ingest_ideas_source(instance,cfg)[1])==128
    future=side/'0200.md'
    if change=='rewrite':
        future.write_text(_block('idea-new-future',999))
    elif change=='rename_add':
        future.rename(side/'0400.md')
        future.write_text(_block('idea-new-future',999))
    else:
        future.unlink()
        (side/'0200-new.md').write_text(_block('idea-new-future',999))
    raw,inserted=idea_ingress.ingest_ideas_source(instance,cfg)
    assert 'idea-new-future' in inserted
    assert raw['idea-new-future']['config']=={'seed':999}
    assert instance.lake.get('idea-new-future')['config']=='seed: 999\n'
    assert len(instance._idea_sidecar_prefix.name_window)<=512
    assert instance.lake.get('idea-side-0200') is None


def test_namespace_change_while_consuming_cached_window_rejects_before_writer(engine, monkeypatch):
    instance,cfg,source=engine
    side=populate(source)
    assert len(idea_ingress.ingest_ideas_source(instance,cfg)[1])==128
    original=idea_ingress._read_source
    changed=[]
    def read(path):
        value=original(path)
        if path==side/'0128.md' and not changed:
            (side/'0999.md').write_text(_block('idea-after-window',1000))
            changed.append(True)
        return value
    monkeypatch.setattr(idea_ingress,'_read_source',read)
    assert idea_ingress.ingest_ideas_source(instance,cfg)==({},[])
    assert changed==[True] and instance.lake.get('idea-side-0128') is None
    assert instance._idea_sidecar_prefix.files==[]
    assert instance._idea_sidecar_prefix.name_window==()
