"""GC lake ranking must tolerate bad history and respect metric direction."""

import sqlite3

from orze.agents.orze_gc import get_top_idea_ids


def test_gc_lake_query_skips_malformed_json_and_keeps_lowest_wer(tmp_path):
    db_path = tmp_path / "idea_lake.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE ideas (idea_id TEXT, eval_metrics TEXT)")
    conn.executemany(
        "INSERT INTO ideas VALUES (?, ?)",
        [
            ("idea-best", '{"avg_wer": 5.0}'),
            ("idea-worse", '{"avg_wer": 10.0}'),
            ("idea-corrupt", "not-json"),
        ],
    )
    conn.commit()
    conn.close()

    keep = get_top_idea_ids(
        tmp_path,
        "avg_wer",
        db_path,
        keep_top=1,
        sort_order="ascending",
    )

    assert keep == {"idea-best"}
