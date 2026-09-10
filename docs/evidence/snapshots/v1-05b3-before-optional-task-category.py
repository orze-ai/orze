def _task(lake, task_id):
    rows = lake.conn.execute(
        "SELECT idea_id, CASE WHEN typeof(config)='text' AND "
        "length(CAST(config AS BLOB))<=65536 THEN config ELSE NULL END, "
        "kind,priority,category FROM main.ideas WHERE idea_id=? COLLATE BINARY LIMIT 2",
        (task_id,),
    ).fetchall()
    if (len(rows) != 1 or rows[0][0] != task_id or type(rows[0][1]) is not str
            or rows[0][2] != "train" or any(type(v) is not str or len(v) > 128 for v in rows[0][3:])):
        raise ReplicationError("replication_task_unavailable")
    return {"config": rows[0][1], "kind": rows[0][2],
            "priority": rows[0][3], "category": rows[0][4]}
