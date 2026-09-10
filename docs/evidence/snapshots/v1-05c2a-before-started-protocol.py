def started(lake, ep, idea_dir, record_start):
    """Record observed process creation without holding a lock across Popen."""
    from orze.engine.evaluation_supervision import ready_binding
    supervision = ready_binding(ep, idea_dir)
    with execution_transaction(lake, idea_dir) as tx:
        row = _owned(lake, ep, states=("LAUNCHING",))
        # Failure leaves the already committed LAUNCHING intent. The launcher
        # must stop the known process; confirmed stop permits failed closure.
        receipt = record_start(ep, idea_dir, phase="evaluation")
        _verify_compute(idea_dir, receipt, process=ep, phase="evaluation",
                        event="start", outcome="started")
        binding = dict(row["binding"])
        binding["process_pid"] = getattr(getattr(ep, "process", None), "pid", None)
        binding["supervision"] = supervision
        mark_running(lake.conn, _ref(ep), binding=binding)
        tx.watch_attempt(_ref(ep))


