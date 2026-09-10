def started(lake, ep, idea_dir, record_start):
    """Record observed process creation without holding a lock across Popen."""
    with execution_transaction(lake, idea_dir) as tx:
        _owned(lake, ep, states=("LAUNCHING",))
        # Failure leaves the already committed LAUNCHING intent. The launcher
        # must stop the known process; confirmed stop permits failed closure.
        receipt = record_start(ep, idea_dir, phase="evaluation")
        _verify_compute(idea_dir, receipt)
        mark_running(lake.conn, _ref(ep))
        tx.watch_attempt(_ref(ep))
