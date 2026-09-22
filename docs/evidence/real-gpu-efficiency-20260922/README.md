# Verified research execution comparison

All 16 episodes are complete and independently verified. Both execution arms use
ParallelRefine and accept 5/8 goals. Mean capped time to final acceptance is
71.44 minutes for control and 54.98 minutes for improved (23.0% lower). Failures
count as 120 minutes. The predeclared mean/success-rate promotion rule passes.

The four-problem 95% t interval for improved minus control is [-37.82, +4.89]
minutes; exact sign-flip p=0.25. This is evidence on four known ASR corpora, not
proof of universal expected superiority or unseen-task research creativity.

- [Report and lessons](../../plans/2026-09-22-execution-efficiency-results.zh-CN.md)
- [Frozen protocol](../../plans/2026-09-22-execution-efficiency.zh-CN.md)
- [Summary](summary.json), [independent verification](verification.json),
  [successful verifier exit](verification-process-closed.json)
- [Predeclared default decision](default-decision.json) and
  [implementation checks](deployment.json)
- [Waiting, errors, unused calls and cleanup](operational-summary.json)
- [Full supplementary rescore](supplemental-audit/final-audit.json)

Run `python rescore_public.py` from this directory to recompute acceptance, means
and the problem-level interval from audit-counts.json; NumPy and SciPy are needed.
Public records contain per-group error counts, not raw transcripts. The earlier
partial audit remains explicitly partial and is superseded by the final audit.

The private Pro repository retains frozen-protocol.tar.gz and
completed-records.tar.gz with SHA-256 manifests. Large audio, features and
checkpoints remain in the documented local study directories. Frozen verification
ran before the main source update; reproduce its source checks using the frozen
archive, not the subsequently updated working tree. The public rescore remains
independent of the current Core/Pro implementation.
