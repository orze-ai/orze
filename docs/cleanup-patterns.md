# Built-in disposable-file cleanup

`cleanup.patterns` selects disposable files beneath direct `idea-*` task
directories. It defaults to `[]` (disabled). This contract covers that built-in
path only: checkpoint GC, cold-storage moves and `cleanup.script` are separate
execution paths and do not inherit these guarantees.

The existing public entry remains `run_cleanup(results_dir, cfg, *, lake=None)`
and returns `None`. Periodic controller maintenance supplies its actual Lake.
The internal `cleanup_pattern_files` helper returns `status`, `deleted`,
`skipped_tasks` and stable, content-free `errors` for diagnosis.

## Selection and containment

- Validate the complete pattern list before deleting any matching file.
  Malformed lists, empty/non-string patterns, absolute paths, parent traversal,
  noncanonical separators and redirected directories are not cleanup targets.
  The list is limited to 128 patterns, each at most 4,096 UTF-8 bytes and 64
  path components; invalid batches do not partially apply valid entries.
- Traverse directories without following symbolic links. Only captured regular
  single-link files may be deleted; directories, links and special files are
  not removed. Matching uses case-sensitive relative path components, including
  recursive `**`. Directory-only patterns ending in `**` still select no files,
  matching the previous `pathlib.Path.glob` file-filter behavior.
- Discover candidates before taking the task guard. A task scan exceeding
  16,384 entries or 64 directory levels is refused without applying its partial
  candidate list.
- Reopen directory components without following links, then compare the
  captured file identity with both the opened file and directory entry before
  unlinking. Sync the parent directory before reporting a confirmed deletion.

## Authority and retained evidence

Cleanup uses the same per-task effect guard as native claims, publication and
reset. Inside it, read a fresh, existing-only SQLite snapshot in `mode=ro` with
`query_only`; never create/migrate a database or rewrite task lifecycle state.
Conflicting, missing, malformed or active authority does not downgrade to a
legacy task. Caller-owned SQL transactions cannot grant cleanup authority.

An authoritative task must have an agreed closed lifecycle and no unclosed
current attempt in any phase. A retained effect owner, incomplete publication
or unconfirmed stop holds cleanup. An offline legacy directory with no database
route and no outstanding claim retains explicit disposable-file cleanup.

Retain framework-private control trees and execution/stop/compute/evaluation
records, claim/metrics/recovery history, task configuration, declared project
inputs, report sources and artifact outputs. The actual routed database and
its SQLite sidecars are protected even when current configuration omits their
path. Native launch-bound output and control locations remain protected when
the cleanup configuration changes; current configuration is not permission to
erase accepted evidence.

Legacy `model_lineage.artifact` and `eval_checkpoint` are task-relative;
`resume.immutable_inputs` are project-relative and may declare whole trees.
When evaluation is enabled without an explicit checkpoint path, retain its
existing `best_model.pt` default. These are declaration-based protections, not
a claim that arbitrary files are scientifically valid or a blanket extension
rule for all model files.

## Failure semantics and limits

`disabled` and `invalid` do not apply the built-in pattern batch. `completed`
means the selected scan completed, not that a research task or observation is
valid. `partial` records a refused/unavailable task or deletion error. Only
confirmed unlinks increment `deleted`; an I/O error after unlink may leave a
removed file without a success count. There is no rollback, atomic multi-file
deletion, automatic trash recovery or claim of a transactional filesystem.

The guard coordinates cooperating framework writers, not arbitrary same-user
programs. This is not an OS sandbox or proof that all unknown descendants have
stopped. Natural-exit descendant closure, standalone GC, user cleanup programs
and Pro process control require their own verification. These tests establish
cleanup mechanisms, not general research-quality or compute-saving gains.
