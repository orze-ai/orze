# Training data boundaries

Orze can prevent a managed training process and its descendants from reading
declared held-out filesystem roots or using the host network. These controls
reduce leakage risk; they do not prove that a pretrained model or an undeclared
copy of a sample is uncontaminated.

```yaml
data_boundaries:
  forbidden_in_training:
    - /datasets/benchmark/held-out
  watch_paths:
    - /datasets/audit-only
  training_network: deny
```

`forbidden_in_training` is a hard boundary. Every target must be an existing,
ordinary, absolute file or directory with no symlink redirection. Before GPU
telemetry, Orze runs a CUDA-hidden capability probe for a private user/mount
namespace. The actual trainer command repeats the mandatory setup, makes mount
propagation private, hides each directory behind an empty `tmpfs` (or each file
behind a read-only empty device), and only then executes the training wrapper.
Missing namespace tools, disabled user namespaces, unavailable or redirected
targets, and any mount failure reject the launch. Hard blocking never degrades
to a Python hook.

`training_network: deny` adds a private network namespace for the trainer and
all descendants. `inherit` retains host networking and is the backward-
compatible default. Namespace capability is checked without CUDA, NVIDIA, HIP,
or ROCm visibility and before the live GPU availability query.

`watch_paths` is audit-only. It records Python `builtins.open()` access but does
not stop it and cannot observe native loaders, `os.open`, memory maps, or remote
fetches. It must not be used as evidence that held-out data was inaccessible.

For a defensible leakage boundary:

1. Run training and evaluation as separate Orze phases. In-loop evaluation
   cannot read a path that is correctly forbidden during training.
2. Declare every canonical local root containing held-out samples under
   `forbidden_in_training`.
3. Set `training_network: deny` after the artifact resolver has pinned and
   cached legitimate training inputs.
4. Pin training-data manifests and audit them for sample/speaker/source overlap
   with the evaluation manifest using the keyed-fingerprint
   [data-separation contract](data-separation.md). Filesystem isolation cannot
   discover an undeclared alias, hard link, copied sample, or semantic duplicate.
5. Treat benchmark receipts as local evidence. They do not prove the data
   history of an externally pretrained model or an official leaderboard rank.

When Orze itself trains the evaluated artifact, enable
[managed model lineage](model-lineage.md) to bind successful completion to the
post-namespace boundary acknowledgement, data-separation receipt, and exact
model bytes.

The Python wrapper also requires a namespace-active marker whenever a hard
boundary is configured. This is defense against a future launcher refactor
accidentally executing user code outside the mandatory namespace; it is not a
substitute for the kernel setup itself.
