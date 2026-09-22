# Real GPU research verification

Verified: 117 closed model calls; 132 closed GPU containers.

| Strategy | Accepted goals | Mean capped time to goal |
|---|---:|---:|
| control | 5/8 | 71.44 min |
| improved | 5/8 | 54.98 min |

Improved minus control: -16.46 min; descriptive problem-level 95% t interval [-37.82, 4.89] min.

No claim of general research superiority follows from this single ASR campaign.

- One pretrained ASR model and one research model; four related ASR corpora.
- Shared GPU interference is measured but not eliminated; assignments swap across repetitions.
- Group bootstrap is descriptive; AMI meeting groups and FLEURS sentence groups do not guarantee speaker independence.
- Repetitions may overlap source examples; problem-level inference has only four units.
- Reused known task splits: an execution follow-up, not unseen-task generalization.
- Execution bundle comparison does not identify the contribution of each component.
- Pair-wise scheduling applies to both arms; no randomized estimate of whole-wave barrier removal.
- Common input preparation excluded; goal delivery precedes settling unused requests, which remain counted and charged.
