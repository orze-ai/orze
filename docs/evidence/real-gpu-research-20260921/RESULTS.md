# Real GPU research verification

Verified: 106 closed model calls; 120 closed GPU containers.

| Strategy | Accepted goals | Mean capped time to goal |
|---|---:|---:|
| control | 5/8 | 60.38 min |
| combined | 5/8 | 73.63 min |

Combined minus control: 13.24 min; descriptive problem-level 95% t interval [-27.56, 54.04] min.

No claim of general research superiority follows from this single ASR campaign.

- One pretrained ASR model and one research model; four related ASR corpora.
- Shared GPU interference is measured but not eliminated; assignments swap across repetitions.
- Group bootstrap is descriptive; AMI meeting groups and FLEURS sentence groups do not guarantee speaker independence.
- Repetitions may overlap source examples; problem-level inference has only four units.
- Common data preparation and calibration time excluded from arm timing and reported separately.
