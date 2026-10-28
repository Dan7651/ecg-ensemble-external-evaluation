# Output-to-command map

| Evidence | Command | Nature |
|---|---|---|
| Original, controls, inclusion, thresholds, calibration main tables | python reproduce.py --table NAME | Stored full-precision aggregates |
| Established aggregate figures | python reproduce.py --figure NAME | Stored results and conditional intervals |
| Original empirical ROC/PR and component metrics | python reproduce_scores.py --metadata metadata.csv | Recomputed from released FP32 scores and public labels |
| Input-matched pair tables/figure | python reproduce_input_view.py --metadata metadata.csv | Recomputed from released scores |
| Diversity/oracle/leave-one-fit-out/NORM statements/prior sensitivity | python reproduce_followup.py --metadata metadata.csv --codebook code.csv | Recomputed from released scores |
| Added diversity/prior figure and LaTeX tables | python render_followup.py | Rendered from immediately preceding score reanalysis |
| Descriptive folds9/10 figure | python render_in_domain_reference.py | Rendered from independently verified actual aggregate assessment |

The in-domain figure uses aggregate results, not newly exposed row-level patient information. Original runtime model/checkpoint replay remains recorded evidence, distinct from independently recomputed saved-score statistics. Full source execution requires separately acquired data/encoder and the documented original runtime environment.
