# Definition-sensitivity and figure-layout extension (28 October 2026)

These post hoc saved-score analyses accompany manuscript revision v12. They were not prospectively registered and do not alter the frozen primary map, models, thresholds or population. The unchanged base distribution remains software v0.4.1, DOI 10.5281/zenodo.23729209; the separate protocol is DOI 10.5281/zenodo.23684380. These new scripts are a repository extension, not a claim that the immutable v0.4.1 deposit already contained them.

Use the pinned analysis environment and exact original public SPH metadata.csv (Figshare record 17912441; SHA256 checked by each command). RELEASE is a checkout containing all v0.4.1 files, or the extracted v0.4.1 archive.

```sh
python reproduce_definition_sensitivity_v12.py --release RELEASE --metadata /path/to/metadata.csv
python render_empirical_curves_v12.py --release RELEASE --metadata /path/to/metadata.csv
```

The first command reconstructs Supplement S2b and all 25 cyclic-pair mapping contrasts. No five-seed added-record scores are available, so the inclusion row is original-pair only. It produces point sensitivities, without new confidence intervals. The second command reconstructs the 1x5 empirical ROC and PR panels for Supplement S4, preserving the actual probability arrays and labels. No raw ECG, checkpoint loading, training or model inference occurs. Full-precision output tables and independent rank-sum verification accompany the local manuscript package.

Base component licenses apply: own analysis code MIT; own numerical exports CC0. No clinical review or fresh confirmation is claimed.
