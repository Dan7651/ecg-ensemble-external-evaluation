# Execution and provenance

Original seed: 42. New A01 training seeds: 101, 202, 303, 404, 505.
Training: PTB-XL positive-target folds 1–7; development: positive-target fold 8.
New external fitting is prohibited; external A01 inference follows completed locks.
Original ResNet checkpoint SHA-256:
fd83f0be7ab01b3ba2445aa1c9582800222d300c3088e96c154711724fdc10c5
Original JEPA head SHA-256:
2b4eab786f806df1a3fba6ba769931b869520cf04982bcf00bd6e04c59717de7
Shared encoder SHA-256:
61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e
Pinned encoder source commit:
d937ad2c2c8a1e22856ce7e4a23a30f84a71217c

2,000 patient-cluster draws: A01/A02/A03/A04/A06 seed 2026102701;
A03 patient-fold assignment 2026102703; A05 offsets seed 2026102705;
A05 paired Brier reused the allowed earlier 20261026 draws.
Both cohort membership and predictions are held fixed except for the explicitly
specified inclusion/duration analyses. Source prevalence is fixed in A04 draws.
Stored assessment SHA-256 values appear in `data/aggregate_results.json`.

Local verification used Python 3.12.14, NumPy 1.26.4, SciPy 1.12.0,
scikit-learn 1.5.2. Remote A02/A06 inference reported Python 3.13.16,
Torch 2.11.0+cu130, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1,
timm 1.0.30 and h5py 3.14.0. A uniform environment is not falsely asserted.
Full runtime manifests/checkpoint receipts remain in the private project.


## Input-view control update (v0.3.0-rc1, 28 October 2026)
Five additional input-matched ResNets completed fitting. Their frozen external evaluation and all2,000patient draws are complete; actual downloaded results were independently verified. This is a post hoc control on the previously viewed SPH cohort, not a new preregistration or untouched confirmation.

`python reproduce_input_view.py --metadata /path/to/metadata.csv` rebuilds the new pair table and four-panel figure from the new public-record-ID FP32 score export and the existing matched-seed export. Add `--bootstrap 2000` to reproduce every patient draw using the public patient grouping. The default command does not rerun the bootstrap. All529,575newprobabilities round-trip bitwise.

New external mean mixed-pair AUROCgain0.00734891 (SD0.00037761), versus input-matched RRgain0.00142199. The paired gain contrast is0.00592692 [0.00431656,0.00659491]. Matching input increased AUROCgain versus original mixed pairs but decreased APgain; one mixed pair increased Briererror. Fixed threshold0.5 and allfive cyclic pairs are retained.

Actual public GitHub/Zenodo software publication is pending; the independently published protocol DOI is not a software-release DOI.

## Additional analyses (28 October 2026)

This is a local software candidate; the published protocol DOI is not a software DOI.

Recompute all diversity, oracle-bound, leave-one-fit-out, statement-level NORM, and prevalence-sensitivity tables from released probabilities and the public source metadata:

```sh
python reproduce_followup.py --metadata /path/to/metadata.csv --codebook /path/to/code.csv
```

Append `--crossed` to recompute the 1,000 model--patient draws (CPU, may take time). Five fits per family support an exploratory finite-model sensitivity only. Patient order is lexicographic by the public Patient_ID string, matching the original analysis; it does not imply independent pretraining. Clinical review is not completed.

The portable entry point checks source metadata identity. The `followup_tools` directory retains original workspace scripts as provenance; use `reproduce_followup.py` for portable score reanalysis. Tables and draws are in `followup_tables`. New figure inputs are actual saved-score tables; no simulated metrics are introduced.

Render the new diversity/prior figure and LaTeX tables after the score command:
```sh
python render_followup.py
python render_in_domain_reference.py
```
The latter uses actual verified aggregate in-domain results. Full inference source, raw prediction receipts and the private clinical packet remain in the complete working archive.
