# Public-score and fitted-head release candidate

Author: Muhammad Adnan. This local candidate has NOT been published to GitHub or assigned a software DOI. The protocol DOI is a separate record.

## Record-level material
Four gzip CSVs contain public SPH ECG_ID keys and FP32 roundtrip probabilities only. No patient identifiers, age, sex, dates, or waveforms are exported. Equal fusion is computed in FP32 from the two component columns. The original all-record file includes the annotation-defined selected flag; the full-coverage file identifies long recordings and window counts. The ten new fits are all included; none is selected on SPH.

Patient clustering and targets must be recovered from the original public metadata.csv/code.csv (Figshare articles 17912441 and 17912507, CC0) using ECG_ID. Do not treat unmapped statements as adjudicated healthy labels. Keep patient identity for analysis and do not release identifying additions. `reproduce_scores.py --metadata /path/metadata.csv --out reproduced_scores` regenerates original ROC/PR coordinates and seed metrics. Aggregate manuscript figures remain available through `reproduce.py --all`.

## Fitted heads and provenance
Five small fitted nonlinear JEPA heads are included, source byte hashes verified against prior provider receipts without local deserialization. The large pretrained encoder and raw ECGs are not redistributed. Use the exact pinned encoder and feature scaler from the model provenance to reproduce model execution; these head files alone are not standalone ECG classifiers. Model and source provenance, family recipes and original development histories are preserved in the accompanying research bundle. Loading a pickle-based checkpoint can execute code; use trusted verified files and weights-only loading when compatible.

## Terms
Own software and fitted-head additions: MIT. Dataset-derived numerical scores: CC0, with source attribution requested. Original source datasets, author encoder source, IEEE templates, protocol and third-party components retain their individual terms. Four current SPH provider records (waveforms, metadata, dictionary and analysis code) were verified as CC0; the mapping-reference record retains earlier cached primary evidence. Source-paper ethics approval is not a determination for this author's institution. No prospective clinical use, protected confirmation, or matched benchmark victory is claimed.


## Input-view control update (v0.3.0-rc1 (historical input-view update), 28 October 2026)
Five additional input-matched ResNets completed fitting. Their frozen external evaluation and all2,000patient draws are complete; actual downloaded results were independently verified. This is a post hoc control on the previously viewed SPH cohort, not a new preregistration or untouched confirmation.

`python reproduce_input_view.py --metadata /path/to/metadata.csv` rebuilds the new pair table and four-panel figure from the new public-record-ID FP32 score export and the existing matched-seed export. Add `--bootstrap 2000` to reproduce every patient draw using the public patient grouping. The default command does not rerun the bootstrap. All529,575newprobabilities round-trip bitwise.

New external mean mixed-pair AUROCgain0.00734891 (SD0.00037761), versus input-matched RRgain0.00142199. The paired gain contrast is0.00592692 [0.00431656,0.00659491]. Matching input increased AUROCgain versus original mixed pairs but decreased APgain; one mixed pair increased Briererror. Fixed threshold0.5 and allfive cyclic pairs are retained.

Actual public GitHub/Zenodo software publication is pending; the independently published protocol DOI is not a software-release DOI.
