# ECG ensemble external evaluation

**Author:** Muhammad Adnan. **Prepared version:**0.4.1 (28October2026). PUBLICATION_STATUS.json records the state at artifact sealing. Current publication state is reported by the version DOI landing page.

Reproducible scores and analyses for *Complementarity and Transfer Limits of Two ECG Pipelines*. The related published protocol is [10.5281/zenodo.23684380](https://doi.org/10.5281/zenodo.23684380), not the software DOI. Software version DOI: [10.5281/zenodo.23729209](https://doi.org/10.5281/zenodo.23729209). Repository: https://github.com/Dan7651/ecg-ensemble-external-evaluation.

## Contents

- Four gzip CSVs of exact FP32 probabilities keyed by public SPH ECG_ID. No waveforms, patient-group identifiers or demographic/date fields are redistributed.
- Five small fitted nonlinear heads with checkpoint hashes/provenance. They require the separately obtained frozen encoder and feature scaler.
- Statistical code, fixed seeds, original training/source records, full-precision tables and stored conditional uncertainty draws.
- Environment pins and commands for each family of tables and figures.

## Quick checks

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python verify_release.py
.venv/bin/python -m unittest test_analysis_core.py
```

## Reproduce

Acquire the exact original public SPH metadata.csv and code.csv from Figshare records17912441 and17912507. These source-release files are acquired separately and joined locally. The CLI requires the pinned metadata hash before joining labels/grouping locally.

```sh
python reproduce.py --all
python reproduce_scores.py --metadata /path/to/metadata.csv
python reproduce_input_view.py --metadata /path/to/metadata.csv
python reproduce_followup.py --metadata /path/to/metadata.csv --codebook /path/to/code.csv
python render_followup.py
python render_in_domain_reference.py
```

Use `reproduce.py --table original`, `--table controls`, `--table sensitivity`, `--table thresholds` or `--table calibration` for individual main tables. Use `--figure matched_controls`, `--figure transfer_diagnostics` or `--figure calibration_review` for individual established figures. See REPRODUCIBILITY.md and OUTPUT_INDEX.md for the precise evidence source and command map. Aggregate rendering redraws stored results; it does not rerun every underlying inference or bootstrap.

Append `--bootstrap 2000` to reproduce_input_view.py to recompute its patient intervals. Append `--crossed` to reproduce_followup.py to recompute1000model–patient draws; these CPU jobs are optional and can take time. Default commands never start model training, download datasets or allocate a GPU.

## Interpretation

One external hospital, one frozen encoder, five fits per family, historically viewed outcomes. Same-encoder controls vary head seeds, not independent pretraining. Error complementarity is observed; a causal representation mechanism is not identified. Calibration/threshold and population findings, including adverse endpoints, remain part of the evidence. The descriptive PTB-XL folds9/10 audit is not untouched confirmation. Clinical review remains pending.

## Terms

Own code/fitted heads: MIT. Scores/own numerical exports: CC0. The protocol and upstream components retain separate terms; read THIRD_PARTY_NOTICES.md. Do not interpret research targets as adjudicated diagnoses or use these artifacts as a clinical device.
