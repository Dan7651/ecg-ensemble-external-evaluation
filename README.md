# ECG ensemble external evaluation

Author: Muhammad Adnan. Version0.4.1.

Reproducible source, public-record-ID probability scores, fitted heads and analyses for **Complementarity and Transfer Limits of Two ECG Pipelines**.

- Software DOI: https://doi.org/10.5281/zenodo.23729209
- Separate analysis protocol: https://doi.org/10.5281/zenodo.23684380

## Download and run

The complete, tested source tree is distributed in `ECG_Ensemble_External_Evaluation_v0.4.1.zip` in this repository. Download and extract it before running commands; the archive preserves its package directories, score files and provenance. The ZIP is also archived on Zenodo with the same byte hash. `SHA256SUMS` authenticates the distribution.

```sh
unzip ECG_Ensemble_External_Evaluation_v0.4.1.zip
cd ecg-ensemble-external-evaluation
python -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python verify_release.py
.venv/bin/python -m unittest test_analysis_core.py
.venv/bin/python reproduce.py --all
```

The extracted README and REPRODUCIBILITY.md provide one command per output family. Source metadata is acquired separately from the public SPH release for locally joining labels/patient grouping. Empirical ROC/PR, input-view controls and additional sensitivity tables can be recomputed from the released exact FP32 scores. Aggregate rendering redraws stored estimates and intervals rather than rerunning model inference.

## Scope and included material

The package contains151sealed files, four score exports, five small fitted classification heads, source/training provenance, statistical code, fixed seeds, checkpoint hashes, tables and interval draws. The pretrained encoder and raw waveforms are excluded; the heads require the pinned encoder/scaler. Patient-group identifiers, demographics and private clinical-review materials are excluded.

One external hospital and one frozen encoder were evaluated. Historical outcome access, calibration and threshold limitations, mixed F1/Brier outcomes and the finite-five-fit limitation remain documented. No untouched confirmation, clinical-device certification or matched published benchmark victory is claimed.

Own code/heads: MIT. Derived scores/own numerical exports: CC0. The protocol retains CC-BY-4.0 and upstream components retain their own notices. See THIRD_PARTY_NOTICES.md and the archive.

Publication metadata inside the archive describes its state at sealing; use the DOI landing page for current publication status.
