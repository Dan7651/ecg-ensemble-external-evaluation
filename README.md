# ECG ensemble external evaluation

**Author:** Muhammad Adnan. Analysis and training scripts, pinned environments, aggregate data and numerical tables are browsable directly in this repository. The complete immutable release, including score exports, fitted heads and provenance, remains available as a checksum-verified artifact.

- Software v0.4.1: [10.5281/zenodo.23729209](https://doi.org/10.5281/zenodo.23729209)
- Separate prospective-extension protocol: [10.5281/zenodo.23684380](https://doi.org/10.5281/zenodo.23684380)
- Additional post hoc definition sensitivities: see [REVISION_V12.md](REVISION_V12.md). These repository extensions are not represented as having been in the earlier immutable deposit or prospective protocol.

## Reproduce the main aggregate tables and figures from a clone

```sh
git clone https://github.com/Dan7651/ecg-ensemble-external-evaluation.git
cd ecg-ensemble-external-evaluation
python -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python reproduce.py --all
```

Aggregate rendering reconstructs stored estimates and intervals; it does not repeat model inference or all bootstrap draws. Use `--table original`, `controls`, `sensitivity`, `thresholds` or `calibration` for individual tables. The code and data used by these commands are native repository files.

## Verify the complete frozen release and recompute from scores

Some fitted-head, probability and legacy provenance assets remain in the release ZIP. Restore them into a separate versioned directory with one command, preserving the browsable repository files:

```sh
.venv/bin/python extract_verified_release.py
cd sealed_v0.4.1
../.venv/bin/python verify_release.py
../.venv/bin/python -m unittest test_analysis_core.py
../.venv/bin/python reproduce_scores.py --metadata /path/to/metadata.csv
```

The extraction command verifies the complete ZIP SHA256 and all 151 sealed file hashes; it refuses to overwrite differing files. Run the original strict `verify_release.py` **inside `sealed_v0.4.1`**, where the original README and full manifest are preserved. The repository README and additional revision scripts are maintained separately from that immutable release.

Obtain exact original public SPH `metadata.csv` and `code.csv` from Figshare records 17912441 and 17912507. Commands check the pinned metadata hash before joining labels or patient grouping. No patient-group IDs, demographics or raw waveforms are redistributed. See the sealed `REPRODUCIBILITY.md` and repository `OUTPUT_INDEX.md` for command coverage, fitted-head requirements and interval recomputation options.

## Revision v12 outputs

From the repository root:

```sh
.venv/bin/python reproduce_definition_sensitivity_v12.py --release sealed_v0.4.1 --metadata /path/to/metadata.csv
.venv/bin/python render_empirical_curves_v12.py --release sealed_v0.4.1 --metadata /path/to/metadata.csv
```

These commands use existing scores only. They do not fit models, thresholds or fusion weights. Five-seed inclusion scores are unavailable, so no such result is inferred. Clinical adjudication remains pending. Historical outcome access, one hospital, one encoder and finite-five-fit limitations qualify interpretation; no untouched confirmation, clinical-device certification, novel architecture or matched published benchmark victory is claimed.

Own code/fitted heads: MIT. Derived scores/own numerical exports: CC0. Protocol and upstream components retain their separate terms; see `THIRD_PARTY_NOTICES.md` and the sealed release. Large pretrained encoders and raw ECGs are excluded.
