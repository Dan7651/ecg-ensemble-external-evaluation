# A01 matched-seed fitting package

The scientific plan is version 1.0.0, independently published as
https://doi.org/10.5281/zenodo.23684380 before these new fits. The historical
"not yet deposited" heading in the sealed protocol is preserved byte for byte;
the actual published-record receipt supersedes that preparation status.

## Fixed scope

Five ResNets and five independently initialized frozen-JEPA nonlinear heads:
seeds 101, 202, 303, 404, 505. Positive-superclass PTB-XL folds 1–7 fit,
fold 8 development. No fold 9/10 prediction or external inference is performed.
All ten selected checkpoints must be locked before the subsequent SPH analysis.
This is a preplanned extension on a previously observed cohort, not fresh cohort
confirmation or a published benchmark victory.

## Existing inputs only

The loader copies missing verified files from the existing amplitude cache and
the 17 cached frozen-encoder feature chunks. It checks their full hashes, row
IDs and patient partitions. It performs no raw-record download, filtering or
encoder extraction. Normalization fits training rows only and is reused after
its receipt is committed. Its exact input hashes are in `input_pins.json`.

The original ResNet source is included for its two class definitions only.
The runner extracts those classes using AST. **Do not execute or import
`multiscale_waveform_dev8.py` as a script**: it contains historical experiments.

## Execution and recovery

Use the existing authorized Colab T4 and readable Drive mount. The submission
wrapper verifies the ZIP and every source member, and starts exactly one
subprocess. `a01_matched_fits.py --fit` verifies the live public protocol and
records the actual environment before fitting. Each family must pass real T4
finite-gradient, model/optimizer/scaler/all-RNG replay and training-only capacity
checks. Neither a local CPU test nor a queued notebook cell satisfies these gates.

The runner uses fresh initialization for each prescribed seed, fixed original
recipes and selection tolerances. The implementation uses `num_workers=0` for
resumable deterministic loading instead of the old persistent two-worker loader;
there is no stochastic waveform augmentation. It does not claim bitwise equality
with the historical seed-42 run or identical shuffle trajectories across loaders.
The frozen shared encoder means JJ varies head fitting, not pretraining seeds.

Two alternating checkpoint generations retain optimizer, scheduler, scaler,
Python/NumPy/Torch CPU/CUDA and shuffle RNG state, plus the selected weights.
SHA receipts commit complete epochs. Recovery restores the latest intact
generation; it does not redo committed epochs. Incomplete epochs may be replayed.
Completed endpoints are verified and reused. All-ten completion is idempotent.
Original models and result folders are untouched.

Run outputs are written under
`MyDrive/ECG_Project_Final_Results/sph_matched_seed_controls_20261027_v1/`.
The endpoint records exact fresh saved-best replay in that runtime. It explicitly
does **not** claim persistence after remount: a separate actual flush/remount and
full artifact verification is required before external inference.

## Local bounded checks

`python test_a01_matched_fits.py` uses tiny synthetic models and temporary data.
It does not load a real ECG, instantiate the actual ResNet, load pretrained
weights or test GPU capacity. The source review and execution receipts state
those limits. The CUDA AMP path remains the original modern GradScaler path;
the disabled CPU scaler has an older-Torch compatibility branch for these tests.

## Release boundary

This private computation bundle contains code, hash manifests and protocol files,
not waveforms, patient-level predictions, credentials or weights. It is not the
eventual public software release or its DOI. Do not rewrite manuscript claims
until the prespecified matched-control analysis and framing decision are complete.
