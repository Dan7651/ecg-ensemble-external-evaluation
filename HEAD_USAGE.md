# Fitted-head use

`checkpoints/jepa_head_seed*.pth` contains a `model` state dict, protocol, selected epoch, and epoch history from the original completed fit. Hash each file before use. These small newly fitted heads are distinct from the large pretrained encoder, which is not redistributed. Each provenance subfolder contains actual protocol, endpoint, history and provider-byte hash receipts.

The exact feature scaler is included in `data/normalization.npz` (`feature_mean`, `feature_scale`, `feature_variance`). `training_source/a01_matched_fits.py` defines the original 768→128→5 GELU/dropout0.5 head and normalization pathway; the original feature chunks and pinned encoder are required to regenerate encoder representations. `data/all_ten_fits_locked.json` and `training_checkpoint_pins.json` authenticate the complete fits. Use the official pinned encoder source/license and original author checkpoint; do not substitute another encoder or pretraining version.

For trusted files, prefer `torch.load(path, map_location="cpu", weights_only=True)` when supported. If a weights-only load rejects a file, diagnose the schema; do not automatically fall back to unrestricted deserialization. No real checkpoint loading was performed by the local release verifier.
