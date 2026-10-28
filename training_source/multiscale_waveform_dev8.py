"""PTB-XL waveform architecture screen on fold 8; fold 9/10 are untouched.

Run on a Colab T4 High-RAM runtime after the fixed-graph ablation ends. The
existing Drive amplitude cache is reused. Each candidate has an independent,
resumable checkpoint. This is an architecture screen, not a test result.
"""
from pathlib import Path
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from google.colab import drive
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

drive.mount('/content/drive')
assert torch.cuda.is_available(), 'Select the T4 GPU runtime.'
device = torch.device('cuda')
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

cache = Path('/content/drive/MyDrive/ECG_Diagnosis_Project/processed_cache/ptb-xl_amplitude_v1')
manifest_bytes = (cache / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
for filename, expected in manifest['files'].items():
    p = cache / filename
    assert p.is_file() and p.stat().st_size == expected['bytes'], f'Incomplete cache: {p}'
with np.load(cache / 'labels_and_folds.npz', allow_pickle=False) as z:
    labels = z['y_standard'].astype(np.float32)
    folds = z['strat_fold']
    ecg_ids = z['ecg_id']
    patient_ids = z['patient_id']
train_rows = np.flatnonzero(np.isin(folds, range(1, 8)) & (labels.sum(1) > 0))
dev_rows = np.flatnonzero((folds == 8) & (labels.sum(1) > 0))
assert len(train_rows) > 10000 and len(dev_rows) > 1000
assert not (set(patient_ids[train_rows]) & set(patient_ids[dev_rows]))
assert np.all(folds[train_rows] <= 7) and np.all(folds[dev_rows] == 8)
signals = np.load(cache / 'ecg_filtered_mv_100hz.npy')
assert signals.shape == (21799, 12, 1000) and signals.dtype == np.float32

# Fold-8 development uses statistics computed only from folds 1-7.
lead_sum = np.zeros(12, np.float64)
lead_sqsum = np.zeros(12, np.float64)
for rows in np.array_split(train_rows, 32):
    batch = signals[rows].astype(np.float64)
    lead_sum += batch.sum(axis=(0, 2))
    lead_sqsum += np.square(batch).sum(axis=(0, 2))
n = len(train_rows) * 1000
mean64 = lead_sum / n
std64 = np.sqrt(np.maximum(lead_sqsum / n - mean64**2, 1e-12))
lead_mean = mean64.astype(np.float32).reshape(12, 1)
lead_std = std64.astype(np.float32).reshape(12, 1)
assert np.isfinite(lead_std).all() and (lead_std > 0).all()

class ECGDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = int(self.rows[index])
        x = ((signals[row] - lead_mean) / lead_std).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(labels[row])

shuffle_rng = torch.Generator().manual_seed(seed)
train_loader = DataLoader(ECGDataset(train_rows), batch_size=64, shuffle=True,
                          generator=shuffle_rng, num_workers=2, pin_memory=True,
                          persistent_workers=True)
dev_loader = DataLoader(ECGDataset(dev_rows), batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True, persistent_workers=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, 1, 3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                                        nn.BatchNorm1d(out_channels)))
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + self.skip(x))

class OriginalResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            ResidualBlock(64, 64), ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128), ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256), ResidualBlock(256, 256, 2))
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(256, 5))
    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat((h.mean(-1), h.amax(-1)), dim=1))

class SqueezeExcite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                 nn.Conv1d(channels, max(16, channels // 8), 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(max(16, channels // 8), channels, 1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.net(x)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Three temporal scales share the same input. Depthwise filters keep
        # this candidate substantially smaller than the old graph network.
        widths = [out_channels // 3, out_channels // 3,
                  out_channels - 2 * (out_channels // 3)]
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels, in_channels, k, stride,
                                    k // 2, groups=in_channels, bias=False),
                          nn.Conv1d(in_channels, width, 1, bias=False))
            for k, width in zip((7, 15, 31), widths)])
        self.fuse = nn.Sequential(nn.BatchNorm1d(out_channels), nn.GELU(),
                                  nn.Conv1d(out_channels, out_channels, 3, padding=1,
                                            bias=False), nn.BatchNorm1d(out_channels))
        self.se = SqueezeExcite(out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride,
                                                   bias=False), nn.BatchNorm1d(out_channels)))
    def forward(self, x):
        h = torch.cat([branch(x) for branch in self.branches], dim=1)
        return torch.relu(self.se(self.fuse(h)) + self.skip(x))

class MultiScaleSEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 64, 15, 2, 7, bias=False),
                                  nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(
            MultiScaleBlock(64, 64), MultiScaleBlock(64, 128, 2),
            MultiScaleBlock(128, 128), MultiScaleBlock(128, 256, 2),
            MultiScaleBlock(256, 256), MultiScaleBlock(256, 256, 2))
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True),
                                  nn.Dropout(0.3), nn.Linear(256, 5))
    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(torch.cat((h.mean(-1), h.amax(-1)), dim=1))

@torch.inference_mode()
def predict(model):
    model.eval()
    truth, probs = [], []
    for x, y in dev_loader:
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x.to(device, non_blocking=True))
        truth.append(y.numpy())
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(truth), np.concatenate(probs)

def metrics(y, p):
    return {'macro_auroc': float(roc_auc_score(y, p, average='macro')),
            'macro_ap': float(average_precision_score(y, p, average='macro')),
            'macro_f1_fixed_0_5': float(f1_score(y, p >= .5, average='macro',
                                                zero_division=0))}

source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
run_root = Path('/content/drive/MyDrive/ECG_Project_Final_Results/waveform_architecture_dev8_seed42')
run_root.mkdir(parents=True, exist_ok=True)
protocol_base = {
    'dataset': 'PTB-XL 1.0.3, 100 Hz filtered physical mV, 5 standard superclasses',
    'cohort': 'records with at least one superclass label',
    'train_folds': '1-7', 'development_fold': 8,
    'selection_fold_9': 'untouched in this screen', 'test_fold_10': 'sealed',
    'train_records': int(len(train_rows)), 'development_records': int(len(dev_rows)),
    'normalization': 'per-lead mean/std from folds 1-7 only',
    'loss': 'unweighted BCEWithLogitsLoss', 'selection_metric': 'macro AUROC',
    'seed': seed, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'source_sha256': source_sha256}

def run_candidate(name, model_class):
    # Reset initialization and shuffle identically for a controlled comparison.
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    shuffle_rng.manual_seed(seed)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    max_epochs, patience = 20, 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs,
                                                           eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()
    run_dir = run_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(protocol_base, model=name,
                    parameters=sum(p.numel() for p in model.parameters()))
    (run_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    last_path, best_path = run_dir / 'last_checkpoint.pth', run_dir / 'best_model.pth'
    start_epoch, best_auc, stale, history = 0, -1., 0, []
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        assert state['protocol'] == protocol, 'Checkpoint does not match source/protocol.'
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        shuffle_rng.set_state(state['shuffle_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_epoch, best_auc, stale = state['epoch'] + 1, state['best_auc'], state['stale']
        history = state['history']
    print(name, 'parameters', protocol['parameters'], 'start epoch', start_epoch + 1,
          flush=True)
    if stale < patience:
        for epoch in range(start_epoch, max_epochs):
            began = time.time()
            model.train()
            total_loss = 0.
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss: {name}, epoch {epoch + 1}')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(x)
            y_dev, p_dev = predict(model)
            assert np.array_equal(y_dev, labels[dev_rows])
            result = metrics(y_dev, p_dev)
            scheduler.step()
            improved = result['macro_auroc'] > best_auc + .0001
            if improved:
                best_auc, stale = result['macro_auroc'], 0
            else:
                stale += 1
            row = {'epoch_one_based': epoch + 1,
                   'train_loss': total_loss / len(train_rows), **result,
                   'minutes': (time.time() - began) / 60}
            history.append(row)
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'shuffle_rng': shuffle_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all(), 'epoch': epoch,
                     'best_auc': best_auc, 'stale': stale, 'history': history,
                     'protocol': protocol}
            if improved:
                torch.save(state, run_dir / 'best_model.tmp')
                os.replace(run_dir / 'best_model.tmp', best_path)
            torch.save(state, run_dir / 'last_checkpoint.tmp')
            os.replace(run_dir / 'last_checkpoint.tmp', last_path)
            (run_dir / 'history.json').write_text(json.dumps(history, indent=2))
            print(name, json.dumps(row), flush=True)
            if stale >= patience:
                print(name, 'early stopped', flush=True)
                break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    y_dev, p_dev = predict(model)
    result = dict(name=name, best_epoch_one_based=best['epoch'] + 1,
                  **metrics(y_dev, p_dev), protocol=protocol)
    np.savez_compressed(run_dir / 'dev8_predictions.npz', ecg_id=ecg_ids[dev_rows],
                        patient_id=patient_ids[dev_rows], y_true=y_dev, y_prob=p_dev)
    (run_dir / 'dev8_metrics.json').write_text(json.dumps(result, indent=2))
    print('SELECTED', json.dumps(result), flush=True)
    del model
    torch.cuda.empty_cache()
    return result

baseline = run_candidate('original_resnet', OriginalResNet)
candidate = run_candidate('multiscale_se_resnet', MultiScaleSEResNet)
with np.load(run_root / 'original_resnet/dev8_predictions.npz', allow_pickle=False) as z:
    ids_a, patients_a, truth_a, prob_a = (z[key] for key in
                                           ('ecg_id', 'patient_id', 'y_true', 'y_prob'))
with np.load(run_root / 'multiscale_se_resnet/dev8_predictions.npz', allow_pickle=False) as z:
    assert np.array_equal(z['ecg_id'], ids_a)
    assert np.array_equal(z['patient_id'], patients_a)
    assert np.array_equal(z['y_true'], truth_a)
    prob_b = z['y_prob']
groups = [np.flatnonzero(patients_a == patient) for patient in np.unique(patients_a)]
rng = np.random.default_rng(4208)
deltas = []
for _ in range(400):
    rows = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
    y = truth_a[rows]
    if np.any((y.sum(0) == 0) | (y.sum(0) == len(y))):
        continue
    deltas.append(float(roc_auc_score(y, prob_b[rows], average='macro')
                        - roc_auc_score(y, prob_a[rows], average='macro')))
assert len(deltas) >= 390
interval = np.percentile(deltas, [2.5, 97.5]).tolist()
advance = (candidate['macro_auroc'] - baseline['macro_auroc'] >= .005
           and interval[0] > 0)
summary = {'original_resnet': baseline, 'multiscale_se_resnet': candidate,
           'candidate_minus_baseline_macro_auroc': candidate['macro_auroc'] - baseline['macro_auroc'],
           'candidate_minus_baseline_patient_bootstrap_95ci': interval,
           'bootstrap_samples': len(deltas),
           'decision_rule': 'advance multiscale only if fold-8 macro AUROC improves by >=0.005 and paired patient-bootstrap interval lower bound is >0',
           'advance_multiscale_to_fold9': advance,
           'note': 'Development fold 8; no fold 9 or fold 10 inference in this experiment.'}
(run_root / 'comparison.json').write_text(json.dumps(summary, indent=2))
print('ARCHITECTURE_SCREEN', json.dumps(summary), flush=True)
