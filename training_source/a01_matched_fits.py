"""Fixed A01 fitting only: no external inference, no fold9/10 prediction.

Entry on a Colab T4 after validating the independently deposited protocol.
Original architecture classes are extracted from the pinned source AST, never
by importing the original script with its mount/training side effects.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import ast
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SEEDS = (101, 202, 303, 404, 505)
FAMILIES = ('resnet', 'jepa_head')
BASE = Path('/content/drive/MyDrive')
CACHE = BASE / 'ECG_Diagnosis_Project/processed_cache/ptb-xl_amplitude_v1'
RESULTS = BASE / 'ECG_Project_Final_Results'
FEATURES = RESULTS / 'jepa_external_frozen_dev8_seed42'
OUT = RESULTS / 'sph_matched_seed_controls_20261027_v1'
LOCAL = Path('/content/sph_a01_existing_cache_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def json_read(path): return json.loads(Path(path).read_text())


def atomic(path, value, kind='json', immutable=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'json':
        data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        if path.exists() and immutable:
            if path.read_bytes() != data: raise RuntimeError('Immutable artifact changed: ' + str(path))
            return
    elif immutable and path.exists(): raise RuntimeError('Existing immutable binary: ' + str(path))
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as f:
        if kind == 'torch': torch.save(value, f)
        elif kind == 'npz': np.savez_compressed(f, **value)
        else: f.write(data)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, np.ndarray): return value.copy()
    return copy.deepcopy(value)


def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, np.ndarray): return np.array_equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def seed_all(seed, generator):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    generator.manual_seed(seed)


def rng_state(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], shuffle=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['shuffle'])


def selected_history(history, minimum_improvement):
    best, selected, stale = -1., 0, 0
    for row in history:
        if row['metrics']['macro_auroc'] > best + minimum_improvement:
            best, selected, stale = row['metrics']['macro_auroc'], row['epoch'], 0
        else: stale += 1
    return best, selected, stale


def architecture_nodes(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
             and n.name in {'ResidualBlock', 'OriginalResNet'}]
    assert [n.name for n in nodes] == ['ResidualBlock', 'OriginalResNet']
    return nodes


def build_model(family):
    if family == 'jepa_head':
        return nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Dropout(.5), nn.Linear(128, 5))
    if family != 'resnet': raise ValueError(family)
    source = HERE / 'multiscale_waveform_dev8.py'
    if sha(source) != '79bb48ea71a033f21dfd734c5fbc4fbb3dc0823be4c2cdd2137f47b5494bc44e':
        raise RuntimeError('Original architecture source drift')
    nodes = architecture_nodes(source.read_text())
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['OriginalResNet']()


def optimizer(model, family):
    return torch.optim.AdamW(model.parameters(), lr=.001 if family == 'resnet' else .0003, weight_decay=.01)


def scaler_new(device, family):
    if device.type == 'cpu':
        # Bounded local guards also run on older CPU-only Torch. T4 uses the
        # same modern CUDA GradScaler as the original recipe below.
        return torch.cuda.amp.GradScaler(enabled=False)
    return torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and family == 'resnet'))


def update(model, opt, scaler, x, y, family, device):
    model.train(); opt.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.float16, enabled=(device.type == 'cuda' and family == 'resnet')):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x.to(device)), y.to(device))
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise FloatingPointError('Missing/nonfinite trainable gradient')
    if family == 'resnet':
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise FloatingPointError('Nonfinite clipped gradients')
    scaler.step(opt); scaler.update()
    return float(loss.detach())


def copy_cached(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha(target) == expected: return
    if target.exists():
        preserved = target.with_name(target.name + '.unverified_' + sha(target)[:12])
        if preserved.exists(): raise RuntimeError('Repeated cache corruption')
        target.rename(preserved)
    tmp = target.with_name(target.name + '.partial'); digest = hashlib.sha256()
    with source.open('rb') as a, tmp.open('wb') as b:
        for block in iter(lambda: a.read(4 * 1024 * 1024), b''):
            digest.update(block); b.write(block)
        b.flush(); os.fsync(b.fileno())
    assert digest.hexdigest() == expected, 'Pinned existing cache hash failed'
    assert sha(tmp) == expected, 'Local cache persistence failed'
    os.replace(tmp, target)


class WaveRows(Dataset):
    def __init__(self, signals, rows, labels, mean, std):
        self.signals, self.rows, self.labels, self.mean, self.std = signals, rows, labels, mean, std
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = int(self.rows[index])
        return torch.from_numpy(((self.signals[row] - self.mean) / self.std).astype(np.float32)), torch.from_numpy(self.labels[row].copy())


def data_prepare():
    pin = json_read(HERE / 'input_pins.json')
    provenance_path = HERE / 'provenance/original_jepa_head_protocol.json'
    assert sha(provenance_path) == pin['original_jepa_head_protocol_sha256']
    provenance = json_read(provenance_path)
    assert provenance['source_sha256'] == '58f225ecf6e572a44a17eb94553a9a5f3fece4f38a4ce65280f7cde33217627c'
    assert provenance['feature_chunk_sha256'] == pin['feature_chunks']
    assert provenance['encoder_checkpoint_sha256'] == '61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e'
    assert provenance['metadata_sha256'] == pin['cache_files']['labels_and_folds.npz']
    assert sha(CACHE / 'manifest.json') == pin['amplitude_manifest_sha256']
    manifest = json_read(CACHE / 'manifest.json')
    for name in ['labels_and_folds.npz', 'ecg_filtered_mv_100hz.npy']:
        assert manifest['files'][name]['sha256'] == pin['cache_files'][name]
        assert (CACHE / name).stat().st_size == manifest['files'][name]['bytes']
        copy_cached(CACHE / name, LOCAL / name, pin['cache_files'][name])
    with np.load(LOCAL / 'labels_and_folds.npz', allow_pickle=False) as z:
        y, folds, ids, patients = (z[k].copy() for k in ['y_standard', 'strat_fold', 'ecg_id', 'patient_id'])
    y = y.astype(np.float32)
    tr = np.flatnonzero(np.isin(folds, range(1, 8)) & (y.sum(1) > 0))
    va = np.flatnonzero((folds == 8) & (y.sum(1) > 0))
    assert (len(tr), len(va)) == (14955, 2129)
    assert not set(patients[tr]).intersection(patients[va])
    assert len(np.unique(ids)) == len(ids) == 21799
    signals = np.load(LOCAL / 'ecg_filtered_mv_100hz.npy', mmap_mode='r')
    assert signals.shape == (21799, 12, 1000) and signals.dtype == np.float32
    normalization_path = OUT / 'normalization.npz'
    normalization_receipt = OUT / 'normalization.receipt.json'
    saved_normalization = None
    if normalization_receipt.exists():
        nr = json_read(normalization_receipt)
        assert nr['input_pins_sha256'] == sha(HERE / 'input_pins.json')
        assert sha(normalization_path) == nr['sha256']
        with np.load(normalization_path, allow_pickle=False) as z:
            saved_normalization = {k: z[k].copy() for k in z.files}
        mean, std = saved_normalization['lead_mean'], saved_normalization['lead_std']
    else:
        # Same FP64 accumulation order as the original folds1-7 reference.
        lead_sum = np.zeros(12, np.float64); lead_sq = lead_sum.copy()
        for rows in np.array_split(tr, 32):
            x = signals[rows].astype(np.float64)
            lead_sum += x.sum(axis=(0, 2)); lead_sq += np.square(x).sum(axis=(0, 2))
        n = len(tr) * 1000; mean64 = lead_sum / n
        std64 = np.sqrt(np.maximum(lead_sq / n - mean64 ** 2, 1e-12))
        mean, std = mean64.astype(np.float32).reshape(12, 1), std64.astype(np.float32).reshape(12, 1)
    assert mean.shape == std.shape == (12, 1) and mean.dtype == std.dtype == np.float32
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all() and (std > 0).all()
    all_rows = np.r_[tr, va]; chunks = []
    for start in range(0, len(all_rows), 1024):
        name = f'{start:06d}.npz'
        copy_cached(FEATURES / 'feature_chunks' / name, LOCAL / 'features' / name, pin['feature_chunks'][name])
        with np.load(LOCAL / 'features' / name, allow_pickle=False) as z:
            assert np.array_equal(z['ecg_id'], ids[all_rows[start:start + 1024]])
            f = z['features'].astype(np.float32)
            assert f.shape == (len(all_rows[start:start + 1024]), 768) and np.isfinite(f).all()
            chunks.append(f)
    f = np.concatenate(chunks)
    if saved_normalization is None:
        transform = StandardScaler().fit(f[:len(tr)])
    else:
        transform = StandardScaler()
        transform.mean_ = saved_normalization['feature_mean']
        transform.scale_ = saved_normalization['feature_scale']
        transform.var_ = saved_normalization['feature_variance']
        transform.n_features_in_, transform.n_samples_seen_ = 768, len(tr)
        assert transform.mean_.shape == transform.scale_.shape == (768,)
        assert np.isfinite(transform.mean_).all() and np.isfinite(transform.scale_).all() and (transform.scale_ > 0).all()
    xt = transform.transform(f[:len(tr)]).astype(np.float32); xv = transform.transform(f[len(tr):]).astype(np.float32)
    signature = hashlib.sha256(ids[all_rows].tobytes() + patients[all_rows].tobytes() + y[all_rows].tobytes()).hexdigest()
    if saved_normalization is None:
        atomic(normalization_path, dict(lead_mean=mean, lead_std=std, feature_mean=transform.mean_,
               feature_scale=transform.scale_, feature_variance=transform.var_), 'npz')
        atomic(normalization_receipt, dict(sha256=sha(normalization_path),
               input_pins_sha256=sha(HERE / 'input_pins.json'), data_identity_sha256=signature), immutable=True)
    else:
        assert nr['data_identity_sha256'] == signature
    return dict(y=y, ids=ids, patients=patients, tr=tr, va=va, signals=signals, mean=mean, std=std,
                xt=xt, xv=xv, data_identity_sha256=signature)


@torch.inference_mode()
def predict(model, family, data, device):
    model.eval(); parts = []
    if family == 'resnet':
        dataset = WaveRows(data['signals'], data['va'], data['y'], data['mean'], data['std'])
        batches = (x for x, _ in DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0))
    else:
        batches = [torch.from_numpy(data['xv'])]  # exact original full development head forward
    for x in batches:
        with torch.autocast(device.type, dtype=torch.float16, enabled=(family == 'resnet' and device.type == 'cuda')):
            logits = model(x.to(device))
        parts.append(logits.float().sigmoid().cpu().numpy())
    p = np.concatenate(parts)
    assert p.shape == (2129, 5) and p.dtype == np.float32 and np.isfinite(p).all()
    return p


def metrics(y, p):
    return dict(macro_auroc=float(roc_auc_score(y, p, average='macro')),
                macro_ap=float(average_precision_score(y, p, average='macro')),
                macro_f1_at_0_5=float(f1_score(y, p >= .5, average='macro', zero_division=0)))


def actual_preflight(family, data, device, global_sha, boot):
    receipt_path = OUT / f'preflight_{family}_{boot}.json'
    if receipt_path.exists():
        r = json_read(receipt_path); assert r['passed'] and r['global_protocol_sha256'] == global_sha; return
    g = torch.Generator(); seed_all(8101, g)
    m = build_model(family).to(device); o = optimizer(m, family); sc = scaler_new(device, family)
    width = (12, 1000) if family == 'resnet' else (768,)
    x = torch.randn((4,) + width); y = torch.randint(0, 2, (4, 5)).float()
    update(m, o, sc, x, y, family, device)
    state = dict(model=cpu_tree(m.state_dict()), opt=cpu_tree(o.state_dict()), scaler=cpu_tree(sc.state_dict()), rng=cpu_tree(rng_state(g)))
    x2 = torch.randn((4,) + width); y2 = torch.randint(0, 2, (4, 5)).float()
    update(m, o, sc, x2, y2, family, device)
    expected = dict(model=cpu_tree(m.state_dict()), opt=cpu_tree(o.state_dict()), scaler=cpu_tree(sc.state_dict()), rng=cpu_tree(rng_state(g)))
    del m, o, sc; torch.cuda.empty_cache()
    replay = build_model(family).to(device); replay.load_state_dict(state['model'], strict=True)
    o = optimizer(replay, family); o.load_state_dict(state['opt']); sc = scaler_new(device, family); sc.load_state_dict(state['scaler'])
    restore_rng(state['rng'], g)
    xx = torch.randn((4,) + width); yy = torch.randint(0, 2, (4, 5)).float()
    assert torch.equal(xx, x2) and torch.equal(yy, y2)
    update(replay, o, sc, xx, yy, family, device)
    observed = dict(model=cpu_tree(replay.state_dict()), opt=cpu_tree(o.state_dict()), scaler=cpu_tree(sc.state_dict()), rng=cpu_tree(rng_state(g)))
    assert equal(expected, observed), 'Actual model/optimizer/scaler/all-RNG continuation failed'
    del replay, o, sc, state, expected, observed; torch.cuda.empty_cache()
    seed_all(8102, g); m = build_model(family).to(device); o = optimizer(m, family); sc = scaler_new(device, family)
    if family == 'resnet':
        dataset = WaveRows(data['signals'], data['tr'][:64], data['y'], data['mean'], data['std'])
        x, y = next(iter(DataLoader(dataset, batch_size=64, num_workers=0)))
    else:
        x, y = torch.from_numpy(data['xt'][:64]), torch.from_numpy(data['y'][data['tr'][:64]])
    for _ in range(2): update(m, o, sc, x, y, family, device)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); tick = time.monotonic()
    for _ in range(4): update(m, o, sc, x, y, family, device)
    torch.cuda.synchronize(); seconds = time.monotonic() - tick
    peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) / 2**30
    estimate = seconds / 4 * ((14955 + 63) // 64) / 60 if family == 'resnet' else None
    passed = peak <= 14 and (estimate is None or estimate <= 15)
    r = dict(passed=bool(passed), family=family, boot_id=boot, global_protocol_sha256=global_sha,
             actual_synthetic_model_optimizer_scaler_all_rng_exact_replay=True, finite_gradients=True,
             unique_training_records=64, warmup_updates=2, measured_updates=4, measured_seconds=seconds,
             peak_gib=peak, estimated_resnet_training_minutes_per_epoch=estimate,
             disposable_updates_discarded=True, no_development_used_for_capacity=True)
    atomic(receipt_path, r, immutable=True)
    del m, o, sc, x, y; torch.cuda.empty_cache()
    assert passed, 'Fixed capacity gate failed; do not relax protocol'
    print('A01_REAL_T4_PREFLIGHT_PASSED ' + json.dumps(r), flush=True)


def load_commit(folder, protocol_sha, family):
    choices = []
    for slot in (0, 1):
        path = folder / f'last_{slot}.pth'; receipt = path.with_suffix('.receipt.json')
        if not path.exists() or not receipt.exists(): continue
        info = json_read(receipt)
        if sha(path) != info['sha256']: continue  # preserve corrupt/incomplete generation; try the other.
        s = torch.load(path, map_location='cpu', weights_only=False)
        assert s['protocol_sha256'] == protocol_sha and s['family'] == family
        assert info['protocol_sha256'] == protocol_sha
        assert s['epoch'] == info['epoch'] and len(s['history']) == s['epoch']
        gap = .0001 if family == 'resnet' else .000001
        assert selected_history(s['history'], gap) == (s['best_auc'], s['selected_epoch'], s['stale'])
        assert s['updates'] == s['epoch'] * ((14955 + (63 if family == 'resnet' else 255)) // (64 if family == 'resnet' else 256))
        choices.append(s)
    if not choices:
        if list(folder.glob('last_*')): raise RuntimeError('No complete recoverable checkpoint; preserve evidence')
        return None
    return max(choices, key=lambda s: s['epoch'])


def verify_endpoint(folder, global_sha, family, seed):
    protocol_path = folder / 'protocol.json'; protocol = json_read(protocol_path)
    assert protocol['global_protocol_sha256'] == global_sha
    assert protocol['family'] == family and protocol['seed'] == seed
    assert protocol['source_sha256'] == sha(__file__)
    e = json_read(folder / 'endpoint.json')
    assert e['protocol_sha256'] == sha(protocol_path)
    assert e['family'] == family and e['seed'] == seed and e['source_sha256'] == sha(__file__)
    assert e['exact_fresh_saved_best_replay'] and e['no_external_inference'] and e['no_fold9_or_10_prediction']
    assert sha(folder / 'best_model.pth') == e['checkpoint_sha256']
    assert sha(folder / 'development_predictions.npz') == e['prediction_sha256']
    history = json_read(folder / 'history.json')
    assert len(history) == e['completed_epochs']
    best, selected, stale = selected_history(history, protocol['selection_gap'])
    assert selected == e['selected_epoch'] and best == e['development_metrics']['macro_auroc']
    assert e['completed_epochs'] == protocol['max_epochs'] or stale >= protocol['patience']
    assert e['updates'] == history[-1]['updates'] == len(history) * ((14955 + protocol['batch'] - 1) // protocol['batch'])
    return e


def fit(family, seed, data, global_sha, device):
    folder = OUT / family / f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(global_protocol_sha256=global_sha, family=family, seed=seed,
                    train_folds=list(range(1, 8)), development_fold=8, train_records=14955,
                    development_records=2129, selection_gap=.0001 if family == 'resnet' else .000001,
                    max_epochs=20 if family == 'resnet' else 40, patience=6 if family == 'resnet' else 8,
                    batch=64 if family == 'resnet' else 256, drop_last=False,
                    data_identity_sha256=data['data_identity_sha256'], source_sha256=sha(__file__))
    atomic(folder / 'protocol.json', protocol, immutable=True); protocol_sha = sha(folder / 'protocol.json')
    if (folder / 'endpoint.json').exists():
        e = verify_endpoint(folder, global_sha, family, seed)
        print('A01_COMPLETED_FIT_REUSED', family, seed, flush=True); return e
    g = torch.Generator(); seed_all(seed, g); model = build_model(family).to(device)
    opt = optimizer(model, family); sc = scaler_new(device, family)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5) if family == 'resnet' else None
    epoch, best_auc, selected, stale, history, best, updates = 0, -1., 0, 0, [], None, 0
    if family == 'resnet':
        dataset = WaveRows(data['signals'], data['tr'], data['y'], data['mean'], data['std'])
        loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=g, num_workers=0, drop_last=False)
    else:
        tx, ty = torch.from_numpy(data['xt']).to(device), torch.from_numpy(data['y'][data['tr']]).to(device)
    s = load_commit(folder, protocol_sha, family)
    if s:
        assert s['seed'] == seed
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); sc.load_state_dict(s['scaler'])
        if sched is not None: sched.load_state_dict(s['scheduler'])
        epoch, best_auc, selected, stale, history, best, updates = (s[k] for k in ['epoch', 'best_auc', 'selected_epoch', 'stale', 'history', 'best_model', 'updates'])
        restore_rng(s['rng'], g)
        print('A01_RESUME_AFTER_COMPLETE_EPOCH', family, seed, epoch, flush=True)
    while epoch < protocol['max_epochs'] and stale < protocol['patience']:
        tick = time.monotonic(); total = 0.; used = 0; rate = float(opt.param_groups[0]['lr'])
        if family == 'resnet': batches = loader
        else:
            order = torch.randperm(len(tx), device=device)
            batches = ((tx[order[i:i+256]], ty[order[i:i+256]]) for i in range(0, len(tx), 256))
        for x, y in batches:
            total += update(model, opt, sc, x, y, family, device) * len(x); used += len(x); updates += 1
        assert used == 14955
        p = predict(model, family, data, device); point = metrics(data['y'][data['va']], p)
        epoch += 1
        if sched is not None: sched.step()
        if point['macro_auroc'] > best_auc + protocol['selection_gap']:
            best_auc, selected, stale, best = point['macro_auroc'], epoch, 0, cpu_tree(model.state_dict())
        else: stale += 1
        history.append(dict(epoch=epoch, train_bce=total/used, metrics=point, updates=updates,
                            learning_rate_used=rate, seconds=time.monotonic()-tick))
        s = dict(protocol_sha256=protocol_sha, family=family, seed=seed, epoch=epoch,
                 model=cpu_tree(model.state_dict()), optimizer=cpu_tree(opt.state_dict()), scaler=cpu_tree(sc.state_dict()),
                 scheduler=cpu_tree(sched.state_dict()) if sched is not None else None,
                 rng=cpu_tree(rng_state(g)), best_auc=best_auc, selected_epoch=selected, stale=stale,
                 best_model=cpu_tree(best), history=copy.deepcopy(history), updates=updates)
        target = folder / f'last_{epoch % 2}.pth'; atomic(target, s, 'torch')
        digest = sha(target)
        atomic(target.with_suffix('.receipt.json'), dict(epoch=epoch, sha256=digest, protocol_sha256=protocol_sha))
        assert sha(target) == digest
        atomic(folder / 'history.json', history)
        atomic(OUT / 'status.json', dict(phase='training', family=family, seed=seed, completed_epoch=epoch,
               selected_epoch=selected, updates=updates, updated_utc=datetime.now(timezone.utc).isoformat()))
        print('A01_EPOCH ' + json.dumps(dict(family=family, seed=seed, selected_epoch=selected, **history[-1])), flush=True)
    assert best is not None and selected_history(history, protocol['selection_gap']) == (best_auc, selected, stale)
    # A crash may have occurred between checkpoint commit and the redundant
    # readable history export. Restore that export even if early stopping has
    # already completed; this never performs another optimizer update.
    atomic(folder / 'history.json', history)
    model.load_state_dict(best); p = predict(model, family, data, device)
    del model, opt, sc, sched; torch.cuda.empty_cache()
    checkpoint = dict(model=best, protocol=protocol, selected_epoch=selected, epoch_history=history)
    atomic(folder / 'best_model.pth', checkpoint, 'torch')
    replay_model = build_model(family).to(device)
    exported = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=False)
    replay_model.load_state_dict(exported['model'], strict=True)
    replay = predict(replay_model, family, data, device)
    assert np.array_equal(p, replay), 'Fresh saved-best prediction replay failed'
    atomic(folder / 'development_predictions.npz', dict(ecg_id=data['ids'][data['va']], patient_id=data['patients'][data['va']],
           y_true=data['y'][data['va']], y_prob=p), 'npz')
    e = dict(family=family, seed=seed, protocol_sha256=protocol_sha, completed_epochs=epoch,
             selected_epoch=selected, updates=updates, development_metrics=metrics(data['y'][data['va']], p),
             checkpoint_sha256=sha(folder / 'best_model.pth'), prediction_sha256=sha(folder / 'development_predictions.npz'),
             exact_fresh_saved_best_replay=True, source_sha256=sha(__file__), no_external_inference=True,
             no_fold9_or_10_prediction=True, checkpoint_persistence_after_remount_verified=False)
    atomic(folder / 'endpoint.json', e, immutable=True)
    del replay_model; torch.cuda.empty_cache()
    print('A01_FIT_COMPLETED ' + json.dumps(e), flush=True)
    return e


def main():
    if sys.flags.optimize:
        raise RuntimeError('Do not disable protocol/integrity assertions with optimized Python')
    ap = argparse.ArgumentParser(); ap.add_argument('--fit', action='store_true'); args = ap.parse_args()
    sys.path.insert(0, str(HERE / 'deposited_protocol'))
    from deposit_gate import require_verified_deposit
    deposit = require_verified_deposit(HERE / 'deposited_protocol')
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'Use the existing authorized T4'
    assert CACHE.is_dir() and FEATURES.is_dir(), 'Reuse a readable Drive mount; do not redownload or preprocess'
    lock = open('/content/sph_a01_matched_fits.lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=True); LOCAL.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), numpy=np.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, gpu=torch.cuda.get_device_name(0), deterministic=True,
               cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    atomic(OUT / 'environment.json', env, immutable=True)
    protocol = dict(deposit=deposit['doi'], deposited_bundle_sha256=deposit['bundle_sha256'],
                    seeds=list(SEEDS), families=list(FAMILIES), source_sha256=sha(__file__),
                    input_pins_sha256=sha(HERE / 'input_pins.json'), environment_sha256=sha(OUT / 'environment.json'),
                    recipe='Deposited PROTOCOL.md section4.1; no external access in this runner',
                    implementation_note='num_workers=0 for exact resumability; no stochastic waveform augmentation; pinned original model classes')
    atomic(OUT / 'training_protocol.json', protocol, immutable=True); global_sha = sha(OUT / 'training_protocol.json')
    if (OUT / 'all_ten_fits_locked.json').exists():
        completed = json_read(OUT / 'all_ten_fits_locked.json')
        assert completed['global_protocol_sha256'] == global_sha
        endpoints = [verify_endpoint(OUT / family / f'seed{seed}', global_sha, family, seed)
                     for family in FAMILIES for seed in SEEDS]
        assert equal(completed['endpoints'], endpoints) and completed['no_external_inference']
        print('A01_ALL_TEN_COMPLETED_FITS_REUSED_NO_TRAINING', flush=True)
        return
    data = data_prepare(); boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for family in FAMILIES: actual_preflight(family, data, torch.device('cuda'), global_sha, boot)
    print('A01_ALL_T4_PREFLIGHT_GATES_PASSED', flush=True)
    if not args.fit: return
    endpoints = []
    for family in FAMILIES:
        for seed in SEEDS: endpoints.append(fit(family, seed, data, global_sha, torch.device('cuda')))
    atomic(OUT / 'all_ten_fits_locked.json', dict(global_protocol_sha256=global_sha, endpoints=endpoints,
           no_external_inference=True, completed_utc=datetime.now(timezone.utc).isoformat()), immutable=True)
    atomic(OUT / 'status.json', dict(phase='all_ten_fits_complete_awaiting_persistence_verification',
           no_external_inference=True, updated_utc=datetime.now(timezone.utc).isoformat()))
    print('A01_ALL_TEN_FITS_COMPLETE_NO_EXTERNAL_INFERENCE', flush=True)


if __name__ == '__main__': main()
