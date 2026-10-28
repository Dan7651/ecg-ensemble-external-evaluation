"""Outcome-free preparation: fixed rules for the deposited extension.

Importing this module reads no data and executes no analysis. CLI entrypoints
must call deposit_gate.require_verified_deposit before opening study scores.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

SEEDS = (101, 202, 303, 404, 505)
METRICS = ('auroc', 'ap', 'f1', 'brier')


def validate(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or y.ndim != 2 or not np.isin(y, (0, 1)).all():
        raise ValueError('Binary labels and scores must have identical N x C shapes')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Nonfinite/out-of-range scores')
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(y),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid record frequency weights')
    return y, p, w


def aligned_scores(reference_ids, reference_patients, reference_y, ids, patients, y, p):
    """Refuse subset/duplicate/label drift and reorder explicitly by ECG ID."""
    if len(set(ids)) != len(ids) or len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Duplicate ECG ID')
    if set(ids) != set(reference_ids):
        raise ValueError('Different ECG populations')
    index = {v: i for i, v in enumerate(ids)}
    order = np.array([index[v] for v in reference_ids])
    if not np.array_equal(np.asarray(patients)[order], reference_patients):
        raise ValueError('Patient alignment mismatch')
    if not np.array_equal(np.asarray(y)[order], reference_y):
        raise ValueError('Label alignment mismatch')
    validate(reference_y, np.asarray(p)[order])
    return np.asarray(p)[order]


def metric_values(y, p, weights=None, thresholds=0.5):
    y, p, w = validate(y, p, weights)
    decision = p >= np.asarray(thresholds)
    tp = (w[:, None] * (decision & (y == 1))).sum(0)
    fp = (w[:, None] * (decision & (y == 0))).sum(0)
    fn = (w[:, None] * (~decision & (y == 1))).sum(0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    pos = (y * w[:, None]).sum(0)
    supported = (pos > 0) & (pos < w.sum())
    auc, ap = np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
    for k in np.flatnonzero(supported):
        auc[k] = roc_auc_score(y[:, k], p[:, k], sample_weight=w)
        ap[k] = average_precision_score(y[:, k], p[:, k], sample_weight=w)
    brier = (((p - y) ** 2) * w[:, None]).sum(0) / w.sum()
    # Ordinary mean propagates unsupported-class NaN; never silently nanmean.
    per_class = np.stack((auc, ap, f1, brier), axis=-1)
    return {'per_class': per_class, 'macro': per_class.mean(0)}


def gain(y, a, b, weights=None):
    ma = metric_values(y, a, weights)['macro']
    mb = metric_values(y, b, weights)['macro']
    me = metric_values(y, (np.asarray(a) + b) / 2, weights)['macro']
    g = me - np.maximum(ma, mb)
    g[-1] = min(ma[-1], mb[-1]) - me[-1]
    return {'component_a': ma, 'component_b': mb, 'ensemble': me, 'gain': g}


def control_blocks(y, resnets, heads, weights=None):
    if len(resnets) != 5 or len(heads) != 5:
        raise ValueError('Exactly five independent fits per family are required')
    result = {}
    for family, aa, bb in [('RR', resnets, resnets), ('JJ', heads, heads), ('RJ', resnets, heads)]:
        result[family] = [gain(y, aa[i], bb[(i + 1) % 5], weights) for i in range(5)]
    return result


def control_summary(blocks):
    summary = {}
    for family in ('RR', 'JJ', 'RJ'):
        values = np.array([v['gain'] for v in blocks[family]])
        summary[family] = dict(individual=values, mean=values.mean(0),
                               sample_sd=values.std(0, ddof=1), minimum=values.min(0), maximum=values.max(0))
    summary['RJ_minus_RR'] = summary['RJ']['mean'] - summary['RR']['mean']
    summary['RJ_minus_JJ'] = summary['RJ']['mean'] - summary['JJ']['mean']
    return summary


def all_pairs(y, resnets, heads):
    out = []
    for family, aa, bb in [('RR', resnets, resnets), ('JJ', heads, heads), ('RJ', resnets, heads)]:
        for i in range(5):
            for j in range(5):
                if family != 'RJ' and i >= j:
                    continue
                out.append(dict(family=family, seed_a=SEEDS[i], seed_b=SEEDS[j], **gain(y, aa[i], bb[j])))
    return out


def patient_weights(patients, rng):
    unique, inv = np.unique(patients, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
    return counts[inv]


def interval(draws, primary=False, minimum_valid=1960):
    x = np.asarray(draws, float)
    good = x[np.isfinite(x)]
    if len(good) < minimum_valid:
        return {'valid': len(good), 'interval': None, 'reason': 'insufficient valid patient draws'}
    limits = (1.25, 98.75) if primary else (2.5, 97.5)
    return {'valid': len(good), 'interval': np.percentile(good, limits).tolist(), 'percentiles': limits}


def full_coverage_starts(samples, window=5000):
    if not isinstance(samples, (int, np.integer)) or samples < window:
        raise ValueError('At least one complete integer-sample window is required')
    return sorted(set(range(0, samples - window + 1, window)) | {samples - window})


def threshold_candidates(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid scores')
    return np.unique(np.r_[p, 0., np.nextafter(1., np.inf)])


def confusion_curve(y, p, weights=None):
    y, p, w = validate(np.asarray(y)[:, None], np.asarray(p)[:, None], weights)
    y, p = y[:, 0], p[:, 0]
    thresholds = threshold_candidates(p)
    order = np.argsort(p, kind='stable')
    s, yy, ww = p[order], y[order], w[order]
    pos = np.r_[0., np.cumsum(ww * yy)]
    neg = np.r_[0., np.cumsum(ww * (1 - yy))]
    index = np.searchsorted(s, thresholds, side='left')
    tp, fp, fn, tn = pos[-1] - pos[index], neg[-1] - neg[index], pos[index], neg[index]
    return thresholds, tp, fp, fn, tn


def oracle_f1(y, p, weights=None):
    t, tp, fp, fn, tn = confusion_curve(y, p, weights)
    if tp[0] <= 0 or tn[-1] <= 0:
        raise ValueError('Both classes required for threshold fitting')
    f = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=2 * tp + fp + fn > 0)
    best = np.flatnonzero(f == f.max())
    k = min(best, key=lambda i: (abs(t[i] - .5), t[i]))
    return float(t[k])


def fixed_specificity_threshold(y, p, target, weights=None):
    if not 0 < target <= 1:
        raise ValueError('Invalid specificity target')
    t, tp, fp, fn, tn = confusion_curve(y, p, weights)
    if tp[0] <= 0 or tn[-1] <= 0:
        raise ValueError('Both classes required')
    spec = tn / (tn + fp)
    candidates = np.flatnonzero(spec >= target)
    k = min(candidates, key=lambda i: (-tp[i], t[i]))
    return dict(threshold=float(t[k]), specificity=float(spec[k]), sensitivity=float(tp[k] / (tp[k] + fn[k])))


def patient_folds(patients, seed=2026102703):
    unique = np.unique(patients)
    if len(unique) < 5:
        raise ValueError('Five distinct patients required')
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {v: i % 5 for i, v in enumerate(shuffled)}
    return np.array([assignment[v] for v in patients])


def prevalence_adjustment(source_prevalence, external_prevalence, thresholds):
    ps, pe = np.asarray(source_prevalence), np.asarray(external_prevalence)
    if ((ps <= 0) | (ps >= 1) | (pe <= 0) | (pe >= 1)).any():
        raise ValueError('Nondegenerate source/external prevalence required')
    delta = logit(pe) - logit(ps)
    return delta, expit(logit(thresholds) - delta)


def calibration(y, p, weights=None):
    y, p, w = validate(np.asarray(y)[:, None], np.asarray(p)[:, None], weights)
    y, p = y[:, 0], p[:, 0]
    if not 0 < np.dot(w, y) < w.sum():
        raise ValueError('Both classes required')
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    offset = brentq(lambda a: np.dot(w, expit(x + a) - y), -100., 100.)
    def objective(ab):
        z = ab[0] + ab[1] * x
        return np.dot(w, np.logaddexp(0, z) - y * z) / w.sum()
    def grad(ab):
        e = w * (expit(ab[0] + ab[1] * x) - y) / w.sum()
        return np.array([e.sum(), np.dot(e, x)])
    fit = minimize(objective, [offset, 1.], jac=grad, method='BFGS', options={'gtol': 1e-9, 'maxiter': 500})
    stable = np.isfinite(fit.x).all() and np.linalg.norm(grad(fit.x), np.inf) < 1e-6
    return {'offset_slope_fixed_1': float(offset), 'joint_intercept': float(fit.x[0]) if stable else None,
            'joint_slope': float(fit.x[1]) if stable else None, 'joint_fit_valid': bool(stable)}


def framing_decision(summary, blocks, rr_interval, jj_interval):
    """Fixed operational wording gate, not a novelty or causal inference test."""
    if summary['RJ_minus_RR'][0] <= 0 or summary['RJ_minus_JJ'][0] <= 0:
        return 'external_validation_of_ensembling'
    consistent = sum(blocks['RJ'][i]['gain'][0] > max(blocks['RR'][i]['gain'][0], blocks['JJ'][i]['gain'][0]) for i in range(5))
    intervals_pass = all(v['interval'] is not None and v['interval'][0] > 0 for v in [rr_interval, jj_interval])
    secondary = all(np.all(summary['RJ']['mean'][1:3] >= summary[f]['mean'][1:3]) for f in ['RR', 'JJ'])
    if intervals_pass and consistent >= 4 and secondary:
        return 'empirical_cross_paradigm_complementarity_with_confounding_disclosed'
    return 'mixed_or_seed_dependent'
