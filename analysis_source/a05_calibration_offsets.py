"""Deposited A05: saved-score calibration; no model or waveform execution.

Run in the isolated CPU environment with one BLAS thread. Fixed-slope offsets
are compared with prior shift alone and source offset plus prior shift.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
HOME = ROOT.parents[1]
sys.path.insert(0, str(HOME))
from deposit_gate import require_verified_deposit
from analysis_core import calibration, interval, validate

CLASSES = ('NORM', 'MI', 'STTC', 'CD', 'HYP')
ARMS = ('resnet', 'jepa', 'blend')
SEED, DRAWS = 2026102705, 2000


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:
        if binary: np.savez_compressed(f, **value)
        else: f.write((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def fixed_offsets(x, y, weights):
    """Safeguarded simultaneous Newton roots; unsupported columns remain NaN.

    x and y are C x N, so each class vector is contiguous. The score equation
    is the weighted mean of sigmoid(x+offset)-y. No joint slope is estimated.
    """
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if x.shape != y.shape or x.ndim != 2 or w.shape != (x.shape[1],):
        raise ValueError('Calibration dimensions differ')
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all() or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid calibration inputs')
    total = w.sum(); prevalence = (y*w).sum(1)/total
    supported = (prevalence > 0) & (prevalence < 1)
    result = np.full(len(x), np.nan)
    if not supported.any(): return result
    xx, yy = x[supported], y[supported]
    lo, hi = np.full(len(xx), -100.), np.full(len(xx), 100.)
    a = np.zeros(len(xx)); converged = np.zeros(len(xx), bool)
    for _ in range(80):
        p = expit(xx+a[:, None])
        gradient = ((p-yy)*w).sum(1)/total
        converged |= abs(gradient) < 1e-12
        if converged.all(): break
        lo = np.where((gradient < 0) & ~converged, a, lo)
        hi = np.where((gradient >= 0) & ~converged, a, hi)
        curvature = (p*(1-p)*w).sum(1)/total
        proposal = a-np.divide(gradient, curvature, out=np.zeros_like(a), where=curvature>0)
        proposal = np.where(np.isfinite(proposal) & (proposal>lo) & (proposal<hi), proposal, (lo+hi)/2)
        a = np.where(converged, a, proposal)
    final = ((expit(xx+a[:, None])-yy)*w).sum(1)/total
    result[np.flatnonzero(supported)[abs(final)<1e-10]] = a[abs(final)<1e-10]
    return result


def logit_matrix(scores):
    return np.ascontiguousarray(np.concatenate([logit(np.clip(p.astype(float), 1e-6, 1-1e-6)) for p in scores], axis=1).T)


def independent_expanded_check(x, yy, record_weights, observed):
    take = np.repeat(np.arange(len(record_weights)), record_weights.astype(int))
    errors = []
    for c in range(len(x)):
        if not np.isfinite(observed[c]): continue
        z, y = x[c, take], yy[c, take]
        fitted = brentq(lambda a: np.mean(expit(z+a)-y), -100., 100., xtol=1e-12)
        errors.append(abs(fitted-observed[c]))
    if errors and max(errors)>1e-8: raise AssertionError('Expanded-patient offset root mismatch')
    return max(errors, default=0.)


def load_inputs():
    from sph_component_analysis_v1 import mapping_and_rows, load_predictions
    study = HOME/'experiments/sph_component_manuscript_20261026_v1'
    original_path = HOME/'experiments/sph_publication_scope_20261026_v1/external_assessment.json'
    assert sha(original_path)=='ce91b9c2894ef07bfaa986bb71a50e58a9b861ba501b97756199c2f9c11a8fab'
    _, rows = mapping_and_rows(study)
    arrays, patients, chunks = load_predictions(study, json.loads(original_path.read_text()), rows)
    ey, er, ej = arrays['y'], arrays['resnet'], arrays['jepa']
    external = [er, ej, (er+ej)*.5]  # preserve original FP32 probability fusion.
    source_path = HOME/'local_verified_cache/dev8_fusion_predictions.npz'
    assert sha(source_path)=='495b150a00349cadb55fb8d7359d63e3b9a2ca1d23e3368169eea4c5600eca5b'
    rule_path = HOME/'frozen_development_decision_rule_observed.json'
    lock_path = HOME/'experiments/sph_publication_scope_20261026_v1/inference_lock.json'
    rule, lock = json.loads(rule_path.read_text()), json.loads(lock_path.read_text())
    assert rule['source_prediction_sha256']==sha(source_path)
    assert rule['jepa_head_sha256']==lock['jepa_head_sha256']=='2b4eab786f806df1a3fba6ba769931b869520cf04982bcf00bd6e04c59717de7'
    assert rule['resnet_checkpoint_sha256']==lock['resnet_sha256']=='fd83f0be7ab01b3ba2445aa1c9582800222d300c3088e96c154711724fdc10c5'
    with np.load(source_path, allow_pickle=False) as z:
        sy, sp, ids = z['y_true'].copy(), z['patient_id'].copy(), z['ecg_id'].copy()
        source = [z[k].copy() for k in ['resnet', 'frozen_jepa', 'raw_equal']]
    assert np.array_equal(source[2], (source[0]+source[1])*.5)
    assert len(sy)==len(np.unique(ids))==2129
    assert np.array_equal(sy.sum(0), [932, 538, 514, 492, 268])
    assert np.array_equal(ey.sum(0), [13905, 260, 5084, 2465, 229])
    for y, scores in [(sy, source), (ey, external)]:
        for p in scores: validate(y, p)
    pins = dict(source_prediction_sha256=sha(source_path), original_assessment_sha256=sha(original_path),
                frozen_rule_sha256=sha(rule_path), inference_lock_sha256=sha(lock_path),
                external_input_chunks=chunks, rule_and_lock_checkpoint_pins_matched=True)
    return sy, sp, source, ey, patients, external, pins


def main():
    deposit = require_verified_deposit(ROOT)  # Must precede study-score access.
    folder = ROOT/'results/a05_calibration'; folder.mkdir(exist_ok=True, parents=True)
    handle = (folder/'analysis.lock').open('a+')
    fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'assessment.json').exists():
        saved = json.loads((folder/'assessment.json').read_text())
        assert saved['source_sha256']==sha(__file__)
        assert sha(folder/'patient_draws.npz')==saved['draws_sha256']
        print('A05_CALIBRATION_COMPLETED_REUSED_NO_ANALYSIS'); return
    started = time.monotonic()
    sy, sp, source, ey, ep, external, pins = load_inputs()
    plan = dict(deposit_doi=deposit['doi'], protocol_sha256=deposit['protocol_sha256'],
                source_sha256=sha(__file__), analysis_core_sha256=sha(ROOT/'analysis_core.py'),
                inputs=pins, source_records=len(sy), external_records=len(ey),
                draws=DRAWS, seed=SEED, rng_order='source patients then external patients within each draw',
                percentiles=[2.5, 97.5], minimum_valid=1960, probability_clip=[1e-6, 1-1e-6])
    if (folder/'execution_plan.json').exists():
        assert json.loads((folder/'execution_plan.json').read_text())==plan
    else: atomic(folder/'execution_plan.json', plan)
    sx, ex = logit_matrix(source), logit_matrix(external)
    syt, eyt = np.tile(sy.T, (3, 1)), np.tile(ey.T, (3, 1))
    source_offset = fixed_offsets(sx, syt, np.ones(len(sy)))
    external_offset = fixed_offsets(ex, eyt, np.ones(len(ey)))
    delta = logit(ey.mean(0))-logit(sy.mean(0))
    joint = {'source':[], 'external':[]}; point_checks=[]
    for domain, y, scores, observed in [('source', sy, source, source_offset), ('external', ey, external, external_offset)]:
        for a, p in enumerate(scores):
            fits = [calibration(y[:, c], p[:, c].astype(float)) for c in range(5)]
            joint[domain].append(fits)
            for c, fitted in enumerate(fits):point_checks.append(abs(fitted['offset_slope_fixed_1']-observed[a*5+c]))
    assert max(point_checks)<1e-8
    _, si = np.unique(sp, return_inverse=True); _, ei = np.unique(ep, return_inverse=True)
    ns, ne = si.max()+1, ei.max()+1
    rng = np.random.default_rng(SEED)
    draws = dict(source_offset=np.full((DRAWS, 15), np.nan), external_offset=np.full((DRAWS, 15), np.nan), delta=np.full((DRAWS, 5), np.nan))
    completed, max_expanded_error = 0, 0.
    checkpoint = folder/'progress.npz'; checkpoint_receipt = folder/'progress.json'
    if checkpoint.exists() or checkpoint_receipt.exists():
        state = json.loads(checkpoint_receipt.read_text()); assert state['checkpoint_sha256']==sha(checkpoint)
        assert state['plan_sha256']==sha(folder/'execution_plan.json')
        completed = state['completed']; rng.bit_generator.state=state['rng']
        max_expanded_error=state['max_expanded_error']
        with np.load(checkpoint, allow_pickle=False) as z:
            for name in draws: draws[name][:completed]=z[name]
        print('A05_CALIBRATION_RESUME_AFTER_DRAW', completed, flush=True)
    for b in range(completed, DRAWS):
        sw = np.bincount(rng.integers(ns, size=ns), minlength=ns)[si]
        ew = np.bincount(rng.integers(ne, size=ne), minlength=ne)[ei]
        draws['source_offset'][b] = fixed_offsets(sx, syt, sw)
        draws['external_offset'][b] = fixed_offsets(ex, eyt, ew)
        ps, pe = (sy*sw[:, None]).sum(0)/sw.sum(), (ey*ew[:, None]).sum(0)/ew.sum()
        supported = (ps>0)&(ps<1)&(pe>0)&(pe<1)
        draws['delta'][b, supported] = logit(pe[supported])-logit(ps[supported])
        if b in (0, DRAWS-1):
            for x, y, w, offsets in [(sx,syt,sw,draws['source_offset'][b]), (ex,eyt,ew,draws['external_offset'][b])]:
                max_expanded_error=max(max_expanded_error,independent_expanded_check(x,y,w,offsets))
        if (b+1)%100==0:
            atomic(checkpoint, {k:v[:b+1] for k,v in draws.items()}, True)
            atomic(checkpoint_receipt, dict(completed=b+1, checkpoint_sha256=sha(checkpoint),
                plan_sha256=sha(folder/'execution_plan.json'), rng=rng.bit_generator.state, max_expanded_error=max_expanded_error))
            print('A05_CALIBRATION_PATIENT_DRAWS', b+1, 'seconds', round(time.monotonic()-started,2), flush=True)
    atomic(folder/'patient_draws.npz', draws, True)
    rows=[]
    for a, arm in enumerate(ARMS):
        for c, label in enumerate(CLASSES):
            idx=a*5+c
            values={'source_offset':(source_offset[idx], draws['source_offset'][:,idx]),
                    'external_offset':(external_offset[idx], draws['external_offset'][:,idx]),
                    'prior_shift_delta':(delta[c], draws['delta'][:,c]),
                    'external_minus_prior_shift':(external_offset[idx]-delta[c], draws['external_offset'][:,idx]-draws['delta'][:,c]),
                    'external_minus_source_plus_prior_shift':(external_offset[idx]-source_offset[idx]-delta[c], draws['external_offset'][:,idx]-draws['source_offset'][:,idx]-draws['delta'][:,c])}
            rows.append(dict(arm=arm, label=label, estimates={k:dict(point=float(v), **interval(d)) for k,(v,d) in values.items()},
                             source_joint=joint['source'][a][c], external_joint=joint['external'][a][c]))
    result=dict(status='A05_CALIBRATION_COMPLETED', completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), deposit=deposit, execution_plan_sha256=sha(folder/'execution_plan.json'),
        source_records=len(sy), source_patients=int(ns), external_records=len(ey), external_patients=int(ne),
        source_prevalence=sy.mean(0).tolist(), external_prevalence=ey.mean(0).tolist(),
        arms=list(ARMS), classes=list(CLASSES), rows=rows, draws=DRAWS, bootstrap_seed=SEED,
        draws_sha256=sha(folder/'patient_draws.npz'), maximum_point_sealed_core_root_error=max(point_checks),
        maximum_expanded_patient_draw_root_error=max_expanded_error, literal_expanded_draws_checked=[0,DRAWS-1],
        elapsed_seconds=time.monotonic()-started,
        interpretation='Offsets are descriptive calibration-in-the-large with slope fixed1; joint slope/intercept is separate. Prior-shift predictions assume unchanged class-conditional score distributions and comparable targets. Viewed cohorts and selected seed42 fits; no fresh confirmation, new model, deployment or benchmark claim.',
        no_model_waveform_training_or_inference=True, matched_seed_control_results_still_pending=True)
    atomic(folder/'assessment.json', result)
    print('A05_CALIBRATION_COMPLETED', sha(folder/'assessment.json'), flush=True)


if __name__=='__main__': main()
