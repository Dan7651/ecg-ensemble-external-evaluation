"""Read-only secondary SPH component analysis; prediction arrays only.

Run using the isolated CPU environment. Inputs are the authenticated saved
prediction chunks and original public metadata, never ECGs or checkpoints.
"""
import argparse
import collections
import csv
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sph_component_statistics_v1 import ScoreStatistics, METRICS, cluster_weights, intervals

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'experiments/sph_component_manuscript_20261026_v1'
ORIGINAL = HERE / 'experiments/sph_publication_scope_20261026_v1'
PLAN_SHA = '28f7520b15a60a7458ecb0ab2828537a1c224a6d769796d0abbd0555054a0e6a'
CLASS = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
ARMS = ['resnet', 'jepa', 'blend', 'blend_frozen_thresholds']
CONTRASTS = [(2, 0, 'blend_minus_resnet'), (2, 1, 'blend_minus_jepa'),
             (3, 2, 'frozen_thresholds_minus_0_5')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if p.exists() and p.read_bytes() != b:
        raise ValueError('Refuse to overwrite differing artifact: ' + str(p))
    p.write_bytes(b)


def mapping_and_rows(study):
    source = study / 'sources'
    assert sha(source / 'code.csv') == '4d22759fd19d37133c78d439a5726f9e0bc412cc17483c424926ce9abb33c8fc'
    assert sha(source / 'metadata.csv') == 'c6f8dc5197758a3970aae554f8b6ee884d96656b121879bc0eb1adf367ea6d86'
    mapping = json.loads((HERE / 'sph_primary_code_map.json').read_text())
    assert sha(HERE / 'sph_primary_code_map.json') == 'cfa4b658400c94bab5e1a0635b9d287d36e70648bb96463278d4ee4d94d9e1d1'
    with (source / 'code.csv').open(newline='', encoding='utf-8-sig') as f:
        dictionary = {int(r['Code']): r for r in csv.DictReader(f)}
    targets = {c: i for i, group in enumerate(CLASS) for c in mapping['groups'][group]}
    rows = []; counts_all = collections.Counter(); counts_eligible = collections.Counter()
    patients = collections.defaultdict(set); exclusions = collections.Counter()
    modifiers = collections.Counter(); total = 0; subjects = set(); seen = set()
    with (source / 'metadata.csv').open(newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            total += 1; subjects.add(row['Patient_ID'])
            assert row['ECG_ID'] not in seen; seen.add(row['ECG_ID'])
            tokens = row['AHA_Code'].split(';')
            assert all(re.fullmatch(r'\d+(?:\+\d+)*', t) for t in tokens)
            primary = [int(t.split('+')[0]) for t in tokens]
            assert all(c in dictionary and c < 300 for c in primary)
            mod = [int(m) for t in tokens for m in t.split('+')[1:]]
            assert all(c in dictionary and c >= 300 for c in mod)
            assert not (1 in primary and any(c != 1 for c in primary))
            counts_all.update(set(primary)); modifiers.update(mod)
            y = [0] * 5
            for c in primary:
                if c in targets: y[targets[c]] = 1
            reason = ('fewer_than_5000_samples' if int(row['N']) < 5000 else
                      'outside_target_ontology' if not any(y) else None)
            if reason:
                exclusions[reason] += 1
                continue
            rows.append({'id': row['ECG_ID'], 'patient_id': row['Patient_ID'], 'labels': y})
            counts_eligible.update(set(primary))
            for c in set(primary): patients[c].add(row['Patient_ID'])
    assert total == 25770 and len(subjects) == 24666 and len(rows) == 21183
    assert len({r['patient_id'] for r in rows}) == 20401
    taxonomy_file = HERE.parents[1] / 'work/ptbxl_cache_build/data/raw/ptb-xl/scp_statements.csv'
    assert sha(taxonomy_file) == 'ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f'
    with taxonomy_file.open(newline='') as f:
        taxonomy = list(csv.DictReader(f))
    code_rows = []
    for c, d in sorted(dictionary.items()):
        matches = [dict(code=t[''], description=t['description'], parent=t['diagnostic_class'])
                   for t in taxonomy if c in {int(x) for x in re.findall(r'\d+', t['AHA code'])}]
        code_rows.append({'aha_code': c, 'source_description': d['Description'],
                          'kind': 'modifier' if c >= 300 else 'primary',
                          'frozen_target': CLASS[targets[c]] if c in targets else None,
                          'all_source_records_with_primary': counts_all[c],
                          'eligible_records_with_primary': counts_eligible[c],
                          'eligible_patients_with_primary': len(patients[c]),
                          'modifier_occurrences': modifiers[c],
                          'ptbxl_taxonomy_AHA_cross_references': matches,
                          'semantic_equivalence_certified': False})
    audit = {'source_records': total, 'source_patients': len(subjects),
             'eligible_records': len(rows), 'eligible_patients': 20401,
             'exclusion_reason_order': 'sample count first, then presence of mapped target',
             'exclusions': dict(exclusions), 'source_codes': code_rows,
             'positive_counts': np.asarray([r['labels'] for r in rows]).sum(0).astype(int).tolist(),
             'source_metadata_sha256': sha(source / 'metadata.csv'),
             'source_dictionary_sha256': sha(source / 'code.csv'),
             'original_map_sha256': sha(HERE / 'sph_primary_code_map.json'),
             'ptbxl_taxonomy_sha256': sha(taxonomy_file),
             'expert_review_completed': False,
             'targets_changed': False,
             'note': 'Literal dictionary cross-references are supporting metadata, not a certified clinical equivalence relation. Absence of a selected code is absence of the mapped annotation, not exclusion of all pathology.'}
    return audit, rows


def load_predictions(study, assessment, rows):
    expected = assessment['prediction_chunks']
    files = list((study / 'saved_predictions').glob('*.npz'))
    assert len(expected) == len(files) == 83
    all_parts = collections.defaultdict(list); ids = []; patients = []
    manifest = []
    for part in expected:
        p = study / 'saved_predictions' / part['filename']
        assert sha(p) == part['sha256']
        assert part['inference_lock_sha256'] == assessment['inference_lock_sha256']
        with np.load(p, allow_pickle=False) as z:
            assert set(z.files) == {'y', 'resnet', 'jepa', 'ids', 'patient_id'}
            for key in ['y', 'resnet', 'jepa']: all_parts[key].append(z[key])
            ids.extend(z['ids'].tolist()); patients.extend(z['patient_id'].tolist())
        manifest.append({'filename': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    assert ids == [r['id'] for r in rows] and patients == [r['patient_id'] for r in rows]
    assert len(set(ids)) == len(ids) == 21183 and len(set(patients)) == 20401
    arrays = {k: np.concatenate(v) for k, v in all_parts.items()}
    assert np.array_equal(arrays['y'], np.asarray([r['labels'] for r in rows], dtype=np.uint8))
    assert arrays['resnet'].dtype == arrays['jepa'].dtype == np.float32
    assert all(a.shape == (21183, 5) for a in arrays.values())
    assert arrays['y'].sum(0).astype(int).tolist() == assessment['positive_counts']
    return arrays, np.asarray(patients), manifest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', type=Path, default=DEFAULT)
    ap.add_argument('--output', type=Path); args = ap.parse_args()
    study = args.study; output = args.output or study / 'results'
    assert sha(study / 'secondary_analysis_plan.json') == PLAN_SHA
    assert not (output / 'component_assessment.json').exists(), 'Completed analysis exists; use a separate --output for intentional reproduction.'
    output.mkdir(parents=True, exist_ok=True)
    assert sha(ORIGINAL / 'external_assessment.json') == 'ce91b9c2894ef07bfaa986bb71a50e58a9b861ba501b97756199c2f9c11a8fab'
    original = json.loads((ORIGINAL / 'external_assessment.json').read_text())
    assert sha(ORIGINAL / 'inference_lock.json') == original['inference_lock_sha256']
    mapping, rows = mapping_and_rows(study)
    arrays, patients, manifest = load_predictions(study, original, rows)
    save(output / 'mapping_technical_audit.json', mapping)
    save(output / 'input_verification.json', {'plan_sha256': PLAN_SHA, 'records': 21183, 'patients': 20401,
         'chunks': manifest, 'metadata_reconstructed_labels_match_every_saved_prediction': True,
         'new_waveform_or_checkpoint_execution': False})
    y = arrays['y']; blend = (arrays['resnet'] + arrays['jepa']) * .5
    scores = [arrays['resnet'], arrays['jepa'], blend, blend]
    thresholds = [.5, .5, .5, original['thresholds']]
    stats = [ScoreStatistics(y, p, t) for p, t in zip(scores, thresholds)]
    point = [s.evaluate() for s in stats]
    for i, (p, t) in enumerate(zip(scores, thresholds)):
        expected = [roc_auc_score(y, p, average='macro'), average_precision_score(y, p, average='macro'),
                    f1_score(y, p >= t, average='macro', zero_division=0)]
        np.testing.assert_allclose(point[i]['macro'][:3], expected, rtol=0, atol=1e-12)
    for i, name in [(0, 'resnet_at_0_5'), (2, 'candidate_at_0_5'), (3, 'candidate_fixed_thresholds')]:
        reference = original['point'][name]
        np.testing.assert_allclose(point[i]['macro'][:3],
                [reference['macro_auroc'], reference['macro_ap'], reference['macro_f1']], rtol=0, atol=1e-12)
        assert abs(point[i]['subset_accuracy'] - reference['subset_accuracy']) < 1e-12
    print('SPH_SAVED_COMPONENT_INPUTS_AND_ORIGINAL_RESULTS_VERIFIED', flush=True)
    unique_patients, inv = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(20261026)
    macro = np.empty((2000, 4, 8)); per_class = np.empty((2000, 4, 5, 7))
    started = time.monotonic()
    for b in range(2000):
        counts = np.bincount(rng.integers(len(unique_patients), size=len(unique_patients)), minlength=len(unique_patients))
        weights = cluster_weights(inv, counts)
        for i, stat in enumerate(stats):
            result = stat.evaluate(weights)
            macro[b, i] = np.r_[result['macro'], result['subset_accuracy']]
            per_class[b, i] = result['per_class']
        if (b + 1) % 250 == 0: print('SPH_SECONDARY_PATIENT_BOOTSTRAP', b + 1, 2000, flush=True)
    bootstrap_seconds = time.monotonic() - started
    np.savez_compressed(output / 'secondary_patient_bootstrap_draws.npz', macro=macro, per_class=per_class)
    arm_results = {}
    for i, name in enumerate(ARMS):
        arm_results[name] = {'macro': dict(zip(METRICS, point[i]['macro'].tolist())),
              'subset_accuracy': point[i]['subset_accuracy'],
              'per_class': {c: dict(zip(METRICS, point[i]['per_class'][j].tolist())) for j, c in enumerate(CLASS)},
              'secondary_macro_interval': intervals(macro[:, i]),
              'secondary_per_class_interval': intervals(per_class[:, i])}
    contrasts = {}
    for a, b, name in CONTRASTS:
        contrasts[name] = {'macro_point_difference': dict(zip(METRICS, (point[a]['macro'] - point[b]['macro']).tolist())),
                          'subset_accuracy_difference': point[a]['subset_accuracy'] - point[b]['subset_accuracy'],
                          'secondary_macro_interval': intervals(macro[:, a] - macro[:, b]),
                          'secondary_per_class_interval': intervals(per_class[:, a] - per_class[:, b])}
    report = {'status': 'Completed saved-prediction secondary component analysis; no model selection',
              'plan_sha256': PLAN_SHA, 'analysis_source_sha256': sha(__file__),
              'statistics_source_sha256': sha(HERE / 'sph_component_statistics_v1.py'),
              'original_assessment_sha256': sha(ORIGINAL / 'external_assessment.json'),
              'inference_lock_sha256': original['inference_lock_sha256'],
              'records': 21183, 'patients': 20401, 'classes': CLASS,
              'macro_interval_column_order': list(METRICS) + ['subset_accuracy'],
              'per_class_interval_column_order': list(METRICS), 'arms': arm_results,
              'paired_contrasts': contrasts, 'original_primary_intervals_seed42810': original['bootstrap_ci95'],
              'bootstrap_seed': 20261026, 'bootstrap_draws': 2000, 'bootstrap_seconds': bootstrap_seconds,
              'bootstrap_draws_sha256': sha(output / 'secondary_patient_bootstrap_draws.npz'),
              'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
              'limitations': ['Post hoc component analyses and intervals are descriptive, not multiplicity/selection adjusted.',
                             'No independent training-seed variance is estimated for JEPA+ResNet.',
                             'Research mapping is not expert-certified clinical equivalence.',
                             'Incomplete accessible overlap screening does not prove complete pretraining nonoverlap.',
                             'No new training, inference, threshold tuning or published-benchmark victory.']}
    save(output / 'component_assessment.json', report)
    print('SPH_COMPONENT_ANALYSIS_COMPLETED', json.dumps({n: a['macro'] for n, a in arm_results.items()}), flush=True)


if __name__ == '__main__':
    main()
