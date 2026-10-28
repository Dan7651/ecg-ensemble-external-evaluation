"""Meaningful synthetic correctness checks only; no study outcomes read."""
import unittest
import numpy as np
from scipy.special import expit, logit
from analysis_core import *


class ScientificRules(unittest.TestCase):
    def setUp(self):
        self.y = np.tile([[0, 1], [1, 0]], (12, 1))
        self.p = np.random.default_rng(7).uniform(.05, .95, self.y.shape)

    def test_weighted_metrics_match_literal_patient_repetition(self):
        patients = np.repeat(np.arange(8), 3)
        w = patient_weights(patients, np.random.default_rng(22))
        take = np.repeat(np.arange(24), w)
        np.testing.assert_allclose(metric_values(self.y, self.p, w)['per_class'], metric_values(self.y[take], self.p[take])['per_class'], atol=1e-14)
        self.assertTrue(all(len(set(w[patients == pid])) == 1 for pid in np.unique(patients)))

    def test_alignment_reorders_and_rejects_patient_or_label_drift(self):
        ids = np.arange(24); patients = ids // 2; rev = ids[::-1]
        out = aligned_scores(ids, patients, self.y, rev, patients[::-1], self.y[::-1], self.p[::-1])
        np.testing.assert_array_equal(out, self.p)
        with self.assertRaises(ValueError):
            aligned_scores(ids, patients, self.y, rev, patients, self.y[::-1], self.p[::-1])
        with self.assertRaises(ValueError):
            aligned_scores(ids, patients, self.y, rev, patients[::-1], 1 - self.y[::-1], self.p[::-1])

    def test_all_pairs_and_fixed_blocks_are_equal_budget(self):
        r = [np.clip(self.p + i * .002, 0, 1) for i in range(5)]
        j = [np.clip(1 - self.p - i * .002, 0, 1) for i in range(5)]
        rows = all_pairs(self.y, r, j)
        self.assertEqual([sum(v['family'] == f for v in rows) for f in ['RR', 'JJ', 'RJ']], [10, 10, 25])
        blocks = control_blocks(self.y, r, j)
        self.assertEqual([len(blocks[f]) for f in ['RR', 'JJ', 'RJ']], [5, 5, 5])
        for value in blocks['RJ']:
            self.assertAlmostEqual(value['gain'][0], value['ensemble'][0] - max(value['component_a'][0], value['component_b'][0]))
            self.assertAlmostEqual(value['gain'][-1], min(value['component_a'][-1], value['component_b'][-1]) - value['ensemble'][-1])

    def test_window_rule_covers_every_sample_without_padding(self):
        for n in [5000, 5001, 9999, 10000, 10001, 28000]:
            starts = full_coverage_starts(n); covered = np.zeros(n, bool)
            for start in starts:
                self.assertGreaterEqual(start, 0); self.assertLessEqual(start + 5000, n)
                covered[start:start + 5000] = True
            self.assertTrue(covered.all()); self.assertEqual(len(starts), len(set(starts)))
            self.assertEqual(starts[0], 0); self.assertEqual(starts[-1], n - 5000)

    def test_threshold_oracle_and_conservative_specificity_with_ties(self):
        y = np.array([0, 0, 0, 1, 1, 1]); p = np.array([.2, .4, .8, .4, .7, .9])
        t = oracle_f1(y, p)
        def f(th):
            d = p >= th; tp = (d & (y == 1)).sum(); fp = (d & (y == 0)).sum(); fn = (~d & (y == 1)).sum()
            return 2 * tp / (2 * tp + fp + fn)
        self.assertEqual(f(t), max(map(f, threshold_candidates(p))))
        result = fixed_specificity_threshold(y, p, .95)
        self.assertGreaterEqual(result['specificity'], .95)
        self.assertEqual(result['sensitivity'], 1 / 3)
        with self.assertRaises(ValueError): oracle_f1(np.zeros(6), p)

    def test_prior_adjusted_threshold_matches_shifted_score_decision(self):
        delta, t = prevalence_adjustment(np.array([.4, .3]), np.array([.05, .5]), np.array([.37, .43]))
        shifted = expit(logit(self.p) + delta)
        np.testing.assert_array_equal(self.p >= t, shifted >= [.37, .43])
        self.assertGreater(t[0], .37); self.assertLess(t[1], .43)

    def test_patient_crossfit_has_no_patient_overlap(self):
        patients = np.repeat(np.arange(20), [1, 2, 3, 4] * 5)
        folds = patient_folds(patients)
        for f in range(5):
            self.assertFalse(set(patients[folds == f]) & set(patients[folds != f]))
        np.testing.assert_array_equal(folds, patient_folds(patients))

    def test_unsupported_class_and_incomplete_bootstrap_are_not_hidden(self):
        result = metric_values(np.zeros_like(self.y), self.p)
        self.assertTrue(np.isnan(result['macro'][0]))
        self.assertIsNone(interval(np.zeros(1959))['interval'])
        self.assertEqual(interval(np.zeros(2000), primary=True)['percentiles'], (1.25, 98.75))

    def test_offset_calibration_solves_score_equation(self):
        y, p = self.y[:, 0], self.p[:, 0]
        out = calibration(y, p)
        self.assertLess(abs((expit(logit(p) + out['offset_slope_fixed_1']) - y).sum()), 1e-9)


if __name__ == '__main__':
    unittest.main()
