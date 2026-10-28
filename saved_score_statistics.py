"""Exact weighted saved-score statistics for patient-cluster resampling.

No model imports, fitting, inference, calibration or threshold search.
"""
import numpy as np

METRICS = ("auroc", "ap", "f1", "precision", "recall", "specificity", "brier")


class ScoreStatistics:
    def __init__(self, y, scores, thresholds=0.5):
        self.y = np.asarray(y, dtype=np.uint8)
        self.scores = np.asarray(scores)
        if self.y.ndim != 2 or self.scores.shape != self.y.shape:
            raise ValueError("Unaligned labels/scores")
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Nonbinary labels")
        if not np.isfinite(self.scores).all() or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError("Invalid probabilities")
        self.pred = self.scores >= np.asarray(thresholds)
        if self.pred.shape != self.y.shape:
            raise ValueError("Invalid threshold shape")
        self.squared_error = (self.scores.astype(np.float64) - self.y) ** 2
        self.exact = np.all(self.pred == self.y, axis=1)
        self.ranks = []
        for c in range(self.y.shape[1]):
            order = np.argsort(self.scores[:, c], kind="stable")
            sorted_scores = self.scores[order, c]
            starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
            self.ranks.append((order, starts))

    def evaluate(self, weights=None):
        n, k = self.y.shape
        w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Invalid observation weights")
        per = np.empty((k, len(METRICS)), dtype=np.float64)
        for c, (order, starts) in enumerate(self.ranks):
            y = self.y[:, c]
            p = self.pred[:, c]
            pos = np.add.reduceat(w[order] * y[order], starts)
            neg = np.add.reduceat(w[order] * (1 - y[order]), starts)
            np_, nn_ = pos.sum(), neg.sum()
            auc = np.sum(pos * (np.cumsum(neg) - neg / 2)) / (np_ * nn_) if min(np_, nn_) > 0 else np.nan
            cp = np.cumsum(pos[::-1]); cn = np.cumsum(neg[::-1])
            precision_at_rank = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=(cp + cn) > 0)
            ap = np.sum(pos[::-1] * precision_at_rank) / np_ if np_ > 0 else np.nan
            tp = w[(y == 1) & p].sum(); fp = w[(y == 0) & p].sum()
            fn = w[(y == 1) & ~p].sum(); tn = w[(y == 0) & ~p].sum()
            ratio = lambda a, b: a / b if b else 0.0
            per[c] = (auc, ap, ratio(2 * tp, 2 * tp + fp + fn), ratio(tp, tp + fp),
                      ratio(tp, tp + fn), ratio(tn, tn + fp), np.dot(w, self.squared_error[:, c]) / w.sum())
        macro = np.mean(per, axis=0)
        return {"per_class": per, "macro": macro,
                "subset_accuracy": float(np.dot(w, self.exact) / w.sum())}


def cluster_weights(patient_inverse, patient_multiplicity):
    """Replicate each selected patient's complete record cluster, not one ECG."""
    return np.asarray(patient_multiplicity)[np.asarray(patient_inverse)]


def intervals(draws):
    draws = np.asarray(draws, dtype=np.float64)
    valid = np.all(np.isfinite(draws), axis=tuple(range(1, draws.ndim)))
    if not valid.any():
        raise ValueError("No valid bootstrap draws")
    return {"valid_draws": int(valid.sum()),
            "ci95": np.percentile(draws[valid], [2.5, 97.5], axis=0).tolist()}
