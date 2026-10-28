"""Bounded CPU guards: synthetic tiny models only, no ECG/model-weight input."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

import a01_matched_fits as r


class Guards(unittest.TestCase):
    def test_01_source_extraction_no_top_level_execution(self):
        source = (r.HERE / 'multiscale_waveform_dev8.py').read_text()
        nodes = r.architecture_nodes(source)
        self.assertEqual([n.name for n in nodes], ['ResidualBlock', 'OriginalResNet'])
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes))
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         '79bb48ea71a033f21dfd734c5fbc4fbb3dc0823be4c2cdd2137f47b5494bc44e')
        self.assertEqual(len(nodes[1].body), 2)
        # No constructor or actual network forward is executed locally.

    def test_02_strict_selection_and_small_unselected_improvement(self):
        hist = [dict(epoch=i+1, metrics={'macro_auroc':v})
                for i, v in enumerate([.8, .8, .80005, .8002, .8002, .80025])]
        self.assertEqual(r.selected_history(hist, .0001), (.8002, 4, 2))
        self.assertEqual(r.selected_history(hist, .000001), (.80025, 6, 0))

    def test_03_exact_tiny_optimizer_scheduler_scaler_rng_resume(self):
        for family in r.FAMILIES:
            g = torch.Generator(); r.seed_all(91, g)
            model = nn.Sequential(nn.Linear(3, 7), nn.ReLU(), nn.Dropout(.3), nn.Linear(7, 5))
            opt = r.optimizer(model, family); sc = r.scaler_new(torch.device('cpu'), family)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
            x, y = torch.randn(8, 3), torch.randint(0, 2, (8, 5)).float()
            r.update(model, opt, sc, x, y, family, torch.device('cpu')); sched.step()
            snapshot = r.cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                  scaler=sc.state_dict(), scheduler=sched.state_dict(), rng=r.rng_state(g)))
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/'synthetic.pth'; r.atomic(path, snapshot, 'torch')
                snapshot = torch.load(path, map_location='cpu', weights_only=False)
            x2, y2 = torch.randn(8, 3), torch.randint(0, 2, (8, 5)).float()
            r.update(model, opt, sc, x2, y2, family, torch.device('cpu')); sched.step()
            expected = r.cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                 scaler=sc.state_dict(), scheduler=sched.state_dict(), rng=r.rng_state(g)))
            resumed = nn.Sequential(nn.Linear(3, 7), nn.ReLU(), nn.Dropout(.3), nn.Linear(7, 5))
            resumed.load_state_dict(snapshot['model']); ro = r.optimizer(resumed, family)
            ro.load_state_dict(snapshot['optimizer']); rs = r.scaler_new(torch.device('cpu'), family)
            rs.load_state_dict(snapshot['scaler'])
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(ro, T_max=20, eta_min=1e-5)
            sched2.load_state_dict(snapshot['scheduler']); r.restore_rng(snapshot['rng'], g)
            xx, yy = torch.randn(8, 3), torch.randint(0, 2, (8, 5)).float()
            self.assertTrue(torch.equal(x2, xx) and torch.equal(y2, yy))
            r.update(resumed, ro, rs, xx, yy, family, torch.device('cpu')); sched2.step()
            actual = r.cpu_tree(dict(model=resumed.state_dict(), optimizer=ro.state_dict(),
                               scaler=rs.state_dict(), scheduler=sched2.state_dict(), rng=r.rng_state(g)))
            self.assertTrue(r.equal(expected, actual))

    def test_04_nonfinite_gradient_fails_before_update(self):
        model = nn.Linear(3, 5); original = r.cpu_tree(model.state_dict())
        opt = r.optimizer(model, 'jepa_head'); sc = r.scaler_new(torch.device('cpu'), 'jepa_head')
        hook = model.weight.register_hook(lambda gradient: gradient * float('nan'))
        with self.assertRaises(FloatingPointError):
            r.update(model, opt, sc, torch.ones(4, 3), torch.zeros(4, 5), 'jepa_head', torch.device('cpu'))
        hook.remove(); self.assertTrue(r.equal(original, model.state_dict()))

    def test_05_checkpoint_generation_recovery_and_protocol_guard(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for epoch in (1, 2):
                history = [dict(epoch=i, metrics={'macro_auroc':.8+i*.001}) for i in range(1, epoch+1)]
                s = dict(protocol_sha256='fixed', family='resnet', epoch=epoch, history=history,
                         best_auc=.8+epoch*.001, selected_epoch=epoch, stale=0, updates=epoch*234)
                path = folder/f'last_{epoch%2}.pth'; r.atomic(path, s, 'torch')
                r.atomic(path.with_suffix('.receipt.json'), dict(epoch=epoch, sha256=r.sha(path), protocol_sha256='fixed'))
            self.assertEqual(r.load_commit(folder, 'fixed', 'resnet')['epoch'], 2)
            with self.assertRaises(AssertionError): r.load_commit(folder, 'other', 'resnet')
            (folder/'last_0.pth').write_bytes(b'incomplete-generation')
            self.assertEqual(r.load_commit(folder, 'fixed', 'resnet')['epoch'], 1)
            (folder/'last_1.pth').write_bytes(b'incomplete-generation')
            with self.assertRaises(RuntimeError): r.load_commit(folder, 'fixed', 'resnet')

    def test_06_cache_copy_missing_only_and_preserve_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            source, target = Path(td)/'source', Path(td)/'target'
            source.write_bytes(b'cached-only'); digest = r.sha(source)
            r.copy_cached(source, target, digest); before = target.stat().st_mtime_ns
            source.unlink(); r.copy_cached(source, target, digest)
            self.assertEqual(target.stat().st_mtime_ns, before)
            target.write_bytes(b'damaged'); source.write_bytes(b'cached-only')
            r.copy_cached(source, target, digest)
            self.assertEqual(r.sha(target), digest)
            self.assertEqual(len(list(Path(td).glob('target.unverified_*'))), 1)

    def test_07_saved_feature_normalization_exact_transform(self):
        random = np.random.default_rng(45)
        fitting = random.normal(size=(17, 9)).astype(np.float32)
        fitting[:, 0] = 3.
        development = random.normal(size=(5, 9)).astype(np.float32)
        scaler = StandardScaler().fit(fitting)
        restored = StandardScaler(); restored.mean_ = scaler.mean_.copy()
        restored.scale_, restored.var_ = scaler.scale_.copy(), scaler.var_.copy()
        restored.n_features_in_, restored.n_samples_seen_ = 9, len(fitting)
        self.assertTrue(np.array_equal(scaler.transform(development), restored.transform(development)))
        self.assertEqual(restored.scale_[0], 1.)

    def test_08_completed_endpoint_integrity_and_idempotent_json(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            protocol = dict(global_protocol_sha256='fixed', family='resnet', seed=101,
                            source_sha256=r.sha(r.__file__), selection_gap=.0001,
                            max_epochs=1, patience=6, batch=64)
            r.atomic(folder/'protocol.json', protocol, immutable=True)
            before = (folder/'protocol.json').stat().st_mtime_ns
            r.atomic(folder/'protocol.json', protocol, immutable=True)
            self.assertEqual((folder/'protocol.json').stat().st_mtime_ns, before)
            with self.assertRaises(RuntimeError):
                r.atomic(folder/'protocol.json', dict(protocol, seed=202), immutable=True)
            (folder/'best_model.pth').write_bytes(b'synthetic-not-a-real-checkpoint')
            (folder/'development_predictions.npz').write_bytes(b'synthetic-not-real-scores')
            r.atomic(folder/'history.json', [dict(epoch=1, metrics={'macro_auroc':.8}, updates=234)])
            endpoint = dict(protocol_sha256=r.sha(folder/'protocol.json'), family='resnet', seed=101,
                source_sha256=r.sha(r.__file__), exact_fresh_saved_best_replay=True,
                no_external_inference=True, no_fold9_or_10_prediction=True,
                checkpoint_sha256=r.sha(folder/'best_model.pth'),
                prediction_sha256=r.sha(folder/'development_predictions.npz'),
                completed_epochs=1, selected_epoch=1, updates=234, development_metrics={'macro_auroc':.8})
            r.atomic(folder/'endpoint.json', endpoint)
            self.assertEqual(r.verify_endpoint(folder, 'fixed', 'resnet', 101), endpoint)
            (folder/'development_predictions.npz').write_bytes(b'changed')
            with self.assertRaises(AssertionError): r.verify_endpoint(folder, 'fixed', 'resnet', 101)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unittest.main(verbosity=2)
