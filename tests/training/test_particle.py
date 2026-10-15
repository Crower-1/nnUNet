"""Mathematical, pipeline and compatibility contracts for particle CC-DiceCE."""
import ast
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from nnunetv2.dataset_conversion.import_synapseseg_cfg import import_cfg
from nnunetv2.utilities.label_geometry import resolve_geometry, geometry_options
from nnunetv2.utilities.particle_config import particle_options
from nnunetv2.utilities.geometry_checkpoint import geometry_snapshot, restore_geometry
from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper
from nnunetv2.training.loss.particle_losses import one_vs_rest_logit, cc_terms, particle_class_loss
from nnunetv2.training.data_augmentation.custom_transforms.particle_targets import particle_partition


def metadata():
    return resolve_geometry({'schema_version': 1, 'targets': [dict(category='new_particle', role='body',
        label_names=['new_particle'], geometry_type='particle')]}, {'background': 0, 'new_particle': 1, 'other': 2})


def upstream_reduction():
    path = Path(__file__).resolve().parents[2] / 'third_party/particle/instance_losses_reference.py'
    tree = ast.parse(path.read_text())
    names = {'_scatter_segment_sum', '_combine_vectorized_loss_terms', '_whole_volume_fallback_loss',
             '_vectorized_cc_reduction'}
    # Execute ONLY pinned pure functions, avoiding upstream compiler and CuPy side effects.
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, vmap=torch.func.vmap, Optional=Optional, VectorizedLossMode=str)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['_vectorized_cc_reduction']


class ParticleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_all_cfg_particle_names_and_conflict(self):
        names = ['ribo', 'proteasome', 'granule', 'anything_new']
        cfg = [dict(category=n, particle=True) for n in names + ['unbound']]
        labels = {'background': 0, **{n: i+1 for i, n in enumerate(names)}}
        raw, resolved, report = import_cfg(cfg, {'labels': labels})
        self.assertEqual(report['particle_targets'], names)
        self.assertEqual(report['unbound_categories'], ['unbound'])
        self.assertTrue(all(t['geometry_type'] == 'particle' for t in resolved['targets']))
        self.assertTrue(geometry_options({}, has_particle_targets=True)['enabled'])
        cfg[0]['filament'] = True
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            import_cfg(cfg, {'labels': labels})
        override = dict(category='ribo', role='body', label_names=['ribo'], geometry_type='particle',
                        override_reason='Explicit particle body role')
        self.assertEqual(import_cfg(cfg, {'labels': labels}, [override])[2]['overrides'], [override])

    def test_legacy_snapshot_stays_disabled_and_new_resume_strict(self):
        labels = {'background': 0, 'new_particle': 1, 'other': 2}
        old_meta = metadata()
        old_meta['targets'][0]['geometry_type'] = 'default'
        old_options = geometry_options({'enabled': True})
        old_options.pop('particle')
        old = dict(options=old_options, metadata=old_meta)
        current = geometry_snapshot(geometry_options({}, True), metadata())
        migrated = restore_geometry(old, current, labels)
        self.assertFalse(migrated['options']['particle']['enabled'])
        self.assertEqual(migrated['metadata'], old_meta)
        self.assertEqual(restore_geometry(migrated, current, labels), migrated)
        self.assertEqual(restore_geometry(current, current, labels), current)
        changed = copy.deepcopy(current)
        changed['options']['particle']['weight'] = .5
        with self.assertRaisesRegex(ValueError, 'differs'):
            restore_geometry(current, changed, labels)

    def test_disabled_and_zero_weight_exact_legacy_objective(self):
        x = torch.randn(1, 3, 9, 9, 9, requires_grad=True)
        y = torch.ones(1, 1, 9, 9, 9, dtype=torch.long)
        for particle in ({'enabled': False}, {'weight': 0}):
            options = geometry_options({'enabled': True, 'particle': particle})
            wrapper = GeometryLossWrapper(lambda o, t: o.square().mean(), metadata(), options)
            expected = x.square().mean()
            with patch('nnunetv2.training.loss.particle_losses.particle_partition', side_effect=AssertionError('CC called')):
                actual = wrapper(x, y)
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(torch.autograd.grad(expected, x, retain_graph=True)[0],
                                        torch.autograd.grad(actual, x)[0]))

    def test_ovr_probability_and_all_channel_gradients(self):
        x = torch.randn(2, 5, 3, 3, 3, requires_grad=True)
        p = one_vs_rest_logit(x, 2).sigmoid()
        expected = x.softmax(1)[:, 2]
        self.assertTrue(torch.allclose(p, expected, atol=1e-7))
        self.assertTrue(torch.allclose(torch.autograd.grad(p.sum(), x, retain_graph=True)[0],
                                       torch.autograd.grad(expected.sum(), x)[0], atol=1e-6))

    def test_scatter_naive_and_upstream_values_gradients(self):
        y = torch.zeros(7, 7, 7, dtype=torch.bool)
        y[1, 1, 1] = True
        y[4:6, 4:6, 4:6] = True
        partition = particle_partition(y, torch.ones_like(y))
        self.assertEqual(partition.count, 2)
        for profile in ('dice', 'dice_ce'):
            x = torch.randn(1, 2, 7, 7, 7, requires_grad=True)
            p = x.softmax(1)[:, 1]
            ce = F.cross_entropy(x, y[None].long(), reduction='none')
            dice, cross = cc_terms(p[0], ce[0], y, partition.regions, 2)
            actual = dice + cross if profile == 'dice_ce' else dice
            naive = []
            for k in (1, 2):
                mask = partition.regions == k
                value = 1 - 2*(p[0][mask]*y[mask]).sum()/(p[0][mask].sum()+y[mask].sum())
                naive.append(value + ce[0][mask].mean() if profile == 'dice_ce' else value)
            naive = torch.stack(naive).mean()
            upstream = upstream_reduction()(p, y[None].float(), ce, partition.regions[None], profile, 2)
            for expected in (naive, upstream):
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
                self.assertTrue(torch.allclose(torch.autograd.grad(actual, x, retain_graph=True)[0],
                    torch.autograd.grad(expected, x, retain_graph=True)[0], atol=1e-6))

    def test_empty_full_singleton_diagonal_and_validity(self):
        valid = torch.ones(7, 7, 7, dtype=torch.bool)
        with patch('scipy.ndimage.label', side_effect=AssertionError('empty CCL')):
            partition = particle_partition(torch.zeros_like(valid), valid)
        self.assertEqual(partition.count, 0)
        y = torch.zeros_like(valid)
        y[2, 2, 2] = y[3, 3, 3] = True
        self.assertEqual(particle_partition(y, valid).count, 1)
        y[5, 5, 5] = True
        self.assertEqual(particle_partition(y, valid).count, 2)
        full = particle_partition(valid, valid)
        self.assertEqual(full.count, 1)
        self.assertTrue((full.regions == 1).all())
        valid[:2] = False
        partition = particle_partition(y, valid)
        self.assertTrue((partition.regions[:2] == 0).all())
        self.assertTrue((partition.regions[valid] > 0).all())

    def test_padding_ignore_and_extreme_amp(self):
        entry = metadata()['targets'][0]
        y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.long)
        y[:, :, 3, 3, 3] = 1
        y[:, :, 0] = 99
        valid = torch.ones_like(y, dtype=torch.bool)
        valid[:, :, :2] = False
        x = (torch.randn(1, 3, 7, 7, 7)*1000).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            values, _ = particle_class_loss(x, y, valid, entry, particle_options(), 99)
        self.assertEqual(values.dtype, torch.float32)
        grad = torch.autograd.grad(values.sum(), x)[0]
        self.assertTrue(torch.isfinite(values).all() and torch.isfinite(grad).all())
        self.assertEqual(grad[:, :, :2].abs().sum().item(), 0)
        altered = x.detach().clone()
        altered[:, :, :2] += 9999
        other, _ = particle_class_loss(altered, y, valid, entry, particle_options(), 99)
        self.assertTrue(torch.equal(values, other))
        empty, _ = particle_class_loss(x, y, torch.zeros_like(valid), entry, particle_options(), 99)
        self.assertEqual(empty.item(), 0)
        self.assertEqual(torch.autograd.grad(empty.sum(), x)[0].abs().sum().item(), 0)

    def test_whole_valid_view_far_false_positive_and_singleton(self):
        y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.long)
        y[:, :, 1, 1, 1] = 1
        valid = torch.ones_like(y, dtype=torch.bool)
        x = torch.zeros(1, 3, 7, 7, 7, requires_grad=True)
        entry = metadata()['targets'][0]
        actual, _ = particle_class_loss(x, y, valid, entry, particle_options())
        probability = x.softmax(1)[:, 1]
        truth = (y[:, 0] == 1).float()
        expected = 1-2*(probability*truth).sum()/(probability.sum()+truth.sum())
        expected += F.binary_cross_entropy_with_logits(one_vs_rest_logit(x, 1), truth)
        self.assertTrue(torch.allclose(actual[0], expected))
        grad = torch.autograd.grad(actual.sum(), x)[0]
        self.assertGreater(grad[0, 1, -1, -1, -1].item(), 0)

    def test_touching_different_classes_are_partitioned_separately(self):
        y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.long)
        y[:, :, 2, 2, 2] = 1
        y[:, :, 3, 3, 3] = 2
        valid = torch.ones_like(y, dtype=torch.bool)
        x = torch.randn(1, 3, 7, 7, 7)
        for label in (1, 2):
            entry = dict(label_ids=[label], channels=[label])
            _, diagnostics = particle_class_loss(x, y, valid, entry, particle_options())
            self.assertEqual(diagnostics[0]['cc_count'], 1)
            self.assertEqual(diagnostics[0]['component_sizes'], [1])

    def test_ignore_cut_is_reported(self):
        y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.long)
        y[:, :, 3, 3, 1:6] = 1
        y[:, :, 3] = 99
        y[:, :, 3, 3, 1:3] = 1
        x = torch.randn(1, 3, 7, 7, 7)
        _, diagnostics = particle_class_loss(x, y, y != 99, metadata()['targets'][0], particle_options(), 99)
        self.assertGreater(diagnostics[0]['unknown_touching_cc_count'], 0)

    def test_highest_resolution_and_fixed_class_denominator(self):
        entry = metadata()['targets'][0]
        second = dict(category='other', role='body', label_names=['other'], label_ids=[2], channels=[2], geometry_type='particle')
        options = geometry_options({'enabled': True})
        x = torch.randn(2, 3, 7, 7, 7, requires_grad=True)
        low = torch.randn(2, 3, 3, 3, 3, requires_grad=True)
        y = torch.zeros(2, 1, 7, 7, 7, dtype=torch.long)
        y[0, :, 3, 3, 3] = 1
        valid = torch.ones_like(y, dtype=torch.bool)
        base = lambda o, t: o[0].sum()*0 + o[1].sum()*0
        wrapper = GeometryLossWrapper(base, {'targets': [entry, second]}, options)
        loss = wrapper([x, low], [y, y], particle_valid=valid)
        values, _ = particle_class_loss(x, y, valid, entry, options['particle'])
        self.assertTrue(torch.allclose(loss, .25*values.mean()/2))
        loss.backward()
        self.assertEqual(low.grad.abs().sum().item(), 0)

    def test_post_augmentation_padding_is_separate_from_semantic_target(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        from nnunetv2.training.data_augmentation.custom_transforms.particle_validity import preserve_particle_validity
        transform = preserve_particle_validity(nnUNetTrainer.get_validation_transforms([[1, 1, 1], [.5, .5, .5]]))
        seg = torch.zeros(1, 8, 8, 8, dtype=torch.int16)
        seg[:, :2] = -1
        seg[:, 5, 5, 5] = 1
        result = transform(image=torch.zeros_like(seg).float(), segmentation=seg)
        self.assertEqual(result['segmentation'][0].min(), 0)
        self.assertFalse(result['particle_valid'][:, :2].any())
        self.assertTrue(result['particle_valid'][:, 2:].all())
        self.assertEqual(result['particle_valid'].shape, result['segmentation'][0].shape)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('cupy'), 'CUDA/CuPy unavailable')
    def test_cupy_parity_without_distance_ties(self):
        y = torch.zeros(6, 5, 5, dtype=torch.bool, device='cuda')
        y[1, 2, 2] = y[4, 2, 2] = True
        valid = torch.ones_like(y)
        cpu = particle_partition(y, valid, 'scipy_reference')
        gpu = particle_partition(y, valid, 'cupy')
        self.assertTrue(torch.equal(cpu.regions, gpu.regions))
        x = torch.randn(1, 3, *y.shape, device='cuda', requires_grad=True)
        labels = y[None, None].long()
        a, _ = particle_class_loss(x, labels, valid[None, None], metadata()['targets'][0], particle_options())
        b, _ = particle_class_loss(x, labels, valid[None, None], metadata()['targets'][0], particle_options({'backend': 'cupy'}))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.allclose(torch.autograd.grad(a.sum(), x, retain_graph=True)[0],
                                       torch.autograd.grad(b.sum(), x)[0], atol=1e-6))


if __name__ == '__main__':
    unittest.main()
