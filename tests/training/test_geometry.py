"""Synthetic objective contracts; real-data quality is validated separately."""
import unittest
import torch
from nnunetv2.utilities.label_geometry import resolve_geometry, geometry_options
from nnunetv2.dataset_conversion.import_synapseseg_cfg import import_cfg
from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.labels = {'background': 0, 'actin': 1, 'membrane': 2, 'ignore': 3}
        self.metadata = {'schema_version': 1, 'targets': [
            {'category': 'actin', 'role': 'body', 'label_names': ['actin'], 'geometry_type': 'filament'},
            {'category': 'membrane', 'role': 'primary', 'label_names': ['membrane'], 'geometry_type': 'membrane'}]}
        self.resolved = resolve_geometry(self.metadata, self.labels)

    def test_mapping_and_import(self):
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            resolve_geometry(self.metadata, self.labels, ['background', 'membrane', 'actin'])
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            resolve_geometry(self.metadata, {'background': 0})
        _, resolved, report = import_cfg([
            {'category': 'actin', 'filament': True}, {'category': 'mt', 'filament': True},
            {'category': 'vesicle', 'with_membrane': True}],
            {'labels': {'background': 0, 'actin': 1, 'mt': 2, 'vesicle': 3, 'vesicle_memb': 4}})
        self.assertEqual([t['geometry_type'] for t in resolved['targets']],
                         ['filament', 'default', 'default', 'membrane'])

    def test_disabled_exact_loss_and_gradient(self):
        x = torch.randn(1, 3, 5, 5, 5, requires_grad=True)
        y = torch.zeros(1, 1, 5, 5, 5, dtype=torch.long)
        def base(out, target):
            return out[0].square().mean() + out[1].mean()
        outputs = [x, x[..., ::2, ::2, ::2]]
        expected = base(outputs, [y, y])
        actual = GeometryLossWrapper(base, self.resolved, geometry_options({}))(outputs, [y, y])
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(torch.autograd.grad(expected, x, retain_graph=True)[0],
                                    torch.autograd.grad(actual, x)[0]))

    def test_highest_resolution_ignore_and_softmax(self):
        x = torch.randn(1, 3, 9, 9, 9, requires_grad=True)
        low = torch.randn(1, 3, 3, 3, 3, requires_grad=True)
        y = torch.zeros(1, 1, 9, 9, 9, dtype=torch.long)
        y[:, :, 4, 4, 2:7] = 1
        opts = geometry_options({'enabled': True, 'surface_weight': 0, 'skeleton_dilation': 0})
        loss = GeometryLossWrapper(lambda o, t: o[0].sum()*0 + o[1].sum()*0,
                                   self.resolved, opts, 3)([x, low], [y, y])
        expected = (1 - x.float().softmax(1)[:, 1, 4, 4, 2:7].mean()) * .25
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
        self.assertEqual(low.grad.abs().sum().item(), 0)
        y.fill_(3)
        ignored = GeometryLossWrapper(lambda o, t: o.sum()*0, self.resolved, opts, 3)(x, y)
        self.assertEqual(ignored.item(), 0)

    def test_pinned_membrain_values_and_gradients(self):
        import importlib.util
        from pathlib import Path
        from types import SimpleNamespace
        from nnunetv2.training.loss.geometry_losses import membrane_target, membrain_surface_loss
        path = Path(__file__).resolve().parents[2] / 'third_party/geometry/membrain_surface_reference.py'
        spec = importlib.util.spec_from_file_location('membrain_reference', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        logits = torch.randn(1, 1, 11, 11, 11, requires_grad=True)
        truth = torch.zeros_like(logits)
        truth[:, :, 5, 2:9, 2:9] = 1
        config = SimpleNamespace(membrane_kernel=15, membrane_sigma=2., membrane_amplitude=1.5,
                                 membrane_iterations=2, membrane_smooth=3.)
        actual = membrain_surface_loss(logits.sigmoid(), truth, membrane_target(truth, config),
                                      torch.ones_like(truth), config)
        expected = 1 - reference.masked_surface_dice(logits, truth, soft_skel_iterations=2).mean()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits)[0], atol=1e-6))

    def test_checkpoint_geometry_guard(self):
        from unittest.mock import patch
        from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerFinetuneGeometry import nnUNetTrainerFinetuneGeometry
        from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerFinetune import nnUNetTrainerFinetune
        trainer = object.__new__(nnUNetTrainerFinetuneGeometry)
        trainer.geometry_options = geometry_options({'enabled': True})
        trainer.geometry_metadata = self.resolved
        trainer.my_init_kwargs = {}
        trainer.global_rank = 1  # This unit test has no output folder; integration tests cover snapshot files.
        with self.assertRaisesRegex(ValueError, 'differs'):
            trainer.load_checkpoint({'init_args': {}})
        saved = {'init_args': {'geometry_state': trainer._geometry_state()}}
        with patch.object(nnUNetTrainerFinetune, 'load_checkpoint') as parent:
            trainer.load_checkpoint(saved)
            parent.assert_called_once_with(saved)
        saved['init_args']['geometry_state']['options']['surface_weight'] = 9
        with self.assertRaisesRegex(ValueError, 'differs'):
            trainer.load_checkpoint(saved)

    def test_union_probability(self):
        metadata = resolve_geometry({'schema_version': 1, 'targets': [dict(
            category='whole', role='body', label_names=['actin', 'membrane'], geometry_type='filament')]}, self.labels)
        x = torch.randn(1, 3, 7, 7, 7, requires_grad=True)
        y = torch.zeros(1, 1, 7, 7, 7, dtype=torch.long)
        y[:, :, 3, 3, 1:4] = 1
        y[:, :, 3, 3, 4:6] = 2
        options = geometry_options({'enabled': True, 'skeleton_dilation': 0})
        actual = GeometryLossWrapper(lambda o, t: o.sum()*0, metadata, options)(x, y)
        expected = .25 * (1-x.softmax(1)[:, 1:3, 3, 3, 1:6].sum(1).mean())
        self.assertTrue(torch.allclose(actual, expected))

    def test_surface_finite_gradient_and_empty(self):
        x = torch.randn(1, 3, 17, 17, 17, requires_grad=True)
        y = torch.zeros(1, 1, 17, 17, 17, dtype=torch.long)
        y[:, :, 8, 3:14, 3:14] = 2
        opts = geometry_options({'enabled': True, 'membrane_iterations': 1, 'membrane_kernel': 3})
        wrapper = GeometryLossWrapper(lambda o, t: o.sum()*0, self.resolved, opts, 3)
        loss = wrapper(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertEqual(wrapper(x, torch.zeros_like(y)).item(), 0)


if __name__ == '__main__':
    unittest.main()
