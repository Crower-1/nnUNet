"""Exercise real loader, augmentation, optimizer steps, snapshots and legacy resume on CPU."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.label_geometry import geometry_options, resolve_geometry
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerFinetuneGeometry import nnUNetTrainerFinetuneGeometry


class ToyDataset:
    identifiers = ['synthetic']

    def load_case(self, identifier):
        mask = np.zeros((1, 5, 5, 5), dtype=np.int16)
        mask[:, 2, 2, 2] = 1
        return np.ones((1, 5, 5, 5), np.float32), mask, None


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(1, 4, 1)
        self.decoder = nn.ReLU()
        self.heads = nn.ModuleDict({name: nn.Conv3d(4, 1, 1) for name in ('background', 'particle', 'other')})

    def forward(self, data):
        features = self.decoder(self.encoder(data))
        return torch.cat([head(features) for head in self.heads.values()], 1)


class PipelineTests(unittest.TestCase):
    def make_trainer(self, directory):
        trainer = object.__new__(nnUNetTrainerFinetuneGeometry)
        trainer.geometry_metadata = resolve_geometry({'schema_version': 1, 'targets': [dict(
            category='particle', role='body', label_names=['particle'], geometry_type='particle')]},
            {'background': 0, 'particle': 1, 'other': 2})
        trainer.geometry_options = geometry_options({}, True)
        trainer.geometry_legacy_particle_disabled = False
        trainer.dataset_json = {'labels': {'background': 0, 'particle': 1, 'other': 2}}
        trainer.label_manager = LabelManager(trainer.dataset_json['labels'], None)
        trainer.configuration_manager = SimpleNamespace(batch_dice=False, network_arch_init_kwargs={})
        trainer.device = torch.device('cpu')
        trainer.network = ToyNetwork()
        trainer.enable_deep_supervision = False
        trainer.is_ddp = False
        trainer.global_rank = 0
        trainer._do_i_compile = lambda: False
        trainer.loss = trainer._build_loss()
        trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=.1)
        trainer.grad_scaler = None
        trainer.current_epoch = 0
        trainer.output_folder = directory
        trainer.my_init_kwargs = {'geometry_state': trainer._geometry_state()}
        trainer.logger = SimpleNamespace(get_checkpoint=lambda: {}, load_checkpoint=lambda _: None)
        trainer._best_ema = None
        trainer.inference_allowed_mirroring_axes = None
        trainer.disable_checkpointing = False
        trainer.was_initialized = True
        trainer.print_to_log_file = lambda *a, **k: None
        return trainer

    def test_loader_train_validation_save_resume(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            transforms = trainer.get_validation_transforms(None)
            loader = nnUNetDataLoader(ToyDataset(), 2, (7, 7, 7), (7, 7, 7), trainer.label_manager,
                                     oversample_foreground_percent=0, transforms=transforms)
            batch = next(loader)
            self.assertIn('particle_valid', batch)
            self.assertEqual(batch['target'].shape, batch['particle_valid'].shape)
            self.assertTrue((batch['target'] >= 0).all())
            self.assertTrue((~batch['particle_valid']).any())
            before = trainer.network.heads['particle'].weight.detach().clone()
            result = trainer.train_step(batch)
            self.assertTrue(np.isfinite(result['loss']))
            self.assertFalse(torch.equal(before, trainer.network.heads['particle'].weight))
            with torch.no_grad():
                result = trainer.validation_step(next(loader))
            self.assertTrue(np.isfinite(result['loss']))
            records = [json.loads(line) for line in Path(directory, 'geometry_steps_rank0.jsonl').read_text().splitlines()]
            self.assertEqual([r['phase'] for r in records], ['train', 'validation'])
            self.assertTrue(records[0]['particle']['particle']['cc_count'] > 0)
            path = str(Path(directory, 'checkpoint_latest.pth'))
            trainer.save_checkpoint(path)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved['init_args']['geometry_state'], trainer._geometry_state())
            trainer.load_checkpoint(path)
            # Legacy run has this same semantic target but no CC objective.
            old = saved['init_args']['geometry_state']
            old.pop('config_hash')
            old['options'].pop('particle')
            old['metadata']['targets'][0]['geometry_type'] = 'default'
            trainer.load_checkpoint(saved)
            self.assertFalse(trainer._particle_active())
            self.assertEqual(trainer.my_init_kwargs['geometry_state']['options']['particle']['enabled'], False)
            saved['init_args'].pop('geometry_state')
            saved['trainer_name'] = 'nnUNetTrainerFinetune'
            trainer.load_checkpoint(saved)
            self.assertFalse(trainer.geometry_options['enabled'])

    def test_training_augmentation_tracks_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            transform = trainer.get_training_transforms((16, 16, 16), (0., 0.), None, (0, 1, 2), False)
            for _ in range(3):
                # Actual random spatial scaling/mirroring; padded area remains separately marked.
                seg = torch.zeros(1, 20, 20, 20, dtype=torch.int16)
                seg[:, :5] = -1
                seg[:, 10, 10, 10] = 1
                result = transform(image=torch.ones_like(seg).float(), segmentation=seg)
                self.assertEqual(result['particle_valid'].shape, result['segmentation'].shape)
                self.assertTrue((result['segmentation'] >= 0).all())
                self.assertTrue(result['particle_valid'][result['segmentation'] == 1].all())

    def test_inference_reload_without_training_backend(self):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        import sys
        architecture = dict(network_class_name='nnunetv2.network_architecture.plainconv_unet_head.PlainConvUNetHead',
            arch_kwargs=dict(n_stages=2, features_per_stage=[4, 8], conv_op='torch.nn.Conv3d',
                kernel_sizes=[[3, 3, 3], [3, 3, 3]], strides=[[1, 1, 1], [2, 2, 2]],
                n_conv_per_stage=[1, 1], n_conv_per_stage_decoder=[1], nonlin='torch.nn.ReLU',
                class_names=['background', 'particle', 'other']), _kw_requires_import=['conv_op', 'nonlin'])
        plans = dict(configurations={'3d_fullres': dict(architecture=architecture, previous_stage=None,
                      geometry_loss={'enabled': True, 'particle': {'backend': 'cupy'}})})
        dataset = dict(labels={'background': 0, 'particle': 1, 'other': 2}, channel_names={'0': 'image'})
        manager = PlansManager(plans)
        config = manager.get_configuration('3d_fullres', dataset)
        net = nnUNetTrainerFinetuneGeometry.build_network_architecture(manager, config, 1, 3, False)
        net.eval()
        sample = torch.randn(1, 1, 8, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'fold_0').mkdir()
            (folder/'dataset.json').write_text(json.dumps(dataset))
            (folder/'plans.json').write_text(json.dumps(plans))
            torch.save(dict(network_weights=net.state_dict(), trainer_name='nnUNetTrainerFinetuneGeometry',
                            init_args={'configuration': '3d_fullres'}), folder/'fold_0/checkpoint_final.pth')
            # Deliberately make training geometry modules and CuPy unavailable.
            with patch.dict(sys.modules, {'cupy': None,
                'nnunetv2.training.loss.particle_losses': None,
                'nnunetv2.training.data_augmentation.custom_transforms.particle_targets': None}), \
                 patch.dict('os.environ', {'nnUNet_compile': 'false'}):
                predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
                predictor.initialize_from_trained_model_folder(str(folder), [0])
                predictor.network.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(net(sample), predictor.network(sample)))
