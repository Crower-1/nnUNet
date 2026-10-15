"""Opt-in geometry supervision with checkpoint-contained configuration and existing fine-tuning modes."""
import json
import time
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.variants.nnUNetTrainerFinetune import nnUNetTrainerFinetune
from nnunetv2.utilities.label_geometry import resolve_geometry, geometry_options
from nnunetv2.utilities.particle_config import particle_active, resolved_class_weights
from nnunetv2.utilities.geometry_checkpoint import geometry_snapshot, restore_geometry


class nnUNetTrainerFinetuneGeometry(nnUNetTrainerFinetune):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda'), *,
                 finetune_mode='scratch', encoder_weights=None, decoder_weights=None, head_weights=None,
                 geometry_state=None):
        super().__init__(plans, configuration, fold, dataset_json, device,
                         finetune_mode=finetune_mode, encoder_weights=encoder_weights,
                         decoder_weights=decoder_weights, head_weights=head_weights)
        metadata = self.configuration_manager.configuration.get('label_geometry')
        has_particles = any(t['geometry_type'] == 'particle' for t in (metadata or {}).get('targets', []))
        self.geometry_options = geometry_options(self.configuration_manager.configuration.get('geometry_loss', {}), has_particles)
        self.geometry_legacy_particle_disabled = False
        self.geometry_metadata = None
        if self.geometry_options['enabled']:
            if self.label_manager.has_regions:
                raise ValueError('Geometry v1 supports integer multiclass labels only')
            if len(self.configuration_manager.patch_size) != 3:
                raise ValueError('Geometry v1 requires 3D patches')
            metadata = self.configuration_manager.configuration.get('label_geometry')
            if metadata is None:
                raise ValueError('Embed label_geometry in the plans configuration before training')
            self.geometry_metadata = resolve_geometry(metadata, dataset_json['labels'],
                self.configuration_manager.network_arch_init_kwargs.get('class_names'))
        if self.geometry_metadata is not None:
            self.geometry_options['particle']['class_weights'] = resolved_class_weights(
                self.geometry_options['particle'], self.geometry_metadata['targets'])
        if geometry_state is not None:
            self._restore_geometry(geometry_state)
        # Capture resolved channels and defaults in the existing checkpoint init_args.
        self.my_init_kwargs['geometry_state'] = self._geometry_state()

    def _geometry_state(self):
        return geometry_snapshot(self.geometry_options, self.geometry_metadata,
                                 getattr(self, 'geometry_legacy_particle_disabled', False))

    def _particle_active(self):
        return particle_active(self.geometry_options, (self.geometry_metadata or {}).get('targets', []))

    def _restore_geometry(self, saved):
        # Labels are only needed for migration; exact new snapshots remain strict.
        legacy = saved is not None and ('particle' not in saved['options'] or saved.get('legacy_particle_disabled'))
        state = restore_geometry(saved, self._geometry_state(), self.dataset_json['labels'] if legacy else {},
            self.configuration_manager.network_arch_init_kwargs.get('class_names') if legacy else None)
        self.geometry_options, self.geometry_metadata = state['options'], state['metadata']
        self.geometry_legacy_particle_disabled = state.get('legacy_particle_disabled', False)

    def _build_loss(self):
        base = super()._build_loss()
        if not self.geometry_options['enabled']:
            return base
        from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper
        return GeometryLossWrapper(base, self.geometry_metadata, self.geometry_options, self.label_manager.ignore_label)

    def initialize(self):
        if self._particle_active():
            from nnunetv2.training.data_augmentation.custom_transforms.particle_targets import check_backend
            check_backend(self.geometry_options['particle']['backend'], self.device)
        super().initialize()
        self.print_to_log_file('Geometry supervision:', self._geometry_state())
        if self.global_rank == 0:
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            Path(self.output_folder, 'geometry_resolved.json').write_text(
                json.dumps(self._geometry_state(), indent=2) + '\n')

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        saved = checkpoint.get('init_args', {}).get('geometry_state')
        if saved is None and checkpoint.get('trainer_name') in ('nnUNetTrainer', 'nnUNetTrainerFinetune'):
            # These trainers predate geometry supervision and used the base loss.
            saved = {'options': {'enabled': False}, 'metadata': None}
        self._restore_geometry(saved)
        if getattr(self, 'was_initialized', False):
            self.loss = self._build_loss()
        if self.geometry_legacy_particle_disabled:
            self.print_to_log_file('Legacy objective restored: particle disabled. Use a new run to enable CC-DiceCE.')
        super().load_checkpoint(checkpoint)
        self.my_init_kwargs['geometry_state'] = self._geometry_state()
        if self.global_rank == 0:
            Path(self.output_folder, 'geometry_resolved.json').write_text(
                json.dumps(self._geometry_state(), indent=2) + '\n')

    def get_training_transforms(self, *args, **kwargs):
        transforms = super().get_training_transforms(*args, **kwargs)
        if self._particle_active():
            from nnunetv2.training.data_augmentation.custom_transforms.particle_validity import preserve_particle_validity
            transforms = preserve_particle_validity(transforms)
        return transforms

    def get_validation_transforms(self, *args, **kwargs):
        transforms = super().get_validation_transforms(*args, **kwargs)
        if self._particle_active():
            from nnunetv2.training.data_augmentation.custom_transforms.particle_validity import preserve_particle_validity
            transforms = preserve_particle_validity(transforms)
        return transforms

    def get_dataloaders(self):
        loaders = super().get_dataloaders()
        if self._particle_active():
            from nnunetv2.training.logging.geometry_diagnostics import TimedBatches
            return tuple(TimedBatches(loader) for loader in loaders)
        return loaders

    def _compute_loss(self, output, target, batch):
        if not self._particle_active():
            return super()._compute_loss(output, target, batch)
        return self.loss(output, target, particle_valid=batch['particle_valid'].to(self.device, non_blocking=True))

    def _step_with_diagnostics(self, batch, phase, step):
        if not self._particle_active():
            return step(batch)
        started = time.perf_counter()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        result = step(batch)
        from nnunetv2.training.logging.geometry_diagnostics import write_geometry_step
        write_geometry_step(self.output_folder, self.global_rank, self.current_epoch, phase,
            batch.get('keys', []), {**self.loss.last_diagnostics, 'config_hash': self._geometry_state()['config_hash']}, time.perf_counter() - started,
            batch.get('data_wait_seconds'), self.device, self.geometry_options['particle']['backend'])
        return result

    def train_step(self, batch):
        return self._step_with_diagnostics(batch, 'train', super().train_step)

    def validation_step(self, batch):
        return self._step_with_diagnostics(batch, 'validation', super().validation_step)

    def on_train_end(self):
        # Base cleanup checks the concrete augmenter type before stopping workers.
        from nnunetv2.training.logging.geometry_diagnostics import TimedBatches
        for name in ('dataloader_train', 'dataloader_val'):
            loader = getattr(self, name, None)
            if isinstance(loader, TimedBatches):
                setattr(self, name, loader.source)
        super().on_train_end()
