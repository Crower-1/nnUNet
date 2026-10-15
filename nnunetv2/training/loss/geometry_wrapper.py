"""Keep the complete base DS objective and add FP32 highest-resolution geometry terms."""
from types import SimpleNamespace
import torch
from torch import nn
from nnunetv2.training.loss.geometry_losses import (
    membrane_target, reliable_region, membrain_surface_loss, skeleton_recall)
from nnunetv2.training.data_augmentation.custom_transforms.geometry_targets import skeleton_support
from nnunetv2.utilities.particle_config import particle_active, resolved_class_weights


class GeometryLossWrapper(nn.Module):
    def __init__(self, base_loss, metadata, options, ignore_label=None):
        super().__init__()
        self.base_loss = base_loss
        self.targets = metadata['targets']
        self.options = options
        self.ignore_label = ignore_label
        self.particle_enabled = particle_active(options, self.targets)
        self.particle_weights = resolved_class_weights(options['particle'], self.targets) if self.particle_enabled else {}
        self.last_diagnostics = {}

    def forward(self, output, target, particle_valid=None):
        base = self.base_loss(output, target)
        if not self.options['enabled']:
            return base
        logits = output[0] if isinstance(output, (list, tuple)) else output
        labels = target[0] if isinstance(target, (list, tuple)) else target
        if logits.ndim != 5:
            raise ValueError('Geometry v1 requires 3D patches')
        config = SimpleNamespace(**self.options)
        self.last_diagnostics = {'base_loss': float(base.detach()), 'particle': {}}
        with torch.autocast(logits.device.type, enabled=False):
            probabilities = logits.float().softmax(1)
            valid = labels >= 0
            if self.ignore_label is not None:
                valid = valid & (labels != self.ignore_label)
            groups = {'membrane': [], 'filament': []}
            for entry in self.targets:
                kind = entry['geometry_type']
                weight = config.surface_weight if kind == 'membrane' else config.skeleton_weight
                if kind in ('default', 'particle') or weight == 0:
                    continue
                truth = torch.zeros_like(labels, dtype=torch.bool)
                for label in entry['label_ids']:
                    truth |= labels == label
                truth = (truth & valid).float()
                probability = probabilities[:, entry['channels']].sum(1, keepdim=True)
                if kind == 'filament':
                    support = skeleton_support(truth, valid, config.skeleton_dilation)
                    value = skeleton_recall(probability, support)
                else:
                    region = reliable_region(valid, config.membrane_kernel // 2 + config.membrane_iterations + 2)
                    middle = membrane_target(truth, config)
                    value = membrain_surface_loss(probability, truth, middle, region, config)
                groups[kind].append(value)
            # Fixed declared target and batch denominators: empty supports contribute zero.
            # No collectives here; DDP averages rank-local losses with equal rank weight.
            for kind, weight in (('membrane', config.surface_weight), ('filament', config.skeleton_weight)):
                if groups[kind]:
                    raw = torch.stack(groups[kind]).mean()
                    self.last_diagnostics[kind + '_raw'] = float(raw.detach())
                    base = base + weight * raw
            if self.particle_enabled:
                if particle_valid is None or particle_valid.shape != labels.shape:
                    raise ValueError('Active particle loss requires a separate high-resolution particle_valid mask')
                from nnunetv2.training.loss.particle_losses import particle_class_loss, ddp_sample_mean
                particle_valid = particle_valid.to(device=labels.device, dtype=torch.bool) & valid
                weighted_sum = logits.float().sum() * 0
                denominator = sum(self.particle_weights.values())
                for entry in self.targets:
                    if entry['geometry_type'] != 'particle':
                        continue
                    name = entry['label_names'][0]
                    values, diagnostics = particle_class_loss(logits, labels, particle_valid, entry,
                                                              config.particle, self.ignore_label)
                    coefficient = config.particle['weight'] * self.particle_weights[name] / denominator
                    weighted_sum = weighted_sum + values.sum() * coefficient
                    self.last_diagnostics['particle'][name] = {
                        'patches': diagnostics, 'class_weight': self.particle_weights[name],
                        'weighted_particle_loss': float(values.detach().mean()) * coefficient}
                particle = ddp_sample_mean(weighted_sum, logits.shape[0])
                self.last_diagnostics['weighted_particle_loss'] = float(particle.detach())
                base = base + particle
        return base
