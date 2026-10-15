"""FP32 class-wise CC-DiceCE with GT partitions and O(N+K) scatter reductions.

Adapted mathematical contract from Learning-to-Look-Closer (Apache-2.0), pinned
sources and differences in third_party/particle/README.md. No global compiler settings.
"""
import time
import torch
import torch.distributed as dist
from torch.nn import functional as F
from nnunetv2.training.data_augmentation.custom_transforms.particle_targets import particle_partition


def one_vs_rest_logit(logits, channel):
    """Equivalent to the selected full-softmax probability, including competitor gradients."""
    if logits.shape[1] < 2 or not 0 <= channel < logits.shape[1]:
        raise ValueError('Invalid multiclass channel')
    others = [c for c in range(logits.shape[1]) if c != channel]
    with torch.autocast(logits.device.type, enabled=False):
        values = logits.float()
        return values[:, channel] - torch.logsumexp(values[:, others], dim=1)


def cc_terms(probability, ce_map, truth, regions, count, eps=1e-7):
    """Return object-mean Dice loss and CE for one patch; region 0 never contributes."""
    with torch.autocast(probability.device.type, enabled=False):
        probability, ce_map, truth = probability.float(), ce_map.float(), truth.float()
        if count == 0:
            zero = probability.sum() * 0 + ce_map.sum() * 0
            return zero, zero
        ids = regions.reshape(-1).long()
        def segment(values):
            return torch.zeros(count + 1, device=values.device, dtype=torch.float32).scatter_add(
                0, ids, values.reshape(-1))[1:]
        intersection = segment(probability * truth)
        denominator = segment(probability) + segment(truth)
        dice = 1 - 2 * intersection / denominator.clamp_min(eps)
        ce = segment(ce_map) / segment(torch.ones_like(probability)).clamp_min(1)
        return dice.mean(), ce.mean()


def ddp_sample_mean(local_sum, local_count):
    """Global sample mean with the scale needed before DDP averages parameter gradients.

    All ranks call once, even with empty local particle foreground. Forward is
    the global mean; backward is local_sum * world_size / global_sample_count.
    Existing base/membrane/filament objectives retain their original reductions.
    """
    if not dist.is_available() or not dist.is_initialized():
        return local_sum / max(local_count, 1)
    packed = torch.stack((local_sum.detach(), local_sum.new_tensor(local_count)))
    dist.all_reduce(packed)
    scaled = local_sum * (dist.get_world_size() / packed[1].clamp_min(1))
    return scaled + (packed[0] / packed[1].clamp_min(1) - scaled.detach())


def particle_class_loss(logits, labels, valid, entry, options, ignore_label=None):
    """Return per-patch values and detached diagnostics for a single named class."""
    logit = one_vs_rest_logit(logits, entry['channels'][0])
    truth = (labels[:, 0] == entry['label_ids'][0]) & valid[:, 0]
    with torch.autocast(logits.device.type, enabled=False):
        probability = logit.sigmoid()
        ce_map = F.binary_cross_entropy_with_logits(logit, truth.float(), reduction='none')
        values, diagnostics = [], []
        for b in range(logits.shape[0]):
            unknown = labels[b, 0] == ignore_label if ignore_label is not None else None
            partition = particle_partition(truth[b], valid[b, 0], options['backend'], unknown)
            started = time.perf_counter()
            dice, ce = cc_terms(probability[b], ce_map[b], truth[b], partition.regions,
                                partition.count, options['eps'])
            value = dice + ce if options['profile'] == 'cc_dice_ce' else dice
            values.append(value)
            diagnostics.append({**partition.diagnostics, 'cc_dice_raw': float(dice.detach()),
                'cc_ce_raw': float(ce.detach()), 'reduction_seconds': time.perf_counter() - started})
        return torch.stack(values), diagnostics
