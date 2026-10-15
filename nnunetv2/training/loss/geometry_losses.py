"""FP32 probability-domain geometry losses.

MemBrain/clDice adaptation: see third_party/geometry/ for pinned sources and licenses.
No sigmoid is applied here: callers supply selected channels (or unions) of multiclass softmax(logits.float()).
"""
import math
import torch
from torch.nn import functional as F


def soft_erode(image):
    return -F.max_pool3d(-image, 3, 1, 1)


def soft_open(image):
    return F.max_pool3d(soft_erode(image), 3, 1, 1)


def soft_skeleton(image, iterations):
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def gaussian_filter(image, kernel_size, sigma):
    grid = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - (kernel_size - 1) / 2
    xyz = torch.stack(torch.meshgrid(grid, grid, grid, indexing='ij'), dim=-1)
    kernel = torch.exp(-torch.sum(xyz**2, dim=-1) / (2 * sigma**2))
    kernel = kernel / ((2 * math.pi * sigma**2) ** (3 / 2))
    kernel = kernel / kernel.sum()
    return F.conv3d(image, kernel[None, None], padding=kernel_size // 2)


def membrane_target(target, config):
    with torch.no_grad(), torch.autocast(target.device.type, enabled=False):
        smoothed = gaussian_filter(target.float(), config.membrane_kernel, config.membrane_sigma)
        return soft_skeleton(smoothed * config.membrane_amplitude, config.membrane_iterations)


def reliable_region(valid, radius):
    """Chebyshev erosion includes the patch exterior as invalid context.

    Repeated radius-one erosion avoids materializing a (2r+1)^3 pooling kernel.
    Gaussian r + N erosions + radius-two opening bounds the GT dependency;
    prediction needs only N+2. Use the larger radius for both score branches.
    """
    region = valid.float()
    for _ in range(radius):
        region = -F.max_pool3d(-F.pad(region, (1, 1, 1, 1, 1, 1), value=0), 3, 1)
    return region.bool()


def skeleton_recall(probability, support, smooth=1e-5):
    with torch.autocast(probability.device.type, enabled=False):
        probability = probability.float(); support = support.float()
        axes = tuple(range(1, probability.ndim))
        size = support.sum(axes)
        recall = ((probability * support).sum(axes) + smooth) / (size + smooth)
        return torch.where(size > 0, 1 - recall, torch.zeros_like(recall)).mean()


def membrain_surface_loss(probability, target, middle, score_valid, config):
    with torch.autocast(probability.device.type, enabled=False):
        probability = probability.float(); target = target.float()
        predicted = soft_skeleton(probability, config.membrane_iterations) * score_valid
        middle = middle.float() * score_valid
        axes = (1, 2, 3, 4); smooth = config.membrane_smooth
        precision = ((predicted * target).sum(axes) + smooth) / (predicted.sum(axes) + smooth)
        sensitivity = ((middle * probability).sum(axes) + smooth) / (middle.sum(axes) + smooth)
        loss = 1 - 2 * precision * sensitivity / (precision + sensitivity)
        return torch.where(middle.sum(axes) > 0, loss, torch.zeros_like(loss)).mean()
