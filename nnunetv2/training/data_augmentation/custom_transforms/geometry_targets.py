"""Build per-target tubular GT support after spatial augmentation; no prediction morphology."""
import numpy as np
import torch


@torch.no_grad()
def skeleton_support(target, valid, dilation=2):
    from scipy import ndimage
    from skimage.morphology import skeletonize
    supports = []
    for y, v in zip(target[:, 0].cpu().numpy(), valid[:, 0].cpu().numpy()):
        foreground = y.astype(bool) & v.astype(bool)
        skeleton = skeletonize(foreground, method='lee') > 0
        if dilation:
            skeleton = ndimage.binary_dilation(skeleton, iterations=dilation)
        supports.append(skeleton & foreground)
    return torch.from_numpy(np.stack(supports))[:, None].to(target.device).float()
