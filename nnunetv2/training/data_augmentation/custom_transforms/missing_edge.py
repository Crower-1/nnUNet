from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch always available in training env
    torch = None
    _TORCH_AVAILABLE = False

def _create_missing_wedge_mask(
    spatial_shape: Tuple[int, int],
    missing_angles: Sequence[float],
) -> np.ndarray:
    """Generate a 2D mask that emulates missing wedge acquisition artefacts."""
    if len(spatial_shape) != 2:
        raise ValueError(f"Expected 2D spatial shape, got {spatial_shape}")

    missing = np.deg2rad(90 - np.asarray(missing_angles, dtype=np.float32))
    height, width = spatial_shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    yy -= height / 2.0
    xx -= width / 2.0

    theta = np.abs(np.arctan2(yy, xx))
    radius_sq = (min(height, width) / 2.0) ** 2
    inside = (xx ** 2 + yy ** 2) <= radius_sq

    mask = np.zeros(spatial_shape, dtype=np.float32)
    condition_pos = (xx > 0) & (yy > 0) & (theta < missing[0])
    condition_neg = (xx < 0) & (yy < 0) & (theta < missing[0])
    condition_pos_neg = (xx > 0) & (yy < 0) & (theta < missing[1])
    condition_neg_pos = (xx < 0) & (yy > 0) & (theta < missing[1])

    combined = inside & (condition_pos | condition_neg | condition_pos_neg | condition_neg_pos)
    mask[combined] = 1.0
    mask[np.isclose(yy, 0.0)] = 1.0
    return mask


def _apply_missing_wedge(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply the missing wedge mask in Fourier space to a 2D/3D array."""
    template = data
    if _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        working = data.detach().cpu().numpy()
    else:
        working = np.asarray(data)

    if working.shape[-2:] != mask.shape:
        raise ValueError(f"Mask shape {mask.shape} incompatible with data shape {working.shape}")

    fft_mask = np.fft.fftshift(mask)
    expand_shape = (1,) * (working.ndim - 2) + fft_mask.shape
    fft_mask = fft_mask.reshape(expand_shape)

    degraded = np.fft.ifft2(np.fft.fft2(working, axes=(-2, -1)) * fft_mask, axes=(-2, -1))
    degraded = np.real(degraded)
    if _TORCH_AVAILABLE and isinstance(template, torch.Tensor):
        return torch.as_tensor(degraded, dtype=template.dtype, device=template.device)
    return degraded.astype(template.dtype, copy=False)


def _compose_plane_permutations(ndim: int, plane_axes: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    plane_axes = tuple(int(ax) for ax in plane_axes)
    if len(plane_axes) != 2:
        raise ValueError(f"Plane axes must contain exactly two entries, got {plane_axes}")
    if any(ax < 0 or ax >= ndim for ax in plane_axes):
        raise ValueError(f"Plane axes {plane_axes} out of bounds for ndim={ndim}")
    remaining = tuple(ax for ax in range(ndim) if ax not in plane_axes)
    perm = remaining + plane_axes
    inv = [0] * ndim
    for idx, ax in enumerate(perm):
        inv[ax] = idx
    return perm, tuple(inv)


def _apply_missing_wedge_on_plane(data: np.ndarray, mask: np.ndarray, plane_axes: Sequence[int]) -> np.ndarray:
    """Reorient data so the selected plane forms the last two dims, apply wedge, then revert."""
    perm, inv_perm = _compose_plane_permutations(data.ndim, plane_axes)
    expected_shape = tuple(int(data.shape[ax]) for ax in plane_axes)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} incompatible with plane axes {plane_axes} (expected {expected_shape})")

    if _TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        working = data.permute(*perm)
        degraded = _apply_missing_wedge(working, mask)
        return degraded.permute(*inv_perm)
    working = np.transpose(data, perm)
    degraded = _apply_missing_wedge(working, mask)
    return np.transpose(degraded, inv_perm)


class MissingEdgeTransform(AbstractTransform):
    """Apply missing edge augmentation to emulate limited-angle tomography artefacts."""

    def __init__(
        self,
        missing_angles: Sequence[float] = (30, 30),
        apply_to_channels: Iterable[int] | None = None,
        data_key: str = "data",
    ):
        self.missing_angles = tuple(missing_angles)
        self.apply_to_channels = apply_to_channels
        self.data_key = data_key
        self._mask_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _get_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        if shape not in self._mask_cache:
            self._mask_cache[shape] = _create_missing_wedge_mask(shape, self.missing_angles)
        return self._mask_cache[shape]

    def __call__(self, **data_dict):
        key = self.data_key if self.data_key in data_dict else 'image' if 'image' in data_dict else None
        if key is None:
            return data_dict

        data = data_dict[key]
        if data is None:
            return data_dict
        if data.ndim < 3:
            return data_dict
        spatial_ndim = data.ndim - 1
        if spatial_ndim < 2:
            return data_dict

        if spatial_ndim == 2:
            plane_axes = (0, 1)
        else:
            last = spatial_ndim - 1
            second = spatial_ndim - 2
            third = spatial_ndim - 3
            plane_options = ((second, last), (third, last), (third, second))
            plane_axes = plane_options[np.random.randint(len(plane_options))]

        spatial_shape = data.shape[1:]
        mask_shape = tuple(int(spatial_shape[ax]) for ax in plane_axes)
        mask = self._get_mask(mask_shape)
        channels = self.apply_to_channels if self.apply_to_channels is not None else range(data.shape[0])

        for c in channels:
            data[c] = _apply_missing_wedge_on_plane(data[c], mask, plane_axes)

        data_dict[key] = data
        return data_dict

def main():
    import argparse
    import mrcfile
    parser = argparse.ArgumentParser(description="Apply missing wedge (mw) to a 3D MRC volume and save to new MRC.")
    parser.add_argument("input_mrc", type=str, help="Path to input 3D MRC (shape Z,Y,X).")
    parser.add_argument("output_mrc", type=str, help="Path to save output MRC.")
    parser.add_argument("--angles", type=str, default="30,30",
                        help="Missing wedge angles in degrees as 'a_pos,a_neg'. Default: 30,30")
    parser.add_argument("--invert-mask", action="store_true",
                        help="If set, use (1 - mask), i.e., truly *remove* (mask out) the wedge.")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Output dtype (e.g., float32, float16). Default: float32")
    args = parser.parse_args()

    a = args.angles.split(",")
    if len(a) != 2:
        raise ValueError("angles must be two comma-separated numbers, e.g. '30,30'")
    missing_angles = (float(a[0]), float(a[1]))

    print(f"[INFO] Reading: {args.input_mrc}")
    with mrcfile.open(args.input_mrc, permissive=True) as mrc:
        vol = mrc.data  # shape (Z, Y, X), usually a memmap
        # 提前拷到内存并转 float 进行频域计算
        vol = np.asarray(vol, dtype=np.float32)
        try:
            voxel_size = mrc.voxel_size.copy()  # nm or Å，取决于文件；mrcfile会保留单位
        except Exception:
            voxel_size = None

    print(f"[INFO] Volume shape: {vol.shape} (expect Z,Y,X); dtype=float32 for processing")
    if vol.ndim != 3:
        raise ValueError(f"Input must be 3D (Z,Y,X). Got shape: {vol.shape}")

    plane_choices = ((1, 2), (0, 2), (0, 1))
    plane_axes = plane_choices[np.random.randint(len(plane_choices))]
    axis_names = ("Z", "Y", "X")
    chosen_plane = (axis_names[plane_axes[0]], axis_names[plane_axes[1]])
    mask_shape = tuple(int(vol.shape[ax]) for ax in plane_axes)
    print(f"[INFO] Creating missing wedge mask for plane {chosen_plane} with shape {mask_shape}, angles={missing_angles}, invert={args.invert_mask}")
    mask = _create_missing_wedge_mask(mask_shape, missing_angles)
    if args.invert_mask:
        mask = 1.0 - mask
        print("[INFO] Inverting mask: keeping measured wedge only")
    print(f"[INFO] Applying missing wedge over plane {chosen_plane} in Fourier space...")
    degraded = _apply_missing_wedge_on_plane(vol, mask, plane_axes)

    out_dtype = np.dtype(args.dtype)
    degraded = degraded.astype(out_dtype, copy=False)

    print(f"[INFO] Saving: {args.output_mrc} (dtype={out_dtype})")
    with mrcfile.new(args.output_mrc, overwrite=True) as out_mrc:
        out_mrc.set_data(degraded)
        if voxel_size is not None:
            # 尽可能保留原 voxel_size
            out_mrc.voxel_size = voxel_size

    print("[DONE] Missing wedge applied and file saved.")

if __name__ == "__main__":
    main()
