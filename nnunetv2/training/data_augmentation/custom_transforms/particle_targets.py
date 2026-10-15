"""Eager GT-only connected components and nearest-foreground Voronoi partitions.

One 3D class/sample at a time; region 0 is invalid. No instance one-hot volumes.
SciPy and optional CuPy implement the same Euclidean voxel-grid contract. Exact
equidistant ties may have different, but valid, nearest-seed assignments.
"""
from dataclasses import dataclass
import time
import torch


@dataclass
class ParticlePartition:
    regions: torch.Tensor
    components: torch.Tensor
    count: int
    diagnostics: dict


def check_backend(backend, device):
    if backend == 'scipy_reference':
        from scipy import ndimage  # noqa: F401
    elif backend == 'cupy':
        if device.type != 'cuda':
            raise ValueError('particle backend=cupy requires CUDA; no automatic CPU fallback')
        try:
            import cupy
            from cupyx.scipy.ndimage import distance_transform_edt  # noqa: F401
            with cupy.cuda.Device(device.index):
                cupy.zeros(1)
        except ImportError as error:
            raise RuntimeError('Install a compatible CuPy explicitly for backend=cupy') from error
    else:
        raise ValueError(f'Unknown particle backend: {backend}')


def _partition_arrays(foreground, valid, unknown, xp, ndi):
    components, count = ndi.label(foreground, structure=xp.ones((3, 3, 3), dtype=bool))
    count = int(count)
    if count == 0:
        regions = xp.zeros_like(components)
    elif count == 1:
        regions = valid.astype(components.dtype)
    else:
        nearest = ndi.distance_transform_edt(~foreground, return_distances=False, return_indices=True)
        regions = components[tuple(nearest)]
        regions[~valid] = 0
    sizes = xp.bincount(components.ravel(), minlength=count + 1)[1:]
    faces = xp.zeros_like(valid)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            faces[tuple(index)] = True
    # Include the valid-field boundary as well as the rectangular crop boundary.
    faces |= ndi.binary_dilation(~valid, structure=xp.ones((3, 3, 3), dtype=bool))
    boundary_ids = xp.unique(components[foreground & faces])
    unknown_ids = xp.unique(components[foreground & ndi.binary_dilation(
        unknown, structure=xp.ones((3, 3, 3), dtype=bool))])
    return components, regions, count, sizes, boundary_ids.size, unknown_ids.size


@torch.compiler.disable
@torch.no_grad()
def particle_partition(target, valid, backend='scipy_reference', unknown=None):
    """Partition every valid voxel by its closest GT component, excluding ignore/padding seeds."""
    if target.ndim != 3 or target.shape != valid.shape:
        raise ValueError('Particle partitions require equal 3D target and valid masks')
    if backend not in ('scipy_reference', 'cupy'):
        raise ValueError(f'Unknown particle backend: {backend}')
    started = time.perf_counter()
    foreground = target.bool() & valid.bool()
    unknown = torch.zeros_like(valid, dtype=torch.bool) if unknown is None else unknown.bool()
    if not bool(foreground.any()):
        zero = torch.zeros_like(target, dtype=torch.int64)
        return ParticlePartition(zero, zero, 0, dict(cc_count=0, component_sizes=[],
            boundary_cc_count=0, unknown_touching_cc_count=0, valid_voxels=int(valid.sum()),
            geometry_seconds=time.perf_counter() - started))
    if backend == 'scipy_reference':
        import numpy as np
        from scipy import ndimage
        arrays = _partition_arrays(foreground.cpu().numpy(), valid.bool().cpu().numpy(),
                                   unknown.cpu().numpy(), np, ndimage)
        cc, regions, count, sizes, boundary, unknown_count = arrays
        components = torch.from_numpy(cc.astype('int64')).to(target.device)
        regions = torch.from_numpy(regions.astype('int64')).to(target.device)
        sizes = sizes.tolist()
    elif backend == 'cupy':
        check_backend(backend, target.device)
        import cupy as cp
        from cupyx.scipy import ndimage
        # Run CuPy on this rank's device and the current PyTorch stream. DLPack
        # owns the exchanged memory; no default-stream race or dangling ndarray.
        with cp.cuda.Device(target.device.index), cp.cuda.ExternalStream(torch.cuda.current_stream(target.device).cuda_stream):
            arrays = _partition_arrays(cp.from_dlpack(foreground.contiguous()),
                cp.from_dlpack(valid.bool().contiguous()), cp.from_dlpack(unknown.contiguous()), cp, ndimage)
            cc, region_array, count, sizes, boundary, unknown_count = arrays
            components = torch.from_dlpack(cc.astype(cp.int64))
            regions = torch.from_dlpack(region_array.astype(cp.int64))
            sizes = sizes.get().tolist()
    else:
        raise ValueError(f'Unknown particle backend: {backend}')
    return ParticlePartition(regions, components, count, dict(cc_count=count, component_sizes=sizes,
        boundary_cc_count=int(boundary), unknown_touching_cc_count=int(unknown_count),
        valid_voxels=int(valid.sum()), geometry_seconds=time.perf_counter() - started))
