"""Rank-local JSONL geometry diagnostics and data-wait timing, outside autograd."""
import json
import copy
import time
from pathlib import Path
import numpy as np
import torch


class TimedBatches:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self

    def __next__(self):
        started = time.perf_counter()
        batch = next(self.source)
        batch['data_wait_seconds'] = time.perf_counter() - started
        return batch

    def __getattr__(self, name):
        return getattr(self.source, name)


def write_geometry_step(folder, rank, epoch, phase, keys, diagnostics, elapsed, data_wait, device, backend):
    record = dict(epoch=epoch, phase=phase, keys=list(keys), step_seconds=elapsed,
                  data_wait_seconds=data_wait, **copy.deepcopy(diagnostics))
    for item in record.get('particle', {}).values():
        patches = item['patches']
        counts = [p['cc_count'] for p in patches]
        count = sum(counts)
        sizes = [size for p in patches for size in p['component_sizes']]
        item.update(patch_count=len(patches), cc_count=count,
                    empty_fraction=sum(k == 0 for k in counts) / max(len(patches), 1),
                    single_cc_patch_fraction=sum(k == 1 for k in counts) / max(len(patches), 1),
                    boundary_cc_fraction=sum(p['boundary_cc_count'] for p in patches) / max(count, 1),
                    singleton_cc_fraction=sum(s == 1 for s in sizes) / max(count, 1))
        for p in patches:
            sizes = np.asarray(p.pop('component_sizes'))
            p['volume_quantiles'] = np.percentile(sizes, [0, 25, 50, 75, 100]).tolist() if len(sizes) else []
            p['equivalent_diameter_voxel_quantiles'] = (2*(3*np.asarray(p['volume_quantiles'])/(4*np.pi))**(1/3)).tolist()
            p['small_cc_fraction_le8'] = float(np.mean(sizes <= 8)) if len(sizes) else 0.
    if device.type == 'cuda':
        record['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        free, total = torch.cuda.mem_get_info(device)
        record['device_used_bytes_sample'] = total - free
        if backend == 'cupy':
            import cupy
            with cupy.cuda.Device(device.index):
                record['cupy_pool_reserved_bytes'] = cupy.get_default_memory_pool().total_bytes()
    path = Path(folder, f'geometry_steps_rank{rank}.jsonl')
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
