"""Synthetic particle loss forward/backward timing; never a training-quality benchmark."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from nnunetv2.utilities.label_geometry import geometry_options, resolve_geometry
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.geometry_wrapper import GeometryLossWrapper


def benchmark(device='cpu', backend='scipy_reference', size=32, repeats=5):
    torch.manual_seed(1)
    torch.set_num_threads(2)
    device = torch.device(device)
    labels = torch.zeros(2, 1, size, size, size, device=device, dtype=torch.long)
    for i, z in enumerate(range(4, size-3, 6)):
        for j, y in enumerate(range(4, size-3, 6)):
            for k, x in enumerate(range(4, size-3, 6)):
                labels[:, :, z, y, x] = 1 + (i+j+k) % 2
    valid = torch.ones_like(labels, dtype=torch.bool)
    logits = torch.randn(2, 3, size, size, size, device=device, requires_grad=True)
    metadata = resolve_geometry({'schema_version': 1, 'targets': [dict(label_names=[n], geometry_type='particle')
        for n in ('ribo', 'novel_particle')]}, {'background': 0, 'ribo': 1, 'novel_particle': 2})
    report = dict(synthetic=True, torch_version=torch.__version__, device=str(device), backend=backend,
                  shape=list(logits.shape), threads=2, repeats=repeats, mapping=metadata, runs={})
    for name, enabled in (('P0', False), ('P1', True)):
        options = geometry_options({'enabled': True, 'particle': {'enabled': enabled, 'backend': backend}})
        base = DC_and_CE_loss({'batch_dice': False, 'do_bg': False, 'smooth': 1e-5, 'ddp': False}, {})
        wrapper = GeometryLossWrapper(base, metadata, options)
        timings = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for iteration in range(repeats + 1):
            logits.grad = None
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss = wrapper(logits, labels, particle_valid=valid)
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-started
            if iteration:
                timings.append(elapsed)
        item = dict(loss=float(loss.detach()), forward_backward_seconds_median=float(np.median(timings)),
                    forward_backward_seconds=timings, finite_gradient=bool(torch.isfinite(logits.grad).all()))
        if enabled:
            patches = [p for c in wrapper.last_diagnostics['particle'].values() for p in c['patches']]
            item.update(last_geometry_seconds=sum(p['geometry_seconds'] for p in patches),
                        last_reduction_seconds=sum(p['reduction_seconds'] for p in patches),
                        cc_counts=[p['cc_count'] for p in patches])
        if device.type == 'cuda':
            item['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            free, total = torch.cuda.mem_get_info(device)
            item['device_used_bytes_sample'] = total-free
            if backend == 'cupy':
                import cupy
                with cupy.cuda.Device(device.index or 0):
                    item['cupy_pool_reserved_bytes'] = cupy.get_default_memory_pool().total_bytes()
        report['runs'][name] = item
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=['scipy_reference', 'cupy'], default='scipy_reference')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.size < 8 or args.repeats < 1:
        raise ValueError('size>=8 and repeats>=1 required')
    report = benchmark(args.device, args.backend, args.size, args.repeats)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['runs'], indent=2))


if __name__ == '__main__':
    main()
