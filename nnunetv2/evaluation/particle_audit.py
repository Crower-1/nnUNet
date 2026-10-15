"""Audit semantic particle masks without modifying labels; export CC/Voronoi slices."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from nnunetv2.utilities.label_geometry import resolve_geometry
from nnunetv2.training.data_augmentation.custom_transforms.particle_targets import particle_partition


def instance_correspondence(components, foreground, instance_ids):
    """Optional reliable ID comparison; never changes semantic CC seeds."""
    ids = np.asarray(instance_ids)
    if ids.shape != components.shape or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError('Instance IDs must be aligned nonnegative integers with 0 for background')
    assigned = foreground & (ids > 0)
    pairs = np.unique(np.stack((components[assigned], ids[assigned]), axis=1), axis=0)
    _, ids_per_cc = np.unique(pairs[:, 0], return_counts=True)
    unique_ids, ccs_per_id = np.unique(pairs[:, 1], return_counts=True)
    return dict(gt_id_count=len(unique_ids), cc_with_multiple_ids=int((ids_per_cc > 1).sum()),
                ids_in_multiple_cc=int((ccs_per_id > 1).sum()),
                foreground_without_id=int((foreground & (ids == 0)).sum()))


def center_correspondence(components, centers):
    """Centers are audit-only, expressed in final-grid D,H,W voxel coordinates."""
    points = np.asarray(centers, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError('Center coordinates must be finite')
    indices = np.rint(points).astype(int)
    inside = ((indices >= 0) & (indices < np.asarray(components.shape))).all(1)
    assigned = components[tuple(indices[inside].T)]
    _, counts = np.unique(assigned[assigned > 0], return_counts=True)
    return dict(center_count=len(points), centers_outside_volume=int((~inside).sum()),
                centers_without_cc=int((assigned == 0).sum()), cc_with_multiple_centers=int((counts > 1).sum()))


def audit_mask(mask, metadata, ignore_label=None, instance_ids=None, centers=None):
    mask = torch.as_tensor(np.array(mask, copy=True))
    valid = mask >= 0
    unknown = mask == ignore_label if ignore_label is not None else torch.zeros_like(valid)
    valid &= ~unknown
    result, previews = {}, {}
    for entry in metadata['targets']:
        if entry['geometry_type'] != 'particle':
            continue
        name = entry['label_names'][0]
        partition = particle_partition(mask == entry['label_ids'][0], valid, unknown=unknown)
        sizes = np.asarray(partition.diagnostics['component_sizes'])
        stats = dict(partition.diagnostics)
        stats.update(empty=partition.count == 0, single_cc=partition.count == 1,
            singleton_fraction=float(np.mean(sizes == 1)) if len(sizes) else 0.,
            small_cc_fraction_le8=float(np.mean(sizes <= 8)) if len(sizes) else 0.,
            volume_quantiles=np.percentile(sizes, [0, 25, 50, 75, 100]).tolist() if len(sizes) else [],
            equivalent_diameter_voxels=(2 * (3 * sizes / (4 * np.pi))**(1/3)).tolist(),
            boundary_cc_fraction=partition.diagnostics['boundary_cc_count']/max(partition.count, 1))
        result[name] = stats
        foreground = mask == entry['label_ids'][0]
        if instance_ids is not None:
            stats['instance_correspondence'] = instance_correspondence(partition.components.cpu().numpy(),
                (foreground & valid).numpy(), instance_ids)
        if centers is not None:
            stats['center_correspondence'] = center_correspondence(partition.components.cpu().numpy(), centers.get(name, []))
        z = int(foreground.sum((1, 2)).argmax()) if partition.count else mask.shape[0] // 2
        stats['preview_z'] = z
        previews[name] = np.stack([(mask[z] == entry['label_ids'][0]).numpy(),
                                  partition.components[z].cpu().numpy(), partition.regions[z].cpu().numpy()])
    return result, previews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', required=True, help='3D semantic .npy, in final training grid order')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--patch-size', nargs=3, type=int)
    parser.add_argument('--max-patches', type=int, default=16)
    parser.add_argument('--resampled-mask', help='Optional actual post-resampling semantic .npy')
    parser.add_argument('--instance-ids', help='Optional reliable, aligned instance-ID .npy (0=background)')
    parser.add_argument('--centers-json', help='Optional audit centers: coordinate_units=voxel, axis_order=DHW, centers by label name')
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text())
    metadata = resolve_geometry(json.loads(Path(args.geometry).read_text()), dataset['labels'])
    mask = np.load(args.mask, mmap_mode='r')
    if mask.ndim != 3:
        raise ValueError('Expected a 3D semantic mask')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    centers = None
    if args.centers_json:
        payload = json.loads(Path(args.centers_json).read_text())
        if payload.get('coordinate_units') != 'voxel' or payload.get('axis_order') != 'DHW':
            raise ValueError('Audit centers must use final-grid DHW voxel coordinates')
        centers = payload['centers']
        names = {t['label_names'][0] for t in metadata['targets'] if t['geometry_type'] == 'particle'}
        if set(centers) - names:
            raise ValueError('Unbound particle center labels')
    summary, previews = audit_mask(mask, metadata, dataset['labels'].get('ignore'),
        np.load(args.instance_ids, mmap_mode='r') if args.instance_ids else None, centers)
    report = dict(case=str(args.mask), classes=summary, patches=[],
                  interpretation='Semantic-only anomalies cannot establish true merge/split rates.')
    for index, (name, images) in enumerate(previews.items()):
        np.savez_compressed(output / f'class_{index}_slices.npz', target=images[0], cc=images[1], voronoi=images[2])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, image, title in zip(axes, images, ('Target', 'GT CC', 'Voronoi')):
            colormap = plt.get_cmap('tab20').copy() if title != 'Target' else plt.get_cmap('gray')
            if title != 'Target':
                colormap.set_bad('black')
                image = np.ma.masked_equal(image, 0)
            ax.imshow(image, interpolation='nearest', cmap=colormap)
            ax.set_title(f'{name}: {title}, z={summary[name]["preview_z"]}')
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(output / f'class_{index}_slices.png')
        plt.close(fig)
    if args.patch_size:
        if min(args.patch_size) <= 0:
            raise ValueError('Patch size must be positive')
        origins = itertools.product(*(range(0, n, step) for n, step in zip(mask.shape, args.patch_size)))
        for origin in itertools.islice(origins, args.max_patches):
            patch = mask[tuple(slice(start, start+size) for start, size in zip(origin, args.patch_size))]
            stats, _ = audit_mask(patch, metadata, dataset['labels'].get('ignore'))
            report['patches'].append(dict(origin=origin, classes=stats))
    if args.resampled_mask:
        after, _ = audit_mask(np.load(args.resampled_mask), metadata, dataset['labels'].get('ignore'))
        report['after_resampling'] = after
        report['resampling_cc_count_delta'] = {name: after[name]['cc_count']-summary[name]['cc_count'] for name in summary}
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
