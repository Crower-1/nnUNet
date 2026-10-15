"""Bind named geometry targets to integer-label softmax channels without reordering."""
import math
from nnunetv2.utilities.particle_config import particle_options


def resolve_geometry(metadata, labels, class_names=None):
    if metadata.get('schema_version') != 1:
        raise ValueError('label_geometry requires schema_version=1')
    if any(not isinstance(v, int) or isinstance(v, bool) for v in labels.values()):
        raise ValueError('Geometry v1 supports integer labels only, not regions')
    ordered = sorted((v, k) for k, v in labels.items() if k != 'ignore')
    if [v for v, _ in ordered] != list(range(len(ordered))):
        raise ValueError('Expected unique contiguous labels starting at background=0')
    expected = [k for _, k in ordered]
    if labels.get('background') != 0:
        raise ValueError('background must have label 0')
    if class_names is not None and list(class_names) != expected:
        raise ValueError(f'Output channel mismatch: class_names={class_names}, label order={expected}')
    targets = []
    seen = set()
    for entry in metadata.get('targets', []):
        names = entry.get('label_names', [])
        geometry = entry.get('geometry_type')
        if geometry not in ('default', 'membrane', 'filament', 'particle') or not names or len(set(names)) != len(names):
            raise ValueError(f'Invalid geometry target: {entry}')
        if geometry == 'particle' and len(names) != 1:
            raise ValueError('Particle CC supervision must be class-wise, not a label union')
        if any(n not in labels or n in ('background', 'ignore') for n in names):
            raise ValueError(f'Unbound geometry target: {names}')
        key = tuple(sorted(names))
        if key in seen:
            raise ValueError(f'Duplicate target: {names}')
        seen.add(key)
        targets.append({**entry, 'label_ids': [labels[n] for n in names],
                        'channels': [expected.index(n) for n in names]})
    return {'schema_version': 1, 'targets': targets}


def geometry_options(options, has_particle_targets=False):
    defaults = dict(enabled=has_particle_targets, apply_to='highest_resolution', surface_weight=0.25,
                    skeleton_weight=0.25, reduction='mean_per_geometry', skeleton_dilation=2,
                    membrane_iterations=5, membrane_kernel=15, membrane_sigma=2.,
                    membrane_amplitude=1.5, membrane_smooth=3., particle={})
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown geometry options: {unknown}')
    defaults.update(options)
    defaults['particle'] = particle_options(defaults['particle'])
    if not isinstance(defaults['enabled'], bool):
        raise ValueError('enabled must be boolean')
    if defaults['apply_to'] != 'highest_resolution' or defaults['reduction'] != 'mean_per_geometry':
        raise ValueError('Only highest_resolution / mean_per_geometry is supported')
    for key in ('surface_weight', 'skeleton_weight', 'membrane_sigma', 'membrane_amplitude', 'membrane_smooth'):
        value = defaults[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key.startswith('membrane') and value == 0):
            raise ValueError(f'Invalid {key}: {value}')
    for key in ('skeleton_dilation', 'membrane_iterations', 'membrane_kernel'):
        value = defaults[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (1 if key == 'membrane_kernel' else 0):
            raise ValueError(f'Invalid {key}: {value}')
    if defaults['membrane_kernel'] % 2 != 1:
        raise ValueError('membrane_kernel must be odd')
    return defaults
