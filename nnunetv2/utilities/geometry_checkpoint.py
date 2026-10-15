"""Canonical geometry snapshots and explicit migration of pre-particle experiments."""
import copy
import hashlib
import json
from nnunetv2.utilities.label_geometry import geometry_options, resolve_geometry
from nnunetv2.utilities.particle_config import resolved_class_weights


def geometry_snapshot(options, metadata, legacy_particle_disabled=False):
    state = copy.deepcopy(dict(options=options, metadata=metadata))
    if legacy_particle_disabled:
        state['legacy_particle_disabled'] = True
    state['config_hash'] = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return state


def restore_geometry(saved, current, labels, class_names=None):
    if saved is None:
        raise ValueError('Geometry configuration differs: checkpoint has no geometry snapshot')
    if 'particle' not in saved['options'] or saved.get('legacy_particle_disabled'):
        # Resume the saved objective, never infer particle targets from today's cfg.
        options = geometry_options({**saved['options'], 'particle': {'enabled': False}})
        metadata = saved['metadata']
        if metadata is not None:
            resolved = resolve_geometry(metadata, labels, class_names)
            if resolved != metadata:
                raise ValueError('Saved geometry label/channel mapping differs from the current dataset')
        if saved.get('config_hash') and geometry_snapshot(saved['options'], metadata, True) != saved:
            raise ValueError('Saved legacy geometry snapshot hash is inconsistent')
        options['particle']['class_weights'] = resolved_class_weights(options['particle'], (metadata or {}).get('targets', []))
        return geometry_snapshot(options, metadata, True)
    if saved != current:
        raise ValueError('Geometry configuration differs from checkpoint; use a new fine-tuning run')
    return copy.deepcopy(saved)
