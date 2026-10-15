"""Versioned particle objective options shared by training, snapshots and audit tools."""
import math

ALGORITHM_VERSION = 'semantic-cc-voronoi-dicece-v1'
UPSTREAM_COMMIT = 'e5013bdb560eee544a94e1f068cb5cb4d003a51c'


def particle_options(options=None):
    result = dict(enabled=True, targets='all_cfg_particles', profile='cc_dice_ce', weight=0.25,
                  class_weights={}, apply_to='highest_resolution', component_source='semantic_cc',
                  connectivity=26, distance_units='voxel', empty_target='skip_aux',
                  valid_region='post_augmentation_nonpadding_and_nonignore',
                  backend='scipy_reference', eps=1e-7, algorithm_version=ALGORITHM_VERSION,
                  upstream_commit=UPSTREAM_COMMIT)
    options = options or {}
    if not isinstance(options, dict) or set(options) - set(result):
        raise ValueError('Unknown or invalid particle options')
    for key in ('targets', 'apply_to', 'component_source', 'connectivity', 'distance_units',
                'empty_target', 'valid_region', 'algorithm_version', 'upstream_commit'):
        if key in options and options[key] != result[key]:
            raise ValueError(f'Unsupported particle {key}: {options[key]}')
    result.update(options)
    if not isinstance(result['enabled'], bool):
        raise ValueError('particle.enabled must be boolean')
    if result['profile'] not in ('cc_dice', 'cc_dice_ce'):
        raise ValueError('particle.profile must be cc_dice or cc_dice_ce')
    if result['backend'] not in ('scipy_reference', 'cupy'):
        raise ValueError('particle.backend must be explicitly scipy_reference or cupy')
    for key in ('weight', 'eps'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid particle.{key}')
    if result['eps'] == 0:
        raise ValueError('particle.eps must be positive')
    weights = result['class_weights']
    if not isinstance(weights, dict) or any(not isinstance(k, str) or isinstance(v, bool) or
            not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for k, v in weights.items()):
        raise ValueError('particle.class_weights must contain positive finite named weights')
    return result


def particle_active(options, targets):
    particle = options.get('particle', {})
    return (options['enabled'] and particle.get('enabled', False) and particle.get('weight', 0) > 0
            and any(t['geometry_type'] == 'particle' for t in targets))


def resolved_class_weights(options, targets):
    names = [t['label_names'][0] for t in targets if t['geometry_type'] == 'particle']
    weights = options['class_weights']
    if set(weights) - set(names):
        raise ValueError(f'Unbound particle class weights: {set(weights) - set(names)}')
    return {name: weights.get(name, 1.) for name in names}
