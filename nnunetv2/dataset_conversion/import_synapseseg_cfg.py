"""Import SynapseSeg JSON cfg into explicit, validated named training targets."""
import argparse
import json
from pathlib import Path
from nnunetv2.utilities.label_geometry import resolve_geometry


def import_cfg(cfg, dataset, overrides=None, class_names=None):
    categories = {c['category']: c for c in cfg}
    if len(categories) != len(cfg):
        raise ValueError('Duplicate cfg category names')
    if any(not isinstance(c.get('particle', False), bool) for c in cfg):
        raise ValueError('cfg particle flags must be boolean')
    overrides = overrides or []
    override_names = {tuple(sorted(t['label_names'])) for t in overrides}
    for entry in overrides:
        if not entry.get('override_reason'):
            raise ValueError('Explicit geometry overrides require override_reason')
    targets = []
    bound = set()
    for name in dataset['labels']:
        if name in ('background', 'ignore'):
            continue
        boundary = name.endswith('_memb')
        category = name[:-5] if boundary else name
        info = categories.get(category)
        geometry = 'default'
        if info:
            bound.add(category)
            if info.get('particle'):
                ambiguous = boundary or info.get('with_membrane') or info.get('filament') or name == 'membrane'
                if ambiguous and (name,) not in override_names:
                    raise ValueError(f'Ambiguous particle flags/output role for {name}; supply an explicit override with reason')
                if not boundary:
                    geometry = 'particle'
            if category != 'mt':
                if not info.get('particle') and (name == 'membrane' or (boundary and info.get('with_membrane'))):
                    geometry = 'membrane'
                elif not info.get('particle') and not boundary and info.get('filament'):
                    geometry = 'filament'
        targets.append(dict(category=category, role='boundary' if boundary else 'primary' if name == 'membrane' else 'body',
                            label_names=[name], geometry_type=geometry))
    # Explicit entries replace automatic targets for exactly the same named set;
    # unions are additional derived targets and never add heads.
    for entry in overrides:
        targets = [t for t in targets if set(t['label_names']) != set(entry['label_names'])]
        targets.append(entry)
    metadata = dict(schema_version=1, targets=targets)
    resolved = resolve_geometry(metadata, dataset['labels'], class_names)
    report = {'unbound_categories': sorted(set(categories) - bound),
              'default_unmatched_labels': [t['label_names'][0] for t in targets if t['category'] not in categories],
              'particle_targets': [t['label_names'][0] for t in targets if t['geometry_type'] == 'particle'],
              'overrides': overrides}
    return metadata, resolved, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--plans', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--overrides', help='JSON list of explicit targets')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text())
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    dataset = read(args.dataset)
    config = PlansManager(read(args.plans)).get_configuration(args.configuration, dataset)
    metadata, resolved, report = import_cfg(read(args.cfg), dataset,
        read(args.overrides) if args.overrides else None, config.network_arch_init_kwargs.get('class_names'))
    Path(args.output).write_text(json.dumps(metadata, indent=2) + '\n')
    print('category\trole\tlabels\tids\tchannels\tgeometry')
    for t in resolved['targets']:
        print('\t'.join(str(t[k]) for k in ('category', 'role', 'label_names', 'label_ids', 'channels', 'geometry_type')))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
