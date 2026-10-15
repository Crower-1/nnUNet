"""Capture post-spatial-augmentation padding before the base pipeline replaces -1 with 0."""
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


class CaptureParticleValidity(BasicTransform):
    def apply(self, data_dict, **params):
        data_dict['particle_valid'] = data_dict['segmentation'][:1] >= 0
        return data_dict


def preserve_particle_validity(compose):
    for index, transform in enumerate(compose.transforms):
        if isinstance(transform, RemoveLabelTansform) and transform.label_value == -1:
            compose.transforms.insert(index, CaptureParticleValidity())
            return compose
    raise RuntimeError('Cannot locate padding replacement in augmentation pipeline')
