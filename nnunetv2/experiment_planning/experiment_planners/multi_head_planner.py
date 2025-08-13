from typing import Union, List, Tuple

import numpy as np

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class MultiHeadPlanner(ExperimentPlanner):
    """Planner with fixed architecture and patch size for 3d_fullres."""

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        if '3d_fullres' in data_identifier:
            patch_size = [192, 192, 192]

            num_input_channels = len(self.dataset_json['channel_names'].keys()
                                     if 'channel_names' in self.dataset_json.keys()
                                     else self.dataset_json['modality'].keys())

            architecture_kwargs = {
                "network_class_name": "nnunetv2.network_architecture.plainconv_unet_head.PlainConvUNetHead",
                "arch_kwargs": {
                    "n_stages": 6,
                    "features_per_stage": [32, 64, 128, 256, 320, 320],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[3, 3, 3]] * 6,
                    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
                    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {"inplace": True},
                },
                "_kw_requires_import": ("conv_op", "norm_op", "dropout_op", "nonlin"),
            }

            label_manager = LabelManager(self.dataset_json['labels'],
                                         self.dataset_json.get('regions_class_order'))
            if label_manager.has_regions:
                class_names = [k for k in self.dataset_json['labels'].keys()
                               if k not in ('background', 'ignore')]
            else:
                class_names = [k for k in self.dataset_json['labels'].keys() if k != 'ignore']
            architecture_kwargs['arch_kwargs']['class_names'] = class_names
            if len(class_names) != label_manager.num_segmentation_heads:
                raise RuntimeError('Number of class names does not match number of segmentation heads')

            key = str(patch_size)
            if key in _cache:
                estimate = _cache[key]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    label_manager.num_segmentation_heads,
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[key] = estimate

            reference = self.UNet_reference_val_3d * (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)
            ref_bs = self.UNet_reference_val_corresp_bs_3d
            batch_size = round((reference / estimate) * ref_bs)

            bs_corresponding_to_5_percent = round(
                approximate_n_voxels_dataset * self.max_dataset_covered / np.prod(patch_size, dtype=np.float64))
            batch_size = max(min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size)

            resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
            resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

            normalization_schemes, mask_is_used_for_norm = \
                self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

            plan = {
                'data_identifier': data_identifier,
                'preprocessor_name': self.preprocessor_name,
                'batch_size': batch_size,
                'patch_size': patch_size,
                'median_image_size_in_voxels': median_shape,
                'spacing': spacing,
                'normalization_schemes': normalization_schemes,
                'use_mask_for_norm': mask_is_used_for_norm,
                'resampling_fn_data': resampling_data.__name__,
                'resampling_fn_seg': resampling_seg.__name__,
                'resampling_fn_data_kwargs': resampling_data_kwargs,
                'resampling_fn_seg_kwargs': resampling_seg_kwargs,
                'resampling_fn_probabilities': resampling_softmax.__name__,
                'resampling_fn_probabilities_kwargs': resampling_softmax_kwargs,
                'architecture': architecture_kwargs
            }
            return plan

        return super().get_plans_for_configuration(spacing, median_shape, data_identifier,
                                                    approximate_n_voxels_dataset, _cache)
