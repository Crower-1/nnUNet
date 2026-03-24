import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple, Optional
import copy


class PlainConvUNetHead(PlainConvUNet):
    """PlainConvUNet with shared decoder and lightweight per-class 1x1 heads."""

    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 class_names: Optional[List[str]] = None):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op,
                         kernel_sizes, strides, n_conv_per_stage, num_classes,
                         n_conv_per_stage_decoder, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                         nonlin_kwargs, deep_supervision, nonlin_first)

        last_seg_layer = copy.deepcopy(self.decoder.seg_layers[-1])

        # handle class names
        if class_names is None:
            class_names = [f'class_{i}' for i in range(num_classes)]
        self.class_names = class_names

        # build lightweight per-class heads, decoder remains shared and complete
        self.heads = nn.ModuleDict()
        for cn in self.class_names:
            self.heads[cn] = type(last_seg_layer)(
                last_seg_layer.in_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=last_seg_layer.bias is not None,
                padding_mode=getattr(last_seg_layer, 'padding_mode', 'zeros')
            )

    def _fused_head_logits(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.heads) == 0:
            raise RuntimeError("PlainConvUNetHead has no heads configured.")

        heads = list(self.heads.values())
        weight = torch.cat([h.weight for h in heads], dim=0)
        bias = None if heads[0].bias is None else torch.cat([h.bias for h in heads], dim=0)

        if features.ndim == 5:
            return F.conv3d(features, weight, bias, stride=1, padding=0)
        if features.ndim == 4:
            return F.conv2d(features, weight, bias, stride=1, padding=0)
        raise RuntimeError(f"Unsupported feature rank {features.ndim}, expected 4D/5D tensor.")

    def forward(self, x):
        skips = self.encoder(x)
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.decoder.stages)):
            y = self.decoder.transpconvs[s](lres_input)
            y = torch.cat((y, skips[-(s + 2)]), 1)
            y = self.decoder.stages[s](y)
            if self.decoder.deep_supervision and s < (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[s](y))
            lres_input = y

        seg = self._fused_head_logits(lres_input)
        seg_outputs.append(seg)
        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        assert len(skip_sizes) == len(self.decoder.stages) + 1

        output = np.int64(0)
        for s in range(len(self.decoder.stages)):
            output += self.decoder.stages[s].compute_conv_feature_map_size(skip_sizes[-(s + 1)])
            output += np.prod([self.encoder.output_channels[-(s + 2)], *skip_sizes[-(s + 1)]], dtype=np.int64)
            if self.decoder.deep_supervision and s < (len(self.decoder.stages) - 1):
                output += np.prod([self.decoder.seg_layers[s].out_channels, *skip_sizes[-(s + 1)]], dtype=np.int64)
        output += np.prod([len(self.heads), *skip_sizes[0]], dtype=np.int64)
        return output
