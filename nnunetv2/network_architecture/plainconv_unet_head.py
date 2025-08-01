import numpy as np
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import Union, Type, List, Tuple


class PlainConvUNetHead(PlainConvUNet):
    """PlainConvUNet with the last decoder block separated into `head`."""

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
                 nonlin_first: bool = False):
        super().__init__(input_channels, n_stages, features_per_stage, conv_op,
                         kernel_sizes, strides, n_conv_per_stage, num_classes,
                         n_conv_per_stage_decoder, conv_bias, norm_op,
                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                         nonlin_kwargs, deep_supervision, nonlin_first)

        # move last decoder stage to head
        self.head = nn.ModuleDict({
            'transpconv': self.decoder.transpconvs[-1],
            'stage': self.decoder.stages[-1],
            'seg_layer': self.decoder.seg_layers[-1]
        })
        self.decoder.transpconvs = self.decoder.transpconvs[:-1]
        self.decoder.stages = self.decoder.stages[:-1]
        self.decoder.seg_layers = self.decoder.seg_layers[:-1]

    def forward(self, x):
        skips = self.encoder(x)
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.decoder.stages)):
            y = self.decoder.transpconvs[s](lres_input)
            y = torch.cat((y, skips[-(s + 2)]), 1)
            y = self.decoder.stages[s](y)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](y))
            lres_input = y

        y = self.head['transpconv'](lres_input)
        y = torch.cat((y, skips[0]), 1)
        y = self.head['stage'](y)
        seg = self.head['seg_layer'](y)
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
            if self.decoder.deep_supervision:
                output += np.prod([self.decoder.num_classes, *skip_sizes[-(s + 1)]], dtype=np.int64)
        output += self.head['stage'].compute_conv_feature_map_size(skip_sizes[0])
        output += np.prod([self.encoder.output_channels[0], *skip_sizes[0]], dtype=np.int64)
        output += np.prod([self.head['seg_layer'].out_channels, *skip_sizes[0]], dtype=np.int64)
        return output
