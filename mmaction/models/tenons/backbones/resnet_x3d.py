import logging
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from ....utils.misc import rgetattr, rhasattr
from .resnet import ResNet
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from ..spatial_temporal_modules.non_local import NonLocalModule
from ..utils.squeeze_and_excite import SE, Swish

from ...registry import BACKBONES

def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)


def conv3x1x1(in_planes, out_planes, spatial_stride=0, temporal_stride=1):
    "3x1x1 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3,1,1),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(1, 0, 0),
        dilation=1,
        bias=False)


def _round_width(width, multiplier, min_depth=8, divisor=8):
    """Round width of filters based on width multiplier."""
    if not multiplier:
        return width

    width *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(width + divisor / 2) // divisor * divisor
    )
    if new_filters < 0.9 * width:
        new_filters += divisor
    return int(new_filters)


def _round_repeats(repeats, multiplier):
    """Round number of layers based on depth multiplier."""
    multiplier = multiplier
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class X3DBottleneck(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 planes_inner,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 if_inflate=True,
                 use_se=True,
                 se_ratio=0.0625,
                 swish_inner=True,
                 style='pytorch',
                 with_cp=False):
        """X3DBottleneck block for X3D.
        """
        super(X3DBottleneck, self).__init__()
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride
        self.conv1 = nn.Conv3d(
            in_planes,
            planes_inner,
            kernel_size=(1,1,1),
            stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
            padding=(0,0,0),
            bias=False)
        self.conv2 = nn.Conv3d(
            planes_inner,
            planes_inner,
            kernel_size=(3 if if_inflate else 1,3,3),
            stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
            padding=(1 if if_inflate else 0, dilation, dilation),
            groups=planes_inner,
            dilation=(1, dilation, dilation),
            bias=False)
        if use_se and se_ratio > 0.0:
            self.se = SE(planes_inner, se_ratio)
        else:
            self.se = None
        if swish_inner:
            self.relu2 = Swish()
        else:
            self.relu2 = nn.ReLU(inplace=True)
            

        self.bn1 = nn.BatchNorm3d(planes_inner)
        self.bn2 = nn.BatchNorm3d(planes_inner)
        self.conv3 = nn.Conv3d(
            planes_inner, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)
        self.bn3.transform_final_bn = True
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            if self.se is not None:
                out = self.se(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   in_planes,
                   out_planes,
                   planes_inner,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   with_cp=False):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * blocks
    downsample = None
    if spatial_stride != 1 or in_planes != out_planes: 
        downsample = nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            nn.BatchNorm3d(out_planes),
        )

    layers = []
    layers.append(
        block(
            in_planes,
            out_planes,
            planes_inner,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            if_inflate= (inflate_freq[0] == 1),
            use_se=True,
            with_cp=with_cp))
    for i in range(1, blocks):
        layers.append(
            block(out_planes,
                out_planes,
                planes_inner, 
                1, 1,
                dilation,
                style=style,
                if_inflate= (inflate_freq[i] == 1),
                use_se=((i + 1) % 2 != 0), # apply se every other residual block
                with_cp=with_cp))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet_X3D(nn.Module):
    """ResNet_X3D backbone.

    Args:
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 width_factor=2,
                 depth_factor=2.2,
                 bottleneck_factor=2.25, 
                 base_dim_conv1=12,
                 dim_out=2048,
                 pretrained=None,
                 num_stages=4,
                 spatial_strides=(2, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_channel=3,
                 conv1_kernel_t=5,
                 frozen_stages=-1,
                 inflate_freq=(1, 1, 1, 1),    # For C2D baseline, this is set to -1.
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False):
        super(ResNet_X3D, self).__init__()
        self.pretrained = pretrained
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        if len(out_indices) > 0:
            assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * num_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        self.block = X3DBottleneck
        stage_blocks = [1, 2, 5, 3] 
        self.stage_blocks = stage_blocks[:num_stages]

        dim_out_conv1 = _round_width(base_dim_conv1, width_factor)

        self.conv1_xy = conv1x3x3(conv1_channel, dim_out_conv1, spatial_stride=2, temporal_stride=1, dilation=1)
        self.conv1_t = nn.Conv3d(dim_out_conv1, dim_out_conv1, kernel_size=(conv1_kernel_t, 1, 1),
                                 stride=(1, 1, 1), padding=(conv1_kernel_t // 2, 0, 0), groups=dim_out_conv1, bias=False)
        self.bn1 = nn.BatchNorm3d(dim_out_conv1)
        self.relu1 = nn.ReLU(inplace=True)

        inplanes = dim_out_conv1
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = _round_width(base_dim_conv1 * 2**i, width_factor) 
            planes_inner = int(bottleneck_factor * planes)
            res_layer = make_res_layer(
                self.block,
                inplanes,
                planes,
                planes_inner,
                _round_repeats(num_blocks, depth_factor),
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                inflate_freq=self.inflate_freqs[i],
                with_cp=with_cp)
            inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.conv5 = nn.Conv3d(planes, planes_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm3d(planes_inner)
        self.relu5 = nn.ReLU(inplace=True)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1_xy(x)
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.relu1(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(ResNet_X3D, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if self.partial_bn:
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                for m in mod.modules():
                    if isinstance(m, nn.BatchNorm3d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1_xy.parameters():
                param.requires_grad = False
            for param in self.conv1_t.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, 'layer{}'.format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
