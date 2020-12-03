import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, normal_init
from mmcv.runner import load_checkpoint

from ...registry import BACKBONES


__all__ = ['JITNet']


class BottleNeck(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 stride=2):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv3d(ch_in, ch_out, (1, 3, 3), padding=(0, 1, 1), stride=(1, stride, stride))
        self.bn1 = nn.BatchNorm3d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2h = nn.Conv3d(ch_out, ch_out, (1, 1, 3), padding=(0, 0, 1))
        self.bn2h = nn.BatchNorm3d(ch_out)
        self.conv2w = nn.Conv3d(ch_out, ch_out, (1, 3, 1), padding=(0, 1, 0))
        self.bn2w = nn.BatchNorm3d(ch_out)
        '''
        self.conv1 = nn.Conv3d(ch_in, ch_out // 4, (1, 1, 1), padding=(0, 0, 0), stride=(1, stride, stride))
        self.bn1 = nn.BatchNorm3d(ch_out // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2h = nn.Conv3d(ch_out // 4, ch_out // 4, (1, 1, 3), padding=(0, 0, 1))
        self.bn2h = nn.BatchNorm3d(ch_out // 4)
        self.conv2w = nn.Conv3d(ch_out // 4, ch_out // 4, (1, 3, 1), padding=(0, 1, 0))
        self.bn2w = nn.BatchNorm3d(ch_out // 4)
        self.conv3 = nn.Conv3d(ch_out // 4, ch_out, (1, 1, 1), padding=(0, 0, 0))
        self.bn3 = nn.BatchNorm3d(ch_out)
        '''

        if stride != 1 or ch_in != ch_out:
            self.downsample = nn.Conv3d(ch_in, ch_out, 1, stride=(1, stride, stride))

    def forward(self, x):
        identity = x

        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2h(self.conv2h(y)))
        y = self.bn2w(self.conv2w(y))
        # y = self.relu(self.bn2w(self.conv2w(y)))
        # y = self.bn3(self.conv3(y))

        if self.downsample is not None:
            identity = self.downsample(x)

        y += identity
        y = self.relu(y)

        return y


@BACKBONES.register_module
class JITNet(nn.Module):

    def __init__(self,
                 pretrained=None,
                 modality='RGB',
                 base_ch=8):
        super(JITNet, self).__init__()

        self.pretrained = pretrained
        self.modality = modality

        inplace = True
        assert modality in ['RGB']
        self.conv1 = nn.Conv3d(3, base_ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(base_ch)
        self.relu1 = nn.ReLU(inplace)
        self.conv2 = BottleNeck(base_ch, base_ch*4, stride=2)
        self.conv3 = BottleNeck(base_ch * 4, base_ch * 8, stride=2)
        self.conv4 = BottleNeck(base_ch * 8, base_ch * 16, stride=2)
        self.conv5 = BottleNeck(base_ch * 16, base_ch * 32, stride=2)


    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    normal_init(m, std=0.01, bias=1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.005, bias=1)


    def forward(self, input):
        conv1 = self.conv1(input)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)

        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        return conv5 

    def train(self, mode=True):
        super(JITNet, self).train(mode)
