import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

@HEADS.register_module
class X3DHead(nn.Module):
    """Simplest classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=432,
                 fc_channels=2048,
                 fc1_use_bn=False,
                 num_classes=101,
		 init_std=0.01,
                 fcn_testing=False):

        super(X3DHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.fc_channels = fc_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.fcn_testing = fcn_testing
        self.num_classes = num_classes

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.fc1 = nn.Conv3d(in_channels, fc_channels, kernel_size=1,
                             stride=1, padding=0, bias=False)
        if fc1_use_bn:
            self.fc1_bn = nn.BatchNorm3d(dim_out)
        else:
            self.fc1_bn = None
        self.fc1_relu = nn.ReLU(inplace=True) 
        
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size), stride=1)

        self.fc_cls = nn.Linear(fc_channels, num_classes)
        self.new_cls = None

    def init_weights(self):
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
         if x.ndimension() == 4:
             x = x.unsqueeze(2)
         if self.with_avg_pool:
             x = self.avg_pool(x)
         x = self.fc1(x)
         if self.fc1_bn is not None:
             x = self.fc1_bn(x)
         x = self.fc1_relu(x)
         if self.new_cls is None:
             self.new_cls = nn.Conv3d(self.fc_channels, self.num_classes, 1,1,0).cuda()
             self.new_cls.load_state_dict({'weight': self.fc_cls.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                           'bias': self.fc_cls.bias})
         class_map = self.new_cls(x)
         class_map = class_map.mean([2,3,4])
         return class_map

    def loss(self,
             cls_score,
             labels):
        losses = dict()
        losses['loss_cls'] = F.cross_entropy(cls_score, labels)

        return losses
