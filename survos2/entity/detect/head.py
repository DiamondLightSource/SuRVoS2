
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.measurements import label as lb
import numpy as np
#import utils.exp_utils as utils
#import utils.model_utils as mutils


class Head_TwoStage_Cls(nn.Module):

    def __init__(self, conv, cf, n_input_channels, n_features, n_output_channels, n_classes, anchor_stride=1):
        super(Head_Cls, self).__init__()
        self.dim = conv.dim
        self.n_classes = n_classes


class Head_OneStage_Cls(nn.Module):
    def __init__(self, conv, n_input_channels, n_features, n_output_channels, n_classes, anchor_stride=1):
        super(Head_OneStage_Cls, self).__init__()
        self.dim = conv.dim
        self.n_classes = n_classes
        self.relu = 'relu' # 'leaky_relu'
        self.norm = 'batch_norm' #'instance_norm'

        print("n_output_channels: ",n_output_channels)

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=anchor_stride, pad=1, relu=self.relu, norm=self.norm)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=self.relu, norm=self.norm)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=self.relu, norm=self.norm)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=anchor_stride, pad=1, relu=self.relu, norm=self.norm)
        self.conv_final = conv(n_features, n_output_channels, ks=1, stride=anchor_stride, pad=1, relu=None)
        
        self.linear_bbox = nn.Linear(2 * (32 + 2)**3, self.n_classes)


    
    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        
        print(x.shape)
        
        class_logits_raw = self.conv_final(x)
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits_raw.permute(*axes)
        class_logits = class_logits.contiguous()        
        class_logits = class_logits.view(x.size()[0], -1,)
        
        linear_logits = self.linear_bbox(class_logits)

        return class_logits_raw, class_logits, linear_logits
