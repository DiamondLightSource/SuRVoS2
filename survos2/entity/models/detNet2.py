import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from survos2.entity.models.fpn import FPN, NDConvGenerator

from survos2.entity.models.head_cnn import (
    Head_TwoStage_Cls,
)


# Parts based on medicaldetectiontoolkit
# https://github.com/MIC-DKFZ/medicaldetectiontoolkit
#

class detNet(nn.Module):
    def __init__(self, cf):
        super(detNet, self).__init__()
        self.cf = cf
        self.logger = logger
        conv = NDConvGenerator(cf.dim)

        # set operate_stride1=True to generate a unet-like FPN.
        self.Fpn = FPN(cf, conv, operate_stride1=True)  # .cuda()
        self.Classifier = Head_TwoStage_Cls(conv, cf.end_filts, 128, 2, 1)
        self.conv_final = conv(
            cf.end_filts, 1, ks=1, pad=0, norm="batch_norm", relu=None
        )

    def forward_pyr(self, x):
        out_features = self.Fpn(x)
        return out_features

    # just seg
    def forward(self, x):
        out_features = self.Fpn(x)
        seg_logits = self.conv_final(out_features[0])
        return seg_logits

    def train_fwd(self, x):
        out_features = self.Fpn(x)[0]
        seg_logits = self.conv_final(out_features)
        smax = F.softmax(seg_logits, dim=1)
        return seg_logits, smax, out_features

    def train_fwd_seg(self, x):
        seg_logits = self.forward(x)
        smax = F.softmax(seg_logits, dim=1)
        return seg_logits, smax
