# v-net modified from https://github.com/Dootmaan/VNet.PyTorch

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.optim import lr_scheduler
from survos2.entity.utils import load_model
from survos2.entity.models.blocks import (
    ResBlock,
    ResBlock_x2,
    ResBlock_x1,
    DeConvBlock,
    DeConvBlock_x2,
    DeConvBlock_x1,
    NDConvGenerator,
    ConvSoftmax,
    ConvPool,
)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        conv = NDConvGenerator(3)
        self.conv_1 = ResBlock_x1(1, 16, conv=conv)
        self.pool_1 = ConvPool(16, 32)
        self.conv_2 = ResBlock_x2(32, 32, 32, conv=conv)
        self.pool_2 = ConvPool(32, 64)
        self.conv_3 = ResBlock(64, 64, 64, conv=conv)
        self.pool_3 = ConvPool(64, 128)
        self.conv_4 = ResBlock(128, 128, 128, conv=conv)
        self.pool_4 = ConvPool(128, 256)
        self.bottom = ResBlock(256, 256, 256, conv=conv)

        self.deconv_4 = DeConvBlock(256, 256)
        self.deconv_3 = DeConvBlock(256, 128)
        self.deconv_2 = DeConvBlock_x2(128, 64)
        self.deconv_1 = DeConvBlock_x1(64, 32)

        self.out = ConvSoftmax(32, 1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)
        bottom = self.bottom(pool)
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        return self.out(deconv)


def prepare_vnet(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    device = torch.device(device)
    model3d = VNet()

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler
