import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#
# FPN code and some blocks modified from medicaldetectiontoolkit
# https://github.com/MIC-DKFZ/medicaldetectiontoolkit
# V-net related code modified from https://github.com/Dootmaan/VNet.PyTorch


class NDConvGenerator(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu="relu"):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == "instance_norm":
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == "batch_norm":
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError("norm type as specified in configs is not implemented...")
                conv = nn.Sequential(conv, norm_layer)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == "instance_norm":
                    norm_layer = nn.InstanceNorm3d(c_out, affine=True)
                elif norm == "batch_norm":
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError(
                        "norm type as specified in configs is not implemented... {}".format(norm)
                    )
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == "relu":
                relu_layer = nn.ReLU(inplace=True)
            elif relu == "leaky_relu":
                relu_layer = nn.LeakyReLU(inplace=True)
            elif relu == "PReLU":
                relu_layer = nn.PReLU()
            else:
                raise ValueError("relu type as specified in configs is not implemented...")
            conv = nn.Sequential(conv, relu_layer)

        return conv


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class ResBlockFPN(nn.Module):
    def __init__(
        self,
        start_filts,
        planes,
        end_filts,
        conv,
        stride=1,
        identity_skip=True,
        norm=None,
        relu="PReLU",
    ):
        """Builds a residual net block with three conv-layers.
        :param start_filts: #input channels to the block.
        :param planes: #channels in block's hidden layers. set start_filts>planes<end_filts for bottlenecking.
        :param end_filts: #output channels of the block.
        :param conv: conv-layer generator.
        :param stride:
        :param identity_skip: whether to use weight-less identity on skip-connection if no rescaling necessary.
        :param norm:
        :param relu:
        """
        super(ResBlock, self).__init__()

        self.conv1 = conv(start_filts, planes, ks=1, pad=2, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, end_filts, ks=1, norm=norm, relu=None)
        if relu == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu == "leaky_relu":
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == "PReLU":
            self.relu = nn.PReLU()
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride != 1 or start_filts != end_filts or not identity_skip:
            self.scale_residual = conv(
                start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None
            )
        else:
            self.scale_residual = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.scale_residual:
            residual = self.scale_residual(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        start_filts,
        planes,
        end_filts,
        conv,
        stride=1,
        identity_skip=True,
        norm="instance_norm",
        relu="PReLU",
    ):
        """Builds a residual net block with three conv-layers.
        :param start_filts: #input channels to the block.
        :param planes: #channels in block's hidden layers. set start_filts>planes<end_filts for bottlenecking.
        :param end_filts: #output channels of the block.
        :param conv: conv-layer generator.
        :param stride:
        :param identity_skip: whether to use weight-less identity on skip-connection if no rescaling necessary.
        :param norm:
        :param relu:
        """
        super(ResBlock, self).__init__()

        self.conv1 = conv(start_filts, planes, ks=5, pad=2, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=5, pad=2, norm=norm, relu=relu)
        self.conv3 = conv(planes, end_filts, ks=5, pad=2, norm=norm, relu=None)
        if relu == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu == "leaky_relu":
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == "PReLU":
            self.relu = nn.PReLU()
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride != 1 or start_filts != end_filts or not identity_skip:
            self.scale_residual = conv(
                start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None
            )
        else:
            self.scale_residual = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.scale_residual:
            residual = self.scale_residual(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class ResBlock_x2(nn.Module):
    def __init__(
        self,
        start_filts,
        planes,
        end_filts,
        conv,
        stride=1,
        identity_skip=True,
        norm="instance_norm",
        relu="PReLU",
    ):
        """Builds a residual net block with two conv layers.
        :param start_filts: #input channels to the block.
        :param planes: #channels in block's hidden layers. set start_filts>planes<end_filts for bottlenecking.
        :param end_filts: #output channels of the block.
        :param conv: conv-layer generator.
        :param stride:
        :param identity_skip: whether to use weight-less identity on skip-connection if no rescaling necessary.
        :param norm:
        :param relu:
        """
        super(ResBlock_x2, self).__init__()

        self.conv1 = conv(start_filts, planes, ks=5, pad=2, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, end_filts, ks=5, pad=2, norm=norm, relu=relu)

        if relu == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu == "leaky_relu":
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == "PReLU":
            self.relu = nn.PReLU()
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride != 1 or start_filts != end_filts or not identity_skip:
            self.scale_residual = conv(
                start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None
            )
        else:
            self.scale_residual = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.scale_residual:
            residual = self.scale_residual(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class ResBlock_x1(nn.Module):
    def __init__(
        self,
        start_filts,
        end_filts,
        conv,
        stride=1,
        identity_skip=True,
        norm="instance_norm",
        relu="PReLU",
    ):
        """Builds a residual net block with one conv layer.
        :param start_filts: #input channels to the block.
        :param end_filts: #output channels of the block.
        :param conv: conv-layer generator.
        :param stride:
        :param identity_skip: whether to use weight-less identity on skip-connection if no rescaling necessary.
        :param norm:
        :param relu:
        """
        super(ResBlock_x1, self).__init__()

        self.conv1 = conv(start_filts, end_filts, ks=5, pad=2, stride=stride, norm=norm, relu=relu)

        if relu == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu == "leaky_relu":
            self.relu = nn.LeakyReLU(inplace=True)
        elif relu == "PReLU":
            self.relu = nn.PReLU()
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride != 1 or start_filts != end_filts or not identity_skip:
            self.scale_residual = conv(
                start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None
            )
        else:
            self.scale_residual = None

    def forward(self, x):
        out = self.conv1(x)
        if self.scale_residual:
            residual = self.scale_residual(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


def UpsampleDeconv(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride), nn.PReLU()
    )


def ConvPool(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0), nn.PReLU()
    )


class ConvSoftmax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSoftmax, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Output shape [batch_size, 1, depth, height, width]."""
        y_conv = self.conv_2(self.conv_1(x))  # Don't normalize output
        return nn.Sigmoid()(y_conv)


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu="PReLU", norm="instance_norm"):
        super(DeConvBlock, self).__init__()
        self.upsample = UpsampleDeconv(in_channels, out_channels)
        conv = NDConvGenerator(3)
        self.lhs_conv = conv(
            out_channels // 2, out_channels, ks=1, stride=stride, relu=relu, norm=norm
        )
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.upsample(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_cat = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x3(rhs_cat) + rhs_up


class DeConvBlock_x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu="PReLU", norm="instance_norm"):
        super(DeConvBlock_x2, self).__init__()
        self.upsample = UpsampleDeconv(in_channels, out_channels)
        conv = NDConvGenerator(3)
        self.lhs_conv = conv(
            out_channels // 2, out_channels, ks=1, stride=stride, relu=relu, norm=norm
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.upsample(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_cat = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x2(rhs_cat) + rhs_up


class DeConvBlock_x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, relu="PReLU", norm="instance_norm"):
        super(DeConvBlock_x1, self).__init__()
        self.upsample = UpsampleDeconv(in_channels, out_channels)
        conv = NDConvGenerator(3)
        self.lhs_conv = conv(
            out_channels // 2, out_channels, ks=1, stride=stride, relu=relu, norm=norm
        )
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2 * out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.upsample(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_cat = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x1(rhs_cat) + rhs_up
