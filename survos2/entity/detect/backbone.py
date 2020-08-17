import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn

from torch.utils import data
import torch.nn.functional as F

from torch import optim
from torch.optim import SGD, Adam, LBFGS
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import models
from torchvision.models import resnet152, resnet18, resnet34, resnet50
from torchvision.datasets import ImageFolder
from torchvision import transforms

import kornia

#layers = [conv]
#    layers += [nn.ReLU(inplace=True)]  # use inplace due to memory constraints
#    layers += [nn.BatchNorm3d(nf)] if norm == 'batch' else []
#    return nn.Sequential(*layers)

def convblock2(in_chan, out_chan, kernel, padding, stride=1, norm=None, relu='relu'):
    
    block = nn.Conv2d(in_chan, out_chan, kernel, padding=padding, stride=stride)
    
    if norm is not None:
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_chan)
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_chan)
        
        block = nn.Sequential(block, norm_layer)

    if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            block = nn.Sequential(block, relu_layer)
 
    return block



def convblock(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )



class EncDec(nn.Module):
    def __init__(self, n_class, convblock):
        super().__init__()

        self.resnet = resnet18(pretrained=True)  #50, 101
        self.convblock = convblock
        
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, 
                                                   padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)

        self.base_layers = list(self.resnet.children())

        self.C0 = nn.Sequential(*self.base_layers[:3])          # (N, 64, H/2, W/2)
        self.C1 = nn.Sequential(*self.base_layers[3:5])         # (N, 64, H/4, W/4)
        self.C2 = self.base_layers[5]                           # (N, 128, H/8, W/8)
        self.C3 = self.base_layers[6]                           # N, 256, H/16, W/16)
        self.C4 = self.base_layers[7]                           # (N, 512, H/32, W/32)
        
        self.C0_1x1 = self.convblock(64, 64, 1, 0)
        self.C1_1x1 = self.convblock(64, 64, 1, 0)
        self.C2_1x1 = self.convblock(128, 128, 1, 0)
        self.C3_1x1 = self.convblock(256, 256, 1, 0)
        self.C4_1x1 = self.convblock(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.upsample = nn.Interpolate(scale_factor=2, mode='bilinear')

        self.conv_up3 = self.convblock(256 + 512, 512, 3, 1)
        self.conv_up2 = self.convblock(128 + 512, 256, 3, 1)
        self.conv_up1 = self.convblock(64 + 256, 256, 3, 1)
        self.conv_up0 = self.convblock(64 + 256, 128, 3, 1)

        self.conv_original_size0 = self.convblock(3, 64, 3, 1)
        self.conv_original_size1 = self.convblock(64, 64, 3, 1)
        self.conv_original_size2 = self.convblock(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.avgpool = self.resnet.avgpool
        self.classifier = self.resnet.fc
        self.gradient = None
        
        self.n_locations = 10
        
        self.hm_conv = nn.Conv2d(16, self.n_locations, kernel_size=1, bias=False)

    def forward_(self, input):    

        # extract the features
        #x = self.features(x)
        
        # register the hook
        #h = x.register_hook(self.activations_hook)
        
        ## complete the forward pass
        #x = self.avgpool(x)
        #x = x.view((1, -1))
        #x = self.classifier(x)
        
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        C0 = self.C0(input)
        C1 = self.C1(C0)
        C2 = self.C2(C1)
        C3 = self.C3(C2)
        C4 = self.C4(C3)
        
        
        C4 = self.C4_1x1(C4)
        x = self.upsample(C4)

        C3 = self.C3_1x1(C3)
        x = torch.cat([x, C3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        C2 = self.C2_1x1(C2)
        x = torch.cat([x, C2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        C1 = self.C1_1x1(C1)
        x = torch.cat([x, C1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        C0 = self.C0_1x1(C0)
        x = torch.cat([x, C0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        out = self.conv_last(x)


        return out, x
    
    def forward(self, input):
        out,x = self.forward_(input)
        return out
    
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forwardgcam(self, x):
        x = self.features(x)
        
        h = x.register_hook(self.activations_hook)
        
        #  forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x
    
    def forward_hm(self, x):      
        out = self.forward_unet(x)
        
        # 2. 1x1 conv -> unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(out)
        
        heatmaps = kornia.dsnt.spatial_softmax_2d(unnormalized_heatmaps)
        #heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        
        #coords = dsntnn.dsnt(heatmaps)
        coords = kornia.geometry.dsnt.spatial_softargmax_2d(heatmaps)
    
        return coords, heatmaps
        

def setup_train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 1
    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, 
                                    model.parameters()), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.21)

    return model, optimizer_ft, exp_lr_scheduler




class ResNetUNet(nn.Module):
    def __init__(self, n_class, convblock):
        super().__init__()

        # define the resnet152
        self.resnet = resnet18(pretrained=True)
        self.convblock = convblock
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, 
                                                   padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)

        self.base_layers = list(self.resnet.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = self.convblock(64, 64, 1, 0)
        
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = self.convblock(64, 64, 1, 0)
        
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = self.convblock(128, 128, 1, 0)
        
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = self.convblock(256, 256, 1, 0)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = self.convblock(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = self.convblock(256 + 512, 512, 3, 1)
        self.conv_up2 = self.convblock(128 + 512, 256, 3, 1)
        self.conv_up1 = self.convblock(64 + 256, 256, 3, 1)
        self.conv_up0 = self.convblock(64 + 256, 128, 3, 1)

        self.conv_original_size0 = self.convblock(3, 64, 3, 1)
        self.conv_original_size1 = self.convblock(64, 64, 3, 1)
        self.conv_original_size2 = self.convblock(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.avgpool = self.resnet.avgpool
        self.classifier = self.resnet.fc
        self.gradient = None
        
        self.n_locations = 10
        
        self.hm_conv = nn.Conv2d(16, self.n_locations, kernel_size=1, bias=False)

    def forward_(self, input):    

        # extract the features
        #x = self.features(x)
        
        # register the hook
        #h = x.register_hook(self.activations_hook)
        
        ## complete the forward pass
        #x = self.avgpool(x)
        #x = x.view((1, -1))
        #x = self.classifier(x)
        
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)

        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        out = self.conv_last(x)

        return out, x
    
    def forward(self, input):
        out,x = self.forward_(input)
        return out
    
    def activations_hook(self, grad):
        self.gradient = grad
    
    def get_gradient(self):
        return self.gradient
    
    def get_activations(self, x):
        return self.features(x)
    
    def forwardgcam(self, x):
        x = self.features(x)
        
        h = x.register_hook(self.activations_hook)
        
        #  forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        
        return x
    
    def forward_hm(self, x):      
        out = self.forward_unet(x)
        
        # 2. 1x1 conv -> unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(out)
        
        heatmaps = kornia.dsnt.spatial_softmax_2d(unnormalized_heatmaps)
        #heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        
        #coords = dsntnn.dsnt(heatmaps)
        coords = kornia.geometry.dsnt.spatial_softargmax_2d(heatmaps)
    
        return coords, heatmaps
        

def setup_train():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 1
    model = ResNetUNet(num_class).to(device)

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, 
                                    model.parameters()), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.21)

    return model, optimizer_ft, exp_lr_scheduler


#def run_training():
#    model = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=27)
 #   return model

