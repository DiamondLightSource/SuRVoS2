


import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import autograd, optim
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import os
from collections import namedtuple

import sys
import os
from collections import namedtuple

#sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
#sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../..")
#from default_configs import DefaultConfigs


class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
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
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)

        return conv

    
class ConvGenerator():
    """conv-layer generator to avoid 2D vs. 3D distinction in code.
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """provides generic conv-layer modules for set dimension.
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: 2D or 3D conv-layer module.
        """

        if self.dim == 2:
            module = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                module = nn.Sequential(module, norm_layer)

        elif self.dim==3:
            module = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                module = nn.Sequential(module, norm_layer)
        else:
            raise Exception("Invalid dimension {} in conv-layer generation.".format(self.dim))

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            module = nn.Sequential(module, relu_layer)

        return module


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, end_filts, conv, stride=1, identity_skip=True, norm=None, relu='relu'):
        """Builds a residual net block.
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

        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, end_filts, ks=1, norm=norm, relu=None)
        if relu == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif relu == 'leaky_relu':
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            raise Exception("Chosen activation {} not implemented.".format(self.relu))

        if stride!=1 or start_filts!=end_filts or not identity_skip:
            self.scale_residual = conv(start_filts, end_filts, ks=1, stride=stride, norm=norm, relu=None)
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


class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, relu_enc="relu", relu_dec=None, operate_stride1=False):
        """
        :param conv: instance of custom conv class containing the dimension info.
        :param relu_enc: string specifying type of nonlinearity in encoder. If None, no nonlinearity is applied.
	    :param relu_dec: same as relu_enc but for decoder.
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        from configs:
	        :param channels: len(channels) is nr of channel dimensions in input data.
	        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
	        :param end_filts: number of feature_maps for output_layers of all levels in decoder.
	        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
	        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
	        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.start_filts, sf = cf.start_filts, cf.start_filts #sf = alias for readability
        self.out_channels = cf.end_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_exp = 4 #factor by which to increase nr of channels in first block layer.
        self.relu_enc = relu_enc
        self.relu_dec = relu_dec
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling


        if operate_stride1:
            self.C0 = nn.Sequential(conv(len(cf.channels), sf, ks=3, pad=1, norm=cf.norm, relu=relu_enc),
                                    conv(sf, sf, ks=3, pad=1, norm=cf.norm, relu=relu_enc))

            self.C1 = conv(sf, sf, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm,
                           relu=relu_enc)

        else:
            self.C1 = conv(len(cf.channels), sf, ks=7, stride=(2, 2, 1) if conv.dim == 3 else 2, pad=3, norm=cf.norm,
                           relu=relu_enc)

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if
                         conv.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(sf, sf, sf*self.block_exp, conv=conv, stride=1, norm=cf.norm,
                                    relu=relu_enc))
        
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(sf*self.block_exp, sf, sf*self.block_exp, conv=conv,
                                        stride=1, norm=cf.norm, relu=relu_enc))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_layers.append(self.block(sf*self.block_exp, sf*2, sf*self.block_exp*2, conv=conv,
                                    stride=2, norm=cf.norm, relu=relu_enc))
        
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(sf*self.block_exp*2, sf*2, sf*self.block_exp*2,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_layers.append(self.block(sf*self.block_exp*2, sf*4, sf*self.block_exp*4,
                                    conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(sf*self.block_exp*4, sf*4, sf*self.block_exp*4,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_layers.append(self.block(sf*self.block_exp*4, sf*8, sf*self.block_exp*8,
                                    conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(sf*self.block_exp*8, sf*8, sf*self.block_exp*8,
                                        conv=conv, norm=cf.norm, relu=relu_enc))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(sf*self.block_exp*8, sf*16, sf*self.block_exp*16,
                                        conv=conv, stride=2, norm=cf.norm, relu=relu_enc))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(sf*self.block_exp*16, sf*16, sf*self.block_exp*16,
                                            conv=conv, norm=cf.norm, relu=relu_enc))
            self.C6 = nn.Sequential(*C6_layers)

        if conv.dim == 2:
            self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
            self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        else:
            self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        if self.sixth_pooling:
            self.P6_conv1 = conv(sf*self.block_exp*16, self.out_channels, ks=1, stride=1, relu=relu_dec)
        
        self.P5_conv1 = conv(sf*self.block_exp*8, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P4_conv1 = conv(sf*self.block_exp*4, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P3_conv1 = conv(sf*self.block_exp*2, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P2_conv1 = conv(sf*self.block_exp, self.out_channels, ks=1, stride=1, relu=relu_dec)
        self.P1_conv1 = conv(sf, self.out_channels, ks=1, stride=1, relu=relu_dec)

        if operate_stride1:
            self.P0_conv1 = conv(sf, self.out_channels, ks=1, stride=1, relu=relu_dec)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)

        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)

        if self.sixth_pooling:
            self.P6_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)

        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)
        #pre_out means last step before prediction output
        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        # plot feature map shapes for debugging.
        # for ii in [c0_out, c1_out, c2_out, c3_out, c4_out, c5_out, c6_out]:
        #     print ("encoder shapes:", ii.shape)
        #
        # for ii in [p6_out, p5_out, p4_out, p3_out, p2_out, p1_out]:
        #     print("decoder shapes:", ii.shape)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list



# legends, nested classes are not handled well in multiprocessing! hence, Label class def in outer scope
Label = namedtuple("Label", ['id', 'name', 'color', 'm_scores']) # m_scores = malignancy scores

binLabel = namedtuple("binLabel", ['id', 'name', 'color', 'm_scores', 'bin_vals'])

boxLabel = namedtuple('boxLabel', ["name", "color"])

class DefaultConfigs:

    def __init__(self, server_env=None, dim=2):
        self.server_env = server_env
        self.cuda_benchmark = True
        self.sysmetrics_interval = 2 # set > 0 to record system metrics to tboard with this time span in seconds.
        #########################
        #         I/O           #
        #########################

        self.dim = dim
        # int [0 < dataset_size]. select n patients from dataset for prototyping.
        self.select_prototype_subset = None

        # some default paths.
        self.source_dir = "./"# os.path.dirname(os.path.realpath(__file__)) # current dir.
        self.backbone_path = os.path.join(self.source_dir, 'models/backbone.py')
        self.input_df_name = 'info_df.pickle'


        if server_env:
            self.select_prototype_subset = None

        #########################
        #      Colors/legends   #
        #########################

        # in part from solarized theme.
        self.black = (0.1, 0.05, 0.)
        self.gray = (0.514, 0.580, 0.588)
        self.beige = (1., 1., 0.85)
        self.white = (0.992, 0.965, 0.890)

        self.green = (0.659, 0.792, 0.251)  # [168, 202, 64]
        self.dark_green = (0.522, 0.600, 0.000) # [133.11, 153.  ,   0.  ]
        self.cyan = (0.165, 0.631, 0.596)  # [ 42.075, 160.905, 151.98 ]
        self.bright_blue = (0.85, 0.95, 1.)
        self.blue = (0.149, 0.545, 0.824) # [ 37.995, 138.975, 210.12 ]
        self.dkfz_blue = (0, 75. / 255, 142. / 255)
        self.dark_blue = (0.027, 0.212, 0.259) # [ 6.885, 54.06 , 66.045]
        self.purple = (0.424, 0.443, 0.769) # [108.12 , 112.965, 196.095]
        self.aubergine = (0.62, 0.21, 0.44)  # [ 157,  53 ,  111]
        self.magenta = (0.827, 0.212, 0.510) # [210.885,  54.06 , 130.05 ]
        self.coral = (1., 0.251, 0.4) # [255,64,102]
        self.bright_red = (1., 0.15, 0.1)  # [255, 38.25, 25.5]
        self.brighter_red = (0.863, 0.196, 0.184) # [220.065,  49.98 ,  46.92 ]
        self.red = (0.87, 0.05, 0.01)  # [ 223, 13, 2]
        self.dark_red = (0.6, 0.04, 0.005)
        self.orange = (0.91, 0.33, 0.125)  # [ 232.05 ,   84.15 ,   31.875]
        self.dark_orange = (0.796, 0.294, 0.086) #[202.98,  74.97,  21.93]
        self.yellow = (0.95, 0.9, 0.02)  # [ 242.25,  229.5 ,    5.1 ]
        self.dark_yellow = (0.710, 0.537, 0.000) # [181.05 , 136.935,   0.   ]


        self.color_palette = [self.blue, self.dark_blue, self.aubergine, self.green, self.yellow, self.orange, self.red,
                              self.cyan, self.black]

        self.box_labels = [
            #           name            color
            boxLabel("det", self.blue),
            boxLabel("prop", self.gray),
            boxLabel("pos_anchor", self.cyan),
            boxLabel("neg_anchor", self.cyan),
            boxLabel("neg_class", self.green),
            boxLabel("pos_class", self.aubergine),
            boxLabel("gt", self.red)
        ]  # neg and pos in a medical sense, i.e., pos=positive diagnostic finding

        self.box_type2label = {label.name: label for label in self.box_labels}
        self.box_color_palette = {label.name: label.color for label in self.box_labels}

        # whether the input data is mono-channel or RGB/rgb
        self.has_colorchannels = False

        #########################
        #      Data Loader      #
        #########################

        #random seed for fold_generator and batch_generator.
        self.seed = 0

        #number of threads for multithreaded tasks like batch generation, wcs, merge2dto3d
        self.n_workers = 16 if server_env else os.cpu_count()

        self.create_bounding_box_targets = True
        self.class_specific_seg = True  # False if self.model=="mrcnn" else True
        self.max_val_patients = "all"
        #########################
        #      Architecture      #
        #########################

        self.prediction_tasks = ["class"]  # 'class', 'regression_class', 'regression_kendall', 'regression_feindt'

        self.weight_decay = 0.0

        # nonlinearity to be applied after convs with nonlinearity. one of 'relu' or 'leaky_relu'
        self.relu = 'relu'

        # if True initializes weights as specified in model script. else use default Pytorch init.
        self.weight_init = None

        # if True adds high-res decoder levels to feature pyramid: P1 + P0. (e.g. set to true in retina_unet configs)
        self.operate_stride1 = False

        #########################
        #  Optimization         #
        #########################

        self.optimizer = "ADAMW" # "ADAMW" or "SGD" or implemented additionals

        #########################
        #  Schedule             #
        #########################

        # number of folds in cross validation.
        self.n_cv_splits = 5

        #########################
        #   Testing / Plotting  #
        #########################

        # perform mirroring at test time. (only XY. Z not done to not blow up predictions times).
        self.test_aug = True

        # if True, test data lies in a separate folder and is not part of the cross validation.
        self.hold_out_test_set = False
        # if hold-out test set: if ensemble_folds is True, predictions of all folds on the common hold-out test set
        # are aggregated (like ensemble members). if False, each fold's parameters are evaluated separately on the test
        # set and the evaluations are aggregated (like normal cross-validation folds).
        self.ensemble_folds = False

        # if hold_out_test_set provided, ensemble predictions over models of all trained cv-folds.
        self.ensemble_folds = False

        # what metrics to evaluate
        self.metrics = ['ap']
        # whether to evaluate fold means when evaluating over more than one fold
        self.evaluate_fold_means = False

        # how often (in nr of epochs) to plot example batches during train/val
        self.plot_frequency = 1

        # color specifications for all box_types in prediction_plot.
        self.box_color_palette = {'det': 'b', 'gt': 'r', 'neg_class': 'purple',
                                  'prop': 'w', 'pos_class': 'g', 'pos_anchor': 'c', 'neg_anchor': 'c'}

        # scan over confidence score in evaluation to optimize it on the validation set.
        self.scan_det_thresh = False

        # plots roc-curves / prc-curves in evaluation.
        self.plot_stat_curves = False

        # if True: evaluate average precision per patient id and average over per-pid results,
        #     instead of computing one ap over whole data set.
        self.per_patient_ap = False

        # threshold for clustering 2D box predictions to 3D Cubes. Overlap is computed in XY.
        self.merge_3D_iou = 0.1

        # number or "all" for all
        self.max_test_patients = "all"

        #########################
        #   MRCNN               #
        #########################

        # if True, mask loss is not applied. used for data sets, where no pixel-wise annotations are provided.
        self.frcnn_mode = False

        self.return_masks_in_train = False
        # if True, unmolds masks in Mask R-CNN to full-res for plotting/monitoring.
        self.return_masks_in_val = False
        self.return_masks_in_test = False # needed if doing instance segmentation. evaluation not yet implemented.

        # add P6 to Feature Pyramid Network.
        self.sixth_pooling = False


        #########################
        #   RetinaNet           #
        #########################
        self.focal_loss = False
        self.focal_loss_gamma = 2.


    
class Configs(DefaultConfigs):

    def __init__(self, server_env=None):
        super(Configs, self).__init__(server_env)

        # dimension the model operates in. one out of [2, 3].
        self.dim = 2

        # 'class': standard object classification per roi, pairwise combinable with each of below tasks.
        # if 'class' is omitted from tasks, object classes will be fg/bg (1/0) from RPN.
        # 'regression': regress some vector per each roi
        # 'regression_ken_gal': use kendall-gal uncertainty sigma
        # 'regression_bin': classify each roi into a bin related to a regression scale
        self.prediction_tasks = ['class']

        self.start_filts = 48 if self.dim == 2 else 18
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' , 'resnet50'
        self.norm = "instance_norm" # one of None, 'instance_norm', 'batch_norm'

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None

        self.regression_n_features = 1
        
        #########################
        #      Data Loader      #
        #########################

        # distorted gt experiments: train on single-annotator gts in a random fashion to investigate network's
        # handling of noisy gts.
        # choose 'merged' for single, merged gt per image, or 'single_annotator' for four gts per image.
        # validation is always performed on same gt kind as training, testing always on merged gt.
        self.training_gts = "merged"

        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size_2D = [320, 320]
        self.patch_size_2D = [320, 320]
        self.pre_crop_size_3D = [160, 160, 96]
        self.patch_size_3D = [160, 160, 96]

        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_2D if self.dim == 2 else self.pre_crop_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        self.batch_random_ratio = 0.1
        self.balance_target =  "class_targets" if 'class' in self.prediction_tasks else 'rg_bin_targets'

        # set 2D network to match 3D gt boxes.
        self.merge_2D_to_3D_preds = self.dim==2

        self.observables_rois = []

        #self.rg_map = {1:1, 2:2, 3:3, 4:4, 5:5}

       
        #########################
        #   Colors and Legends  #
        #########################
        self.plot_frequency = 5

    
        oneclass_cl_labels = []
        
        twoclass_cl_labels = [ Label(1, 'thing',  (*self.dark_green, 1.),  (1, 2)), ]
        
        binary_cl_labels = [Label(1, 'benign',  (*self.dark_green, 1.),  (1, 2)),
                            Label(2, 'malignant', (*self.red, 1.),  (3, 4, 5))]
        quintuple_cl_labels = [Label(1, 'MS1',  (*self.dark_green, 1.),      (1,)),
                               Label(2, 'MS2',  (*self.dark_yellow, 1.),     (2,)),
                               Label(3, 'MS3',  (*self.orange, 1.),     (3,)),
                               Label(4, 'MS4',  (*self.bright_red, 1.), (4,)),
                               Label(5, 'MS5',  (*self.red, 1.),        (5,))]
        # choose here if to do 2-way or 5-way regression-bin classification
        task_spec_cl_labels = twoclass_cl_labels #binary_cl_labels #quintuple_cl_labels

        self.class_labels = [
            #       #id #name     #color              #malignancy score
            Label(  0,  'bg',     (*self.gray, 0.),  (0,))]
        if "class" in self.prediction_tasks:
            self.class_labels += task_spec_cl_labels

        else:
            self.class_labels += [Label(1, 'lesion', (*self.orange, 1.), (1,2,3,4,5))]

        if any(['regression' in task for task in self.prediction_tasks]):
            self.bin_labels = [binLabel(0, 'MS0', (*self.gray, 1.), (0,), (0,))]
            self.bin_labels += [binLabel(cll.id, cll.name, cll.color, cll.m_scores,
                                         tuple([ms for ms in cll.m_scores])) for cll in task_spec_cl_labels]
            self.bin_id2label = {label.id: label for label in self.bin_labels}
            self.ms2bin_label = {ms: label for label in self.bin_labels for ms in label.m_scores}
            bins = [(min(label.bin_vals), max(label.bin_vals)) for label in self.bin_labels]
            self.bin_id2rg_val = {ix: [np.mean(bin)] for ix, bin in enumerate(bins)}
            self.bin_edges = [(bins[i][1] + bins[i + 1][0]) / 2 for i in range(len(bins) - 1)]

        if self.class_specific_seg:
            self.seg_labels = self.class_labels
        else:
            self.seg_labels = [  # id      #name           #color
                Label(0, 'bg', (*self.gray, 0.)),
                #Label(1, 'fg', (*self.orange, 1.))
            ]

        self.class_id2label = {label.id: label for label in self.class_labels}
        self.class_dict = {label.id: label.name for label in self.class_labels if label.id != 0}
        # class_dict is used in evaluator / ap, auc, etc. statistics, and class 0 (bg) only needs to be
        # evaluated in debugging
        self.class_cmap = {label.id: label.color for label in self.class_labels}

        self.seg_id2label = {label.id: label for label in self.seg_labels}
        self.cmap = {label.id: label.color for label in self.seg_labels}

        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        self.has_colorchannels = False
        self.plot_class_ids = True

        self.num_classes = len(self.class_dict)  # for instance classification (excl background)
        self.num_seg_classes = len(self.seg_labels)  # incl background




