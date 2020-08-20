"""
Filtering and survos feature generation

Uses external libraries:
    dask
    dask_image
    skimage, scipy.ndimage
    kornia
    pyradiomics
    geodesic
"""
import os
import sys

import math
import numbers
import tqdm
import numpy as np
import time
from PIL import Image
from typing import Tuple

from collections import namedtuple
from functools import partial

from typing import List
from dataclasses import dataclass

import warnings
import numpy as np
import pandas as pd
#import pytest
import sklearn.datasets
import sklearn.utils.extmath

import dask_image
from dask_image import ndfilters

import dask.array as da
import dask.delayed
import dask.dataframe as dd
from dask.array.utils import assert_eq
from dask.distributed import Client

from sklearn.cluster import KMeans as SKKMeans
from sklearn.utils.estimator_checks import check_estimator

import skimage
from skimage.filters import gaussian
from skimage import img_as_float
from skimage.segmentation import flood_fill, flood
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import gaussian

from scipy import ndimage
import scipy
#import SimpleITK as sitk

import torch
from torch import nn
from torch.nn import functional as F
import kornia

from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.improc.segmentation.mappings import normalize
from survos2.utils import logger

from survos2.server.model import Features
from survos2.frontend.nb_utils import show_images
    

def prepare_prediction_features(filtered_layers):
    # reshaping for survos
    dataset_feats_reshaped = [f.reshape(1, filtered_layers[0].shape[0],
                                           filtered_layers[0].shape[1],
                                           filtered_layers[0].shape[2]) for f in filtered_layers]

    dataset_feats = np.vstack(dataset_feats_reshaped).astype(np.float32)

    features_stack = []

    for i, feature in enumerate(dataset_feats):
        features_stack.append(feature[...].ravel())

    features_stack = np.stack(features_stack, axis=1).astype(np.float32)

    return dataset_feats, features_stack


def prepare_features(features, roi_crop, resample_amt):
    """Calculate filters on image volume to generate features for survos segmentation
    
    Arguments:
        features {list of string} -- list of feature uri
        roi_crop {tuple of int} -- tuple defining a bounding box for cropping the image volume
        resample_amt {float} -- amount to scale the input volume
    
    Returns:
        features -- dataclass containing the processed image layers, and a stack made from them 
    """
    #features_stack = []
    filtered_layers = []

    for i, feature in enumerate(features):

        logger.info(f"Loading feature number {i}: {os.path.basename(feature)}")

        data = dataset_from_uri(feature, mode='r')
        data = data[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3], roi_crop[4]:roi_crop[5]]
        data = scipy.ndimage.zoom(data, resample_amt, order=1)

        logger.info(f"Cropped and resampled feature shape: {data.shape}")
        filtered_layers.append(data)

        #features_stack.append(data[...].ravel())

    #features_stack = np.stack(features_stack, axis=1)

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    features = Features(filtered_layers, dataset_feats, features_stack)

    return features

def feature_factory(filtered_layers):
    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    features = Features(filtered_layers, dataset_feats, features_stack)

    return features


def generate_features(img_vol, feature_params, roi_crop, resample_amt):
    def proc_layer(layer):
        layer_crop = layer[roi_crop[0]: roi_crop[1], 
                           roi_crop[2]: roi_crop[3], 
                           roi_crop[4]: roi_crop[5]].astype(np.float32, copy=False)
        layer_proc = (scipy.ndimage.zoom(layer_crop, resample_amt, order=1))
        layer_proc = normalize(layer_proc, norm='unit')
        return layer_proc

    logger.info(f"From img vol of shape: {img_vol.shape}")
    logger.info(f"Generating features with params: {feature_params}")

    # map_blocks through Dask
    filtered_layers = ([proc_layer(map_blocks(filter_fn, img_vol, **params_dict))
                        for filter_fn, params_dict in feature_params])


    filtered_layers = np.array(filtered_layers).astype(np.float32)

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    logger.info(f"Shape of feature data: {dataset_feats.shape}")

    features = Features(filtered_layers, dataset_feats, features_stack)

    return features


# todo replace with two calls to one function
def crop_and_resample(dataset_in, layers, roi_crop, resample_amt):

    dataset_proc = dataset_in[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3],
                   roi_crop[4]:roi_crop[5]].copy()

    logger.info(f"Prepared region: {dataset_in.shape}")
    dataset_proc = scipy.ndimage.zoom(dataset_proc, resample_amt, order=1)
    logger.info(f"Cropped and resized volume shape: {dataset_proc.shape}")

    layers_proc = []

    for layer in layers:
        # Annotation
        layer_proc = layer[roi_crop[0]:roi_crop[1], roi_crop[2]:roi_crop[3], roi_crop[4]:roi_crop[5]]
        layer_proc = scipy.ndimage.zoom(layer_proc, resample_amt, order=1)
        logger.info(f"Cropped and resized layer with shape: {layer_proc.shape}")
        layers_proc.append(layer_proc)

    return dataset_proc, layers_proc

def select_region(imvol : np.ndarray) -> Tuple[float, float, float, float, float, float]:
    vol_shape_z = imvol.shape[0]
    vol_shape_x = imvol.shape[1]
    vol_shape_y = imvol.shape[2]

    zstart, zend = 0, vol_shape_z
    xstart, xend = 0, vol_shape_x
    ystart, yend = 0, vol_shape_y
    return (zstart, zend, xstart, xend, ystart, yend)


def simple_invert(data, sigma=5.0):
    return 1.0 - data


def simple_laplacian(I, sigma=5.0):
    locNaNs = np.isnan(I)    
    I = np.nan_to_num(I)
    I = ndimage.laplace(gaussian(I, sigma))    
    I = I / np.max(I)
    return I

 
#
# Kornia features
#

def spatial_gradient_3d(vol_gray: np.ndarray, sigma=1.0) -> np.ndarray:
    """Spatial gradient of a array of intensity values
    
    Arguments:
        vol_gray {np.ndarray} -- input array
    Returns:
        np.ndarray -- filtered array
    """
    img_gray = img_as_float(np.clip(vol_gray, 0.0, 1.0))

    t_gray = kornia.utils.image_to_tensor(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0)
    spatialgradient3d = kornia.filters.SpatialGradient3d(mode='diff')
    result = spatialgradient3d(t_gray)
    result = result[0,0,0,:]
    result_arr: np.ndarray = kornia.tensor_to_image(result.float())
    logger.debug(f"Calculated gradient of shape {result_arr.shape}")
    return result_arr


def laplacian(img_gray: np.ndarray, kernel_size=5) -> np.ndarray:
    """Laplacian filter a numpy array

    Arguments:
        img_gray {np.ndarray} -- 
    
    Returns:
        np.ndarray -- filtered array
    """

    float_img_gray = img_as_float(np.clip(img_gray, 0.0, 1.0))
    t_gray = kornia.utils.image_to_tensor(np.array(float_img_gray)).float().unsqueeze(0)
    laplacian: torch.Tensor = kornia.laplacian(t_gray, kernel_size=kernel_size)
    laplacian_img: np.ndarray = kornia.tensor_to_image(laplacian.float())
    return laplacian_img


def gradient(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial gradient of an rgb numpy array
    Converts to grayscale

    Arguments:
        img_rgb {np.ndarray} -- input array
    
    Returns:
        np.ndarray -- input array
    """

    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    t_rgb = kornia.utils.image_to_tensor(np.array(img_rgb)).float().unsqueeze(0)
    t_gray = kornia.rgb_to_grayscale(t_rgb.float() / 255.)
    grads: torch.Tensor = kornia.spatial_gradient(t_gray, order=1)  # BxCx2xHxW
    grads_x = grads[:, :, 0]
    grads_y = grads[:, :, 1]
    img_grads_x: np.ndarray = kornia.tensor_to_image(grads_x.float())
    img_grads_y: np.ndarray = kornia.tensor_to_image(grads_y.float())
    return (img_grads_x,img_grads_y)

#
# Pytorch
#

class GaussianSmoothing(nn.Module):
    """
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        
        super(GaussianSmoothing, self).__init__()
        
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
        
def gaussian_blur3d(img_gray : np.ndarray, sigma: float =1.0):
    t_gray = kornia.utils.image_to_tensor(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0)
    logger.debug(f"Pytorch gblur returns {t_gray.shape}")    
    smoothing = GaussianSmoothing(1, 5, sigma, dim=3)
    #input_t = torch.rand(1, 1, 100, 100, 100)
    input_t = F.pad(t_gray, (2, 2, 2, 2,2,2))
    output = smoothing(input_t)

    #smoothing = GaussianBlur(3, 3, sigma)
    #t_gray = F.pad(t_gray, (2, 2, 2, 2), mode='reflect')
    #output = smoothing(t_gray)
    logger.debug(f"Pytorch gblur returns {output.shape}")
    output: np.ndarray = kornia.tensor_to_image(output.squeeze(0).squeeze(0).float())
    return output


