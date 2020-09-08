"""
Filtering and survos feature generation

"""
import math
import numbers
import numpy as np

from skimage.filters import gaussian
from skimage import img_as_float
from scipy import ndimage

import torch
from torch import nn
from torch.nn import functional as F
import kornia
from loguru import logger


def ndimage_laplacian(I, kernel_size=1.0):
    locNaNs = np.isnan(I)
    I = np.nan_to_num(I)
    I = ndimage.laplace(gaussian(I, kernel_size))
    I = I / np.max(I)
    return I


#
# Kornia features
#


def spatial_gradient_3d(vol_gray: np.ndarray) -> np.ndarray:
    """Spatial gradient of a array of intensity values
    
    Arguments:
        vol_gray {np.ndarray} -- input array
    Returns:
        np.ndarray -- filtered array
    """
    img_gray = img_as_float(np.clip(vol_gray, 0.0, 1.0))

    t_gray = (
        kornia.utils.image_to_tensor(np.array(img_gray))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    spatialgradient3d = kornia.filters.SpatialGradient3d(mode="diff")
    result = spatialgradient3d(t_gray)
    result = result[0, 0, 0, :]
    result_arr: np.ndarray = kornia.tensor_to_image(result.float())
    logger.debug(f"Calculated gradient of shape {result_arr.shape}")

    return result_arr


def laplacian(img_gray: np.ndarray, kernel_size=5.0) -> np.ndarray:
    """Laplacian filter a numpy array

    Arguments:s
        img_gray {np.ndarray} -- 
    
    Returns:
        np.ndarray -- filtered array
    """

    float_img_gray = img_as_float(np.clip(img_gray, 0.0, 1.0))
    t_gray = kornia.utils.image_to_tensor(np.array(float_img_gray)).float().unsqueeze(0)
    laplacian: torch.Tensor = kornia.laplacian(t_gray, kernel_size=kernel_size)
    laplacian_img: np.ndarray = kornia.tensor_to_image(laplacian.float())

    return laplacian_img
