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

from .blur import gaussian_blur_kornia
from .base import rescale_denan

#
# Ndimage
#
def ndimage_laplacian(img, kernel_size=1.0):
    """Laplacian filter
    Uses ndimage implementation

    Parameters
    ----------
    I : np.array (D,H,W)
        Input image
    kernel_size : float, optional
        Kernel size, by default 1.0

    Returns
    -------
    np.array (D,H,W)
        Filtered image
    """

    locNaNs = np.isnan(img)
    img = np.nan_to_num(img)
    img = ndimage.laplace(gaussian(img, kernel_size))

    return img


#
# Kornia features
#
def spatial_gradient_3d(vol_gray: np.ndarray, dim=0) -> np.ndarray:
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
    result = result[0, 0, dim, :]
    result_arr: np.ndarray = kornia.tensor_to_image(result.float())
    logger.debug(f"Calculated gradient of shape {result_arr.shape}")

    return result_arr


def laplacian(img: np.ndarray, kernel_size) -> np.ndarray:
    """Laplacian filter a numpy array

    Arguments: np.ndarray (D,H,W)
        Input image

    Returns:
        np.ndarray -- filtered array
    """
    img_clean = rescale_denan(img_as_float(np.clip(img, 0.0, 1.0)))
    img_clean_t = kornia.utils.image_to_tensor(np.array(img_clean)).float().unsqueeze(0)
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    laplacian: torch.Tensor = kornia.filters.laplacian(img_clean_t, kernel_size=kernel_size)
    laplacian_img: np.ndarray = kornia.tensor_to_image(laplacian.float())
    return np.nan_to_num(laplacian_img)


def compute_difference_gaussians(
    data, sigma, sigma_ratio, threshold=False, dark_response=False
):
    """Difference of Gaussians (DoG) filter

    Parameters
    ----------
    data : np.array (D,H,W)
        Input image
    sigma : Vector of 3 floats
        Kernel size
    sigma_ratio : Float
        Ratio between the kernel of the two gaussian filters
    threshold : bool, optional
        Threshold removal of values less than 0, by default False
    dark_response:
        Use the negative of the input data
    Returns
    -------
    np.array (D,H,W)
        Filtered array
    """
    sigma = np.asarray(sigma)
    sigma2 = np.asarray(sigma) * sigma_ratio

    if dark_response:
        data *= -1

    g1 = gaussian_blur_kornia(data, sigma)
    g2 = gaussian_blur_kornia(data, sigma2)

    response = g1 - g2

    if threshold:
        response[response < 0] = 0

    response = np.nan_to_num(response)

    return response
