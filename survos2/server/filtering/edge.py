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


#
# Ndimage
#
def ndimage_laplacian(I, kernel_size=1.0):
    logger.info("+ Computing ndimage laplacian")
    locNaNs = np.isnan(I)
    I = np.nan_to_num(I)
    I = ndimage.laplace(gaussian(I, kernel_size))
    I = I - np.min(I)
    I = I / np.max(I)
    return I


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
    logger.info("+ Calculating spatial gradient")
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


def laplacian(img_gray: np.ndarray, kernel_size=5.0) -> np.ndarray:
    """Laplacian filter a numpy array

    Arguments:s
        img_gray {np.ndarray} --

    Returns:
        np.ndarray -- filtered array
    """
    logger.info("+ Calculating laplacian")
    float_img_gray = img_as_float(np.clip(img_gray, 0.0, 1.0))
    t_gray = kornia.utils.image_to_tensor(np.array(float_img_gray)).float().unsqueeze(0)
    laplacian: torch.Tensor = kornia.laplacian(t_gray, kernel_size=int(kernel_size[0]))
    laplacian_img: np.ndarray = kornia.tensor_to_image(laplacian.float())

    return rescale_denan(laplacian_img)

def rescale_denan(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.nan_to_num(img)
    return img


def compute_difference_gaussians(data, sigma, sigma_ratio, threshold=False):
    sigma = np.asarray(sigma)
    sigma2 = np.asarray(sigma) * sigma_ratio

    logger.info("+ Computing difference of Gaussians with {sigma} {sigma_ratio}")

    # if 'Response' in params and params['Response'] == 'Dark':
    #    data *= -1

    g1 = gaussian_blur_kornia(data, sigma)
    g2 = gaussian_blur_kornia(data, sigma2)

    response = g1 - g2

    if threshold:
        response[response < 0] = 0

    response = rescale_denan(response)

    return response


# def compute_laplacian_gaussian(data=None, params=None):
#     sz, sy, sx = params['Sigma']
#     out = np.zeros_like(data)

#     for i, (oz, oy, ox) in enumerate([(2,0,0),(0,2,0),(0,0,2)]):
#         kz = make_gaussian_1d(sz, order=oz, trunc=3)
#         ky = make_gaussian_1d(sy, order=oy, trunc=3)
#         kx = make_gaussian_1d(sx, order=ox, trunc=3)

#         if 'Response' in params and params['Response'] == 'Bright':
#             if i == 0: kz *= -1
#             if i == 1: ky *= -1
#             if i == 2: kx *= -1

#         log.info('+ Padding data')
#         d, h, w = kz.size//2, ky.size//2, kx.size//2
#         tmp = np.pad(data, ((d,d),(h,h),(w,w)), mode='reflect')

#         log.info('   - Computing convolutions (radius={})'.format((d, h, w)))
#         gauss = gconvssh(tmp, kz, ky, kx, gpu=DM.selected_gpu)
#         out += gauss

#     if 'Threshold' in params and params['Threshold']:
#         out[out < 0] = 0

#     return out
