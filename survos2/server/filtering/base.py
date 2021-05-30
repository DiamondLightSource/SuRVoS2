import math
import numbers
import numpy as np

from skimage.filters import gaussian
from skimage import img_as_float
from scipy import ndimage

from skimage import exposure

import torch
from torch import nn
from torch.nn import functional as F
import kornia
from loguru import logger


def simple_invert(data):
    """Invert the input image

    Parameters
    ----------
    data : np.ndarray (D,H,W)
        Input image
    Returns
    -------
    np.ndarray (D,H,W)
        Inverted image
    """
    return 1.0 - data


def median_filter(data, size=5):
    """Median filter, using ndimage implementation.

    Parameters
    ----------
    data : np.ndarray (D,H,W)
        Input image
    size : int, optional
        Median size, by default 5

    Returns
    -------
    np.ndarray (D,H,W)
        Median filtered image
    """
    return ndimage.median_filter(data, size=size)


def rescale_denan(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.nan_to_num(img)
    return img

def gamma_adjust(data, gamma=1.0):
    """Gamma adjust filter using skimage implementation

    Parameters
    ----------
    data : np.ndarray (D,H,W)
        Input image 
    gamma : float
        Gamma

    Returns
    -------
    np.ndarray
        Gamma adjusted image
    """
    return np.nan_to_num(exposure.adjust_gamma(data, gamma))

