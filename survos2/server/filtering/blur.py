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


def make_gaussian_kernel(kernel_size, sigma, dim=3):
    """Make gaussian kernel

    Parameters
    ----------
    kernel_size : Int
        size of square kernel side dimension
    sigma : Tuple of Float
        Sigma value
    dim : int, optional
        [description], by default 3

    Returns
    -------
    torch.Tensor
        the kernel
    """

    if np.max(np.array(sigma)) > kernel_size:
        kernel_size = sigma

    channels = 1

    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim

    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
        )

    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, *kernel.size())
    # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

def gaussian_blur_t(img_t: torch.Tensor, sigma):
    kernel_size = 3 # min size, expands to sigma if sigma is larger 
    # kernel = torch.ones(1, 3, 3, 3) # example kernel
    kernel = make_gaussian_kernel(kernel_size, sigma, dim=3)
    from kornia.filters import filter3D
    p = int((kernel_size-1)/ 2)
    padded_img_t = F.pad(img_t, (p, p, p, p, p, p))
    output_t = filter3D(padded_img_t, kernel)
    output_t = output_t[:,:,p:-p,p:-p,p:-p]
    return output_t

def gaussian_blur_kornia(img: np.ndarray, sigma):
    """Gaussian blur using Kornia Filter3D

    Parameters
    ----------
    img : np.ndarray
        Input image array
    sigma : float, optional
        sigma value, by default 1.0

    Returns
    -------
    output
        filtered numpy array
    """
    logger.info("+ Computing gaussian blur")
    img_t = (
        kornia.utils.image_to_tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0)
    )
    output = gaussian_blur_t(img_t, sigma)
    output: np.ndarray = kornia.tensor_to_image(output.squeeze(0).squeeze(0).float())
    
    return output


class GaussianSmoothing(nn.Module):
    """ Pure pytorch Gaussian smoothing"""

    def __init__(self, channels, kernel_size, sigma, dim=2):
        kernel = make_kernel(kernel_size, sigma, dim=3)
        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Kernel dimension must be 1D, 2D or 3D.")

    def forward(self, input):
        logger.info("+ Convolving image with GaussianSmoothing filter")
        return self.conv(input, weight=self.weight, groups=self.groups)


def gaussian_blur_pytorch(img_gray: np.ndarray, sigma: float = 1.0):
    t_gray = (
        kornia.utils.image_to_tensor(np.array(img_gray))
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    smoothing = GaussianSmoothing(1, 5, sigma, dim=3)
    input_t = F.pad(t_gray, (2, 2, 2, 2, 2, 2))
    output = smoothing(input_t)
    output: np.ndarray = kornia.tensor_to_image(output.squeeze(0).squeeze(0).float())

    return output


def gaussian_center(data, sigma=0.5, **kwargs):
    """
    Performs Gaussian centering, where the mean in a Gaussian neighbourhood is
    substracted to every voxel.

    Parameters
    ----------
    data : 3 dimensional array
        The data to be filtered

    sigma : float or array of floats
        The standard deviation of the Gaussian filter used to calculate the
        mean. Controls the radius and strength of the filter.
        If an array is given, it has to satisfy `len(sigma) = data.ndim`.
        Default: 0.5

    Returns
    -------
    result : 2 or 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """

    result = data - gaussian_blur_kornia(data, sigma=sigma)

    return result


def gaussian_norm(data, sigma=0.5, **kwargs):
    """
    Performs Gaussian normalization to an input dataset. This is, every voxel
    is normalized by substracting the mean and dividing it by the standard
    deviation in a Gaussian neighbourhood around it.

    Parameters
    ----------
    data : 3 dimensional numpy array
        The data to be filtered
    sigma : float or array of floats
        The standard deviation of the Gaussian filter used to estimate the mean
        and standard deviation of the kernel. Controls the radius and strength
        of the filter. If an array is given, it has to satisfy
        `len(sigma) = data.ndim`. Default: 0.5

    Returns
    -------
    result : 2 or 3 dimensional filtered `GPUArray`
        The result of the filtering resulting from PyCuda. Use `.get()` to
        retrieve the corresponding Numpy array.
    """
    num = gaussian_center(data, sigma=sigma)
    den = np.sqrt(gaussian_blur_kornia(num ** 2, sigma=sigma))

    # TODO numerical precision ignore den < 1e-7
    num /= den

    return num
