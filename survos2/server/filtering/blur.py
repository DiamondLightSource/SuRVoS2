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
from kornia.filters import filter3d

from survos2.model import DataModel

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

    return kernel


def gaussian_blur_t(img_t: torch.Tensor, sigma, device=None):
    kernel_size = 3  # min size, expands to sigma if sigma is larger
    # kernel = torch.ones(1, 3, 3, 3) # example kernel
    kernel = make_gaussian_kernel(kernel_size, sigma, dim=3)
    

    p = int((kernel_size - 1) / 2)
    padded_img_t = F.pad(img_t, (p, p, p, p, p, p))
    if device:
        padded_img_t.to(device)
    output_t = filter3d(padded_img_t, kernel)
    output_t = output_t[:, :, p:-p, p:-p, p:-p]
    if device:
        output_t = output_t.cpu()
    return output_t


def gaussian_blur_kornia(img: np.ndarray, sigma, device=None):
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

    img_t = (
        kornia.utils.image_to_tensor(np.array(img)).float().unsqueeze(0).unsqueeze(0)
    )
    output = gaussian_blur_t(img_t, sigma, device=DataModel.g.device)
    #print(f"Calculating gaussian blur on device {DataModel.g.device}")
    output: np.ndarray = kornia.tensor_to_image(output.squeeze(0).squeeze(0).float())

    return output


class GaussianSmoothing(nn.Module):
    """Pure pytorch Gaussian smoothing"""

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
        The result of the filtering resulting. Use `.get()` to
        retrieve the corresponding Numpy array.
    """

    result = data - gaussian_blur_kornia(data, sigma=sigma)

    result -= np.min(result)
    result /= np.max(result)
    
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
    sigma : float

    Returns
    -------
    result : 3 dimensional numpy array
    """
    num = gaussian_center(data, sigma=sigma)
    den = np.sqrt(gaussian_blur_kornia(num ** 2, sigma=sigma))

    # TODO numerical precision ignore den < 1e-7
    num /= den

    num -= np.min(num)
    num /= np.max(num)

    return num


# adapted from kornia.losses.TotalVariation
class TotalVariation(nn.Module):
    r"""Computes the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, D, H, W)`.
        - Output: :math:`(N, D, H, W)`.
    """

    def __init__(self) -> None:
        super(TotalVariation, self).__init__()

    def forward(self, img) -> torch.Tensor:  # type: ignore
        return total_variation(img)


def total_variation(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Total Variation.
    Adapted from :class:`~kornia.losses.TotalVariation`.
    """
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    img_shape = img.shape
    if len(img_shape) == 3 or len(img_shape) == 4:
        pixel_dif1 = img[..., :, 1:, :] - img[..., :, :-1, :]
        pixel_dif2 = img[..., :, :, 1:] - img[..., :, :, :-1]
        pixel_dif3 = img[..., 1:, :, :] - img[..., :-1, :, :]

        # reduce_axes = (-3, -2, -1)
        reduce_axes = list(range(1, 4))
    else:
        raise ValueError(
            "Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape))
        )

    return (
        pixel_dif1.abs().sum(dim=reduce_axes)
        + pixel_dif2.abs().sum(dim=reduce_axes)
        + pixel_dif3.abs().sum(dim=reduce_axes)
    )


class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image, regularization_amount=0.001):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction="mean")
        self.regularization_term = TotalVariation()
        self.denoised_image = torch.nn.Parameter(
            data=noisy_image.clone(), requires_grad=True
        )
        self.noisy_image = noisy_image
        self.regularization_amount = regularization_amount

    def forward(self):
        return self.l2_term(
            self.denoised_image, self.noisy_image
        ) + self.regularization_amount * self.regularization_term(self.denoised_image)

    def get_denoised_image(self):
        return self.denoised_image


def tvdenoise_kornia(img, regularization_amount=0.001, max_iter=50):
    img_t = kornia.utils.image_to_tensor(np.array(img)).float().unsqueeze(0)
    tv_denoiser = TVDenoise(img_t, regularization_amount)
    optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = tv_denoiser()
        if i % 25 == 0:
            logger.debug(
                "Loss in iteration {} of {}: {:.3f}".format(i, max_iter, loss.item())
            )
        loss.backward()
        optimizer.step()

    img_clean: np.ndarray = kornia.tensor_to_image(tv_denoiser.get_denoised_image())
    return img_clean


