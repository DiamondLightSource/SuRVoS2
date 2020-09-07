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



#
# Pytorch
#

class GaussianSmoothing(nn.Module):
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

        kernel = kernel / torch.sum(kernel)
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
                'Only 1, 2 and 3 dimensions are supported.'
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)
        
def gaussian_blur(img_gray : np.ndarray, sigma: float =1.0):
    t_gray = kornia.utils.image_to_tensor(np.array(img_gray)).float().unsqueeze(0).unsqueeze(0)
    logger.debug(f"Pytorch gblur returns {t_gray.shape}")    
    smoothing = GaussianSmoothing(1, 5, sigma, dim=3)
    #input_t = torch.rand(1, 1, 100, 100, 100)
    input_t = F.pad(t_gray, (2, 2, 2, 2,2,2))
    output = smoothing(input_t)

    logger.debug(f"Pytorch gblur returns {output.shape}")
    output: np.ndarray = kornia.tensor_to_image(output.squeeze(0).squeeze(0).float())
    
    return output

