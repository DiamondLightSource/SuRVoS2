import numpy as np
from loguru import logger

# from pycuda import cumath, gpuarray
# from ..cuda import asgpuarray
from ..utils import gpufeature


@gpufeature
def compute_local_mean(data, radius):
    from .conv import conv_sep

    logger.info(f"+ Padding data with radius {radius}")
    d, h, w = radius
    d = int(d)
    h = int(h)
    w = int(w)
    data = np.pad(data, ((d, d), (h, h), (w, w)), mode="reflect")

    logger.info("+ Computing local mean features (radius={})".format((d, h, w)))
    kernelz = np.ones(d * 2 + 1, np.float32)
    kernelz /= kernelz.sum()
    kernely = np.ones(h * 2 + 1, np.float32)
    kernely /= kernely.sum()
    kernelx = np.ones(w * 2 + 1, np.float32)
    kernelx /= kernelx.sum()

    features = conv_sep(
        data, [kernelz, kernely, kernelx]
    )  # gconvssh(data, kernelz, kernely, kernelx, gpu=DM.selected_gpu)

    return features


@gpufeature
def compute_local_std(data, radius):
    from .conv import conv_sep

    logger.info("+ Padding data")
    d, h, w = radius
    d = int(d)
    h = int(h)
    w = int(w)
    data = np.pad(data, ((d, d), (h, h), (w, w)), mode="reflect")

    logger.info("+ Computing local mean features (radius={})".format((d, h, w)))
    kernelz = np.ones(d * 2 + 1, np.float32)
    kernelz /= kernelz.sum()
    kernely = np.ones(h * 2 + 1, np.float32)
    kernely /= kernely.sum()
    kernelx = np.ones(w * 2 + 1, np.float32)
    kernelx /= kernelx.sum()

    logger.info("   - Computing local mean squared")
    meansq = conv_sep(
        data, [kernelz, kernely, kernelx]
    )  # gconvssh(data, kernelz, kernely, kernelx, gpu=DM.selected_gpu)
    meansq **= 2

    logger.info("   - Computing local squared mean")
    data **= 2

    sqmean = conv_sep(
        data, [kernelz, kernely, kernelx]
    )  # gconvssh(data, kernelz, kernely, kernelx, gpu=DM.selected_gpu)

    sqmean -= meansq
    np.maximum(sqmean, 0, out=sqmean)
    np.sqrt(sqmean, sqmean)  # inplace

    return sqmean


def compute_local_centering(data=None, params=None):
    mean = compute_local_mean(data=data, params=params)
    return data - mean


def compute_local_norm(data=None, params=None):
    mean = compute_local_mean(data=data, params=params)
    std = compute_local_std(data=data, params=params)
    mask = ~np.isclose(std, 0)
    result = np.zeros_like(data)
    result[mask] = (data[mask] - mean[mask]) / std[mask]
    return result


def compute_local_magnitude(data=None, params=None):
    gz, gy, gx = np.gradient(data)
    mag = np.sqrt(gz**2 + gy**2 + gx**2)
    mean = compute_local_mean(data=mag, params=params)
    return mean
