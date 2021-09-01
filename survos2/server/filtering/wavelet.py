import numpy as np
from itertools import combinations_with_replacement
from loguru import logger
from .blur import gaussian_blur_kornia
import torch
import kornia
import numbers
from .base import rescale_denan
from pywt import wavedecn, waverecn


def wavelet(I, level, wavelet="sym3", threshold=64.0, hard=True):
    mode = "symmetric"
    arr = np.float32(I)
    coeffs = wavedecn(arr, wavelet=wavelet, mode=mode, level=level)

    coeffs_H = list(coeffs)

    if hard:
        coeffs_H[0][coeffs_H[0] < threshold] = 0
    else:
        coeffs_H[0] = np.sign(coeffs_H[0]) * np.abs(coeffs_H[0] - threshold)

    arr_rec = waverecn(coeffs_H, wavelet=wavelet, mode=mode)

    return arr_rec
