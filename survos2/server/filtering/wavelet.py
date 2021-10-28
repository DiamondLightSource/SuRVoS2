import numpy as np
from itertools import combinations_with_replacement
from loguru import logger
from .blur import gaussian_blur_kornia
import torch
import kornia
import numbers
from .base import rescale_denan
from pywt import wavedecn, waverecn
import pywt
from pywt import wavedec2, waverec2
from skimage.filters import gaussian, difference_of_gaussians

def wavelet3d(I, level, wavelet="sym3", threshold=64.0, hard=True):
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



def wavelet(I, level, wavelet="db3", threshold=64.0, hard=True):
    mode = "symmetric"
    result = np.zeros_like(I)
    print(I.shape)
    for i in range(I.shape[0]):
        s = np.zeros_like(result[i,:])
        s[0:I.shape[1],0:I.shape[2]] = I[i,:]
        arr = np.float32(s)
        print(f"arr.shape {arr.shape}")
        coeffs = wavedec2(arr, wavelet=wavelet, level=7)
        coeffs_H = list(coeffs)

        idx = int(level)

        if hard:
            for idx in range(idx,7):
                for idx2 in [0,1,2]:
                    #print(np.mean(coeffs_H[idx][idx2]), np.max(coeffs_H[idx][idx2]), np.min(coeffs_H[idx][idx2]))
                    coeffs_H[idx][idx2][coeffs_H[idx][idx2] < threshold] =  0

            #coeffs_H[-idx] == tuple([np.zeros_like(v) for v in coeffs_H[-idx]])
        else:
            for idx in range(idx,0,-1):
                for idx2 in [0,1,2]:
                    coeffs_H[idx] = list(coeffs_H[idx])
                    #coeffs_H[idx][idx2] = np.sign(coeffs_H[idx][idx2]) * np.abs(coeffs_H[idx][idx2] + threshold)
                    coeffs_H[idx][idx2][coeffs_H[idx][idx2] < threshold] =  difference_of_gaussians(coeffs_H[idx][idx2][coeffs_H[idx][idx2] < threshold],1)
            
        # if hard:
        #     coeffs_H[idx][1][coeffs_H[idx][1] < threshold] = 0
        # else:
        #     coeffs_H[idx][1] = np.sign(coeffs_H[idx][1]) * np.abs(coeffs_H[idx][1] - threshold)

        arr_rec = waverec2(coeffs_H, wavelet=wavelet)
        print(f"arr_rec {arr_rec.shape}")
        result[i,0:I.shape[1],0:I.shape[2]] = arr_rec[0:I.shape[1],0:I.shape[2]].copy()

    return result

