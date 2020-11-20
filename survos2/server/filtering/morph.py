from scipy import ndimage
import numpy as np


def erode(I, thresh, num_iter):
    I = (I >= thresh) * 1.0
    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_erosion(I, structure=struct2).astype(I.dtype)

    return I


def dilate(I, thresh, num_iter):
    I = (I >= thresh) * 1.0
    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_dilation(I, structure=struct2).astype(I.dtype)

    return I


def median(I, thresh, median_size, num_iter):
    I = (I >= thresh) * 1.0
    for i in range(num_iter):
        I = ndimage.median_filter(I, median_size).astype(I.dtype)

    return I
