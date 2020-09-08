import numpy as np
import pandas as pd

from loguru import logger

#
# GEOMETRIES
#


def centroid_3d(arr):
    length = arr.shape[0]

    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])

    return sum_x / length, sum_y / length, sum_z / length


def rescale_3d(X, x_scale, y_scale, z_scale):
    X_rescaled = np.zeros_like(X)
    X_rescaled[:, 0] = X[:, 0] * x_scale
    X_rescaled[:, 1] = X[:, 1] * y_scale
    X_rescaled[:, 2] = X[:, 2] * z_scale

    return X_rescaled
