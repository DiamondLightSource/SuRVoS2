"""
Mask generation functions 

"""

import numpy as np
import math
from numpy.linalg import LinAlgError
from numpy.lib.stride_tricks import as_strided as ast
from numpy.random import permutation
from numpy import linalg
from numba import cuda, float32
from numba import jit
import random

import hdbscan
from collections import Counter
from statistics import mode, StatisticsError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

import time
import glob

import collections
import numpy as np
import pandas as pd
from typing import NamedTuple, Tuple, Union
import itertools

from scipy import ndimage
import torch.utils.data as data

import skimage
from skimage.morphology import thin
from skimage.io import imread, imread_collection
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


from survos2.frontend.nb_utils import summary_stats, show_images


from survos2.entity.anno.geom import centroid_3d, rescale_3d

from survos2.entity.anno.point_cloud import chip_cluster
from skimage.segmentation import mark_boundaries

from loguru import logger


def create_ellipsoidal_mask(
    d: int,
    w: int,
    h: int,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: int = 8,
    debug_verbose: bool = False,
) -> np.ndarray:
    """Paint ellipsoidal mask into a volume

    Args:
        d (int): Depth
        w (int): Width
        h (int): Height
        a (float, optional): z axis scale. Defaults to 1.0.
        b (float, optional): x axis scale. Defaults to 1.0.
        c (float, optional): y axis scale. Defaults to 1.0.
        center (Tuple[float, float, float], optional): ellipse centre. Defaults to (0.0,0.0,0.0).
        radius (int, optional): Ellipse radius. Defaults to 8.
        debug_verbose (bool, optional): Debug messages. Defaults to False.

    Returns:
        np.ndarray: Volume of ellipse mask
    """

    if center is None:
        center = [np.int(w / 2.0), np.int(h / 2.0), np.int(d / 2.0)]

    if radius is None:
        radius = np.min(
            center[0], center[1], center[2], w - center[0], h - center[1], d - center[2]
        )

    if debug_verbose:
        logger.debug(f"Making ellipse of radius: {radius} at center {center}")
    Z, X, Y = np.ogrid[: int(d), : int(w), : int(h)]

    dist_from_center = np.sqrt(
        (
            ((X - center[1]) ** 2) / a
            + ((Y - center[2]) ** 2) / b
            + ((Z - center[0]) ** 2) / c
        )
    )

    mask = dist_from_center <= np.float(radius)

    if debug_verbose:
        logger.debug(f"Area of mask: {np.sum(mask)}")

    mask = (mask * 1.0).astype(np.float32)

    return mask  # , dist_from_center


def create_rect_mask(
    mask_dim: Tuple[int, int, int] = (100, 100, 100),
    center: Union[None, Tuple[float, float, float]] = None,
    box_dim: Tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    d, w, h = mask_dim

    if center is None:  # use the middle of the image
        center = [np.int(w / 2.0), np.int(h / 2.0), np.int(d / 2.0)]

    logger.debug("At center: {}".format(center))

    mask = np.zeros((d, w, h))

    mask[
        center[0] - box_dim[0] // 2 : center[0] + box_dim[0] // 2,
        center[1] - box_dim[1] // 2 : center[1] + box_dim[1] // 2,
        cener[2] - box_dim[2] // 2 : center[2] + box_dim[2] // 2,
    ] == 1

    logger.debug("Area of mask: {}".format(np.sum(mask)))

    mask = (mask * 1.0).astype(np.float32)

    return mask


def paint_ellipsoids():

    cluster_centroids_df = pd.DataFrame(cluster_centroids)
    cluster_centroids_offset = cluster_centroids.copy() * 1000
    cluster_centroids_offset = cluster_centroids_offset.astype(np.uint32)

    img_in = np.zeros((99, 256, 256))

    total_mask = 0.1 * np.random.random(
        (img_in.shape[0], img_in.shape[1], img_in.shape[2])
    )  # wide_patch.shape[0]))

    for c, ecc in cluster_coords_offset[0:16]:

        sphere_mask = create_ellipsoidal_mask(
            img_in.shape[0],
            img_in.shape[1],
            img_in.shape[2],
            center=(c[0], c[1], c[2]),
            a=ecc[0],
            b=ecc[1],
            c=ecc[2],
            radius=7,
        )
        total_mask += sphere_mask


def ellipse_mask(w, h, a=1.0, b=1.0, center=None, radius=4.0):
    """Generate 2d ellipse masks

    Args:
        w (int): Width of mask
        h (int): Height of mask
        a (float, optional): scale of major axis Defaults to 1.0.
        b (float, optional): scale of minor axis. Defaults to 1.0.
        center ([type], optional): center of ellipse Defaults to None.
        radius (float, optional): radius of ellipse Defaults to 4.0.

    Returns:
        np.ndarry: mask
    """
    if center is None:
        center = [np.int(w / 2.0), np.int(h / 2.0)]

    X, Y = np.ogrid[: int(w), : int(h)]

    dist_from_center = np.sqrt(
        (((X - center[0]) ** 2) / a + ((Y - center[1]) ** 2) / b)
    )

    mask = dist_from_center <= np.float(radius)

    mask = (mask * 1.0).astype(np.float32)

    return mask


def bw_to_points(bwimg, sample_prop):
    """ Given a thresholded binary img, returns points sampled from mask regions.
    
    Parameters
    ----------
 
    bwimg: a binary image
    sample_prop: a number from 0 to 1 representing the proportion of pixels to sample
 
    Returns
    ndarray of points
 
    """

    pp = nonzero(bwimg)
    points = zeros([len(pp[0]), 2])
    points[:, 0] = pp[0]
    points[:, 1] = pp[1]

    num_samp = sample_prop * points.shape[0]
    points = np.floor(permutation(points))[0:num_samp, :]

    return points


def point_in_vol(input_array, pt, sz):

    test_z = (pt[0] - sz[0] >= 0) and (pt[0] + sz[0] < input_array.shape[0])
    test_x = (pt[1] - sz[1] >= 0) and (pt[1] + sz[1] < input_array.shape[1])
    test_y = (pt[2] - sz[2] >= 0) and (pt[2] + sz[2] < input_array.shape[2])
    return test_z & test_x & test_y


# @jit(nopython=False)
def calc_sphere(
    image_shape: Tuple[int, int, int] = (100, 100, 100),
    center: Tuple[int, int, int] = (40, 40, 40),
    radius: int = 30,
):

    grid = np.mgrid[[slice(i) for i in image_shape]]

    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid) ** 2, 0))
    res = np.int8(phi > 0)

    return res


def generate_sphere_masks_fast(
    input_array: np.ndarray, classwise_pts: np.ndarray, radius: int = 10
):
    """Copies a small sphere mask to locations centered on classwise_pts

    Args:
        input_array (np.ndarray): input array. a blank array of the same dimensions is made.
        classwise_pts (np.ndarray): array of 3d points that serve as the locations of the spheres
        radius (int, optional): sphere radius. Defaults to 10.
    """
    total_mask = np.zeros_like(input_array)
    sz = np.array((radius, radius, radius)).astype(np.uint32)

    count = 0

    classwise_pts = classwise_pts.astype(np.uint32)

    init_ls = calc_sphere(
        (radius, radius, radius),
        (radius // 2, radius // 2, radius // 2),
        radius=radius // 2,
    )

    for pt in classwise_pts:
        st = np.array((pt[0] - radius // 2, pt[2] - radius // 2, pt[1] - radius // 2))
        st = st.astype(np.uint32)
        if (
            st[0] < input_array.shape[0]
            and st[1] < input_array.shape[1]
            and st[2] < input_array.shape[2]
        ):

            if point_in_vol(input_array, pt, sz):
                total_mask[
                    st[0] : st[0] + sz[0], st[1] : st[1] + sz[1], st[2] : st[2] + sz[2]
                ] = init_ls

    return total_mask


def generate_sphere_masks(I_out, classwise_pts, radius=10):

    total_mask = np.zeros_like(I_out)

    count = 0

    classwise_pts = classwise_pts.astype(np.uint32)
    logger.debug(f"Rendering {len(classwise_pts)} masks")

    for i in range(len(classwise_pts)):

        ss, yy, xx = classwise_pts[i, 0], classwise_pts[i, 1], classwise_pts[i, 2]
        ss = int(ss)
        init_ls = calc_sphere(I_out.shape, (ss, xx, yy), radius=radius)
        logger.debug(init_ls.shape)
        total_mask += init_ls

    return total_mask
