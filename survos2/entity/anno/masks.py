"""
Mask generation functions 

"""

# from __future__ import division
from functools import lru_cache
from numba import cuda, float32
import numpy as np
import math

from numba import jit
import random

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

from numpy.lib.stride_tricks import as_strided as ast
from numpy.random import permutation
from numpy import linalg

from survos2.frontend.nb_utils import summary_stats, show_images

from numpy.linalg import LinAlgError


from skimage.segmentation import mark_boundaries
from survos2.entity.pipeline import Patch
from survos2.server.state import cfg


def generate_anno(
    precropped_wf2,
    classwise_entities,
    cfg,
    padding=(50, 50, 50),
    eccentricity=(1.0, 1.0, 1.0),
    remove_padding=False,
    core_mask_radius=(14, 14, 14),
    shell=False,
):
    # calculate padding
    padded_vol = np.zeros(
        (
            precropped_wf2.shape[0] + padding[0] * 2,
            precropped_wf2.shape[1] + padding[1] * 2,
            precropped_wf2.shape[2] + padding[2] * 2,
        )
    )
    # pad input image
    padded_vol[
        padding[0] : precropped_wf2.shape[0] + padding[0],
        padding[1] : precropped_wf2.shape[1] + padding[1],
        padding[2] : precropped_wf2.shape[2] + padding[2],
    ] = precropped_wf2

    # set params
    cfg["pipeline"]["mask_params"]["core_mask_radius"] = core_mask_radius
    cfg["pipeline"]["mask_params"]["padding"] = padding
    cfg["pipeline"]["mask_params"]["eccentricity"] = eccentricity
    print(f"Using padding {padding} eccentricity {eccentricity}")

    anno_masks = []

    # for each class, set params and make masks
    for k, v in classwise_entities.items():
        p = Patch({"Main": padded_vol}, {}, {"Points": classwise_entities[k]["entities"]}, {})
        cfg["pipeline"]["mask_params"]["mask_radius"] = v["size"]
        cfg["pipeline"]["mask_params"]["eccentricity"] = eccentricity
        from survos2.entity.pipeline_ops import make_masks

        p = make_masks(p, cfg["pipeline"])

        if remove_padding:
            mask = p.image_layers["total_mask"][
                padding[0] : precropped_wf2.shape[0] + padding[0],
                padding[1] : precropped_wf2.shape[1] + padding[1],
                padding[2] : precropped_wf2.shape[2] + padding[2],
            ]
            core_mask = p.image_layers["core_mask"][
                padding[0] : precropped_wf2.shape[0] + padding[0],
                padding[1] : precropped_wf2.shape[1] + padding[1],
                padding[2] : precropped_wf2.shape[2] + padding[2],
            ]
        else:
            mask = (p.image_layers["total_mask"] > 0) * 1.0
            core_mask = (p.image_layers["core_mask"] > 0) * 1.0

        shell_mask = mask - core_mask

        # store masks in dictionary
        classwise_entities[k]["mask"] = mask
        classwise_entities[k]["core_mask"] = core_mask
        classwise_entities[k]["shell_mask"] = shell_mask

    return classwise_entities, padded_vol


@lru_cache(maxsize=64)
def ellipsoidal_mask(
    d: int,
    w: int,
    h: int,
    a: float = 1,
    b: float = 1,
    c: float = 1,
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
        center = [np.int(d / 2.0, np.int(w / 2.0), np.int(h / 2.0))]

    if radius is None:
        radius = np.min(center[0], center[1], center[2], w - center[0], h - center[1], d - center[2])

    if debug_verbose:
        print("Making sphere of radius: {}".format(radius))
    Z, X, Y = np.ogrid[: int(d), : int(w), : int(h)]

    if debug_verbose:
        print("At center: {}".format(center))

    dist_from_center = np.sqrt(
        (((X - center[1]) ** 2) / b + ((Y - center[2]) ** 2) / c + ((Z - center[0]) ** 2) / a)
    )

    mask = dist_from_center <= float(radius)

    if debug_verbose:
        print("Area of mask: {}".format(np.sum(mask)))

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

    print("At center: {}".format(center))

    mask = np.zeros((d, w, h))

    mask[
        center[0] - box_dim[0] // 2 : center[0] + box_dim[0] // 2,
        center[1] - box_dim[1] // 2 : center[1] + box_dim[1] // 2,
        center[2] - box_dim[2] // 2 : center[2] + box_dim[2] // 2,
    ] == 1

    print("Area of mask: {}".format(np.sum(mask)))

    mask = (mask * 1.0).astype(np.float32)

    return mask


def bw_to_points(bwimg, sample_prop):
    """Given a thresholded binary img, returns points sampled from mask regions.

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
    # print(pt, input_array.shape)

    test_z = (pt[0] - sz[0] >= 0) and (pt[0] + sz[0] < input_array.shape[0])
    test_x = (pt[1] - sz[1] >= 0) and (pt[1] + sz[1] < input_array.shape[1])
    test_y = (pt[2] - sz[2] >= 0) and (pt[2] + sz[2] < input_array.shape[2])
    # print(test_z, test_x, test_y)
    return test_z & test_x & test_y


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
    input_array: np.ndarray,
    classwise_pts: np.ndarray,
    patch_size=(64, 64, 64),
    radius: int = 24,
    ecc=(1.0, 1.0, 1.0),
):
    """Copies a small sphere mask to locations centered on classwise_pts

    Args:
        input_array (np.ndarray): input array. a blank array of the same dimensions is made.
        classwise_pts (np.ndarray): array of 3d points that serve as the locations of the spheres
        radius (int, optional): sphere radius. Defaults to 10.
    """
    total_mask = np.zeros_like(input_array)
    patch_size = (radius * 2, radius * 2, radius * 2)
    sz = np.array(patch_size).astype(np.uint32)
    count = 0
    classwise_pts = classwise_pts.astype(np.uint32)
    c = (np.array(patch_size) // 2).astype(np.uint32)

    #     init_ls = calc_sphere(
    #        patch_size,
    #        c,
    #        radius=radius//2,
    #     )

    init_ls = ellipsoidal_mask(
        patch_size[0],
        patch_size[1],
        patch_size[2],
        center=(c[0], c[1], c[2]),
        a=ecc[0],
        b=ecc[1],
        c=ecc[2],
        radius=radius,
    )

    for pt in classwise_pts:

        st = np.array((pt[0] - radius, pt[2] - radius, pt[1] - radius))
        st = st.astype(np.uint32)

        if st[0] < input_array.shape[0] and st[1] < input_array.shape[1] and st[2] < input_array.shape[2]:
            if point_in_vol(input_array, pt, sz):
                total_mask[st[0] : st[0] + sz[0], st[1] : st[1] + sz[1], st[2] : st[2] + sz[2]] += init_ls

    return total_mask


def generate_sphere_masks(I_out, classwise_pts, radius=10):

    total_mask = np.zeros_like(I_out)

    count = 0

    classwise_pts = classwise_pts.astype(np.uint32)

    print(f"Rendering {len(classwise_pts)} masks")

    for i in range(len(classwise_pts)):
        ss, yy, xx = classwise_pts[i, 0], classwise_pts[i, 1], classwise_pts[i, 2]
        ss = int(ss)
        init_ls = calc_sphere(I_out.shape, (ss, xx, yy), radius=radius)
        total_mask += init_ls

    return total_mask
