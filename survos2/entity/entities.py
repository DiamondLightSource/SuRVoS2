"""
A Entity Dataframe has 'z','x','y','class_code'


"""

import collections
import glob
import itertools
import json
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from statistics import StatisticsError, mode
from typing import List, NamedTuple

import h5py
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import torch.utils.data as data
from loguru import logger
from numpy import linalg
from numpy.lib.stride_tricks import as_strided as ast
from numpy.linalg import LinAlgError
from numpy.random import permutation
from scipy import ndimage
from skimage import data
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.io import imread, imread_collection
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, closing, disk, square, thin
from skimage.segmentation import clear_border, find_boundaries
from sklearn.model_selection import StratifiedKFold, train_test_split

from survos2.entity.anno.geom import centroid_3d, rescale_3d
from survos2.entity.sampler import crop_vol_and_pts_bb, sample_bvol
from survos2.frontend.nb_utils import slice_plot, summary_stats
from survos2.entity.sampler import viz_bvols, centroid_to_bvol, offset_points


def make_entity_mask(wf, dets, flipxy=True, padding=(32, 32, 32)):
    from survos2.entity.utils import pad_vol

    offset_dets = offset_points(dets, (-padding[0], -padding[1], -padding[2]))
    offset_det_bvol = centroid_to_bvol(offset_dets, bvol_dim=padding, flipxy=flipxy)
    padded_vol = pad_vol(wf.vols[0], padding)
    det_mask = viz_bvols(padded_vol, offset_det_bvol)
    return det_mask, offset_dets, padded_vol


def calc_bounding_vol(m):
    return [
        m[0][0],
        m[0][0] + m[1][0],
        m[0][1],
        m[0][1] + m[1][1],
        m[0][2],
        m[0][2] + m[1][2],
    ]


def calc_bounding_vols(main_bv):
    for k, v in main_bv.items():
        main_bv[k]["bb"] = calc_bounding_vol(v["key_coords"])
    return main_bv


def uncrop_pad(img, orig_img, crop_bb):
    blank_img = np.zeros_like(orig_img)
    blank_img[
        crop_bb[0] : crop_bb[0] + img.shape[0],
        crop_bb[2] : crop_bb[2] + img.shape[1],
        crop_bb[4] : crop_bb[4] + img.shape[2],
    ] = img
    return blank_img


def offset_points(pts, patch_pos):
    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]

    logger.debug(f"Offset: {offset_x}, {offset_y}, {offset_z}")

    z = pts[:, 0].copy() - offset_z
    x = pts[:, 1].copy() - offset_x
    y = pts[:, 2].copy() - offset_y

    c = pts[:, 3].copy()

    offset_pts = np.stack([z, x, y, c], axis=1)

    return offset_pts


def make_entity_df(pts, flipxy=True):
    if flipxy:
        entities_df = pd.DataFrame(
            {"z": pts[:, 0], "x": pts[:, 2], "y": pts[:, 1], "class_code": pts[:, 3]}
        )
    else:
        entities_df = pd.DataFrame(
            {"z": pts[:, 0], "x": pts[:, 1], "y": pts[:, 2], "class_code": pts[:, 3]}
        )

    entities_df = entities_df.astype(
        {"x": "int32", "y": "int32", "z": "int32", "class_code": "int32"}
    )
    return entities_df


def make_entity_feats_df(pts, flipxy=True):
    if flipxy:
        entities_df = pd.DataFrame(
            {"z": pts[:, 0], "x": pts[:, 2], "y": pts[:, 1], "class_code": pts[:, 3]}
        )
    else:
        entities_df = pd.DataFrame(
            {"z": pts[:, 0], "x": pts[:, 1], "y": pts[:, 2], "class_code": pts[:, 3]}
        )

    entities_df = entities_df.astype(
        {"x": "int32", "y": "int32", "z": "int32", "class_code": "int32"}
    )
    return entity_feats_df


def make_entity_df2(pts):
    entities_df = pd.DataFrame(
        {"z": pts[:, 0], "x": pts[:, 2], "y": pts[:, 1], "class_code": pts[:, 3]}
    )

    entities_df = entities_df.astype(
        {"x": "float32", "y": "float32", "z": "int32", "class_code": "int32"}
    )

    return entities_df


def make_entity_bvol(bbs, flipxy=False):
    if flipxy:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "area": bbs[:, 1],
                "z": bbs[:, 2],
                "x": bbs[:, 4],
                "y": bbs[:, 3],
                "bb_s_z": bbs[:, 5],
                "bb_s_x": bbs[:, 6],
                "bb_s_y": bbs[:, 7],
                "bb_f_z": bbs[:, 8],
                "bb_f_x": bbs[:, 9],
                "bb_f_y": bbs[:, 10],
            }
        )
    else:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "area": bbs[:, 1],
                "z": bbs[:, 2],
                "x": bbs[:, 3],
                "y": bbs[:, 4],
                "bb_s_z": bbs[:, 5],
                "bb_s_x": bbs[:, 6],
                "bb_s_y": bbs[:, 7],
                "bb_f_z": bbs[:, 8],
                "bb_f_x": bbs[:, 9],
                "bb_f_y": bbs[:, 10],
            }
        )

    entities_df = entities_df.astype(
        {
            "x": "int32",
            "y": "int32",
            "z": "int32",
            "class_code": "int32",
            "bb_s_z": "int32",
            "bb_s_x": "int32",
            "bb_s_y": "int32",
            "bb_f_z": "int32",
            "bb_f_x": "int32",
            "bb_f_y": "int32",
            "area": "int32",
        }
    )
    return entities_df


def make_bvol_df(bbs, flipxy=False):
    entities_df = pd.DataFrame(
        {
            "bb_s_z": bbs[:, 0],
            "bb_s_x": bbs[:, 1],
            "bb_s_y": bbs[:, 2],
            "bb_f_z": bbs[:, 3],
            "bb_f_x": bbs[:, 4],
            "bb_f_y": bbs[:, 5],
            "class_code": bbs[:, 6],
        }
    )

    entities_df = entities_df.astype(
        {
            "bb_s_z": "int32",
            "bb_s_x": "int32",
            "bb_s_y": "int32",
            "bb_f_z": "int32",
            "bb_f_x": "int32",
            "bb_f_y": "int32",
            "class_code": "int32",
        }
    )
    return entities_df


def make_bounding_vols(entities, patch_size=(14, 14, 14)):
    p_z, p_x, p_y = patch_size

    bbs = []

    for z, x, y, c in entities:
        bb_s_z = z - p_z
        bb_s_x = x - p_x
        bb_s_y = y - p_y
        bb_f_z = z + p_z
        bb_f_x = x + p_x
        bb_f_y = y + p_y
        area = (2 * p_z) * (2 * p_x) * (2 * p_y)
        bbs.append([c, area, z, x, y, bb_s_z, bb_s_y, bb_s_x, bb_f_z, bb_f_y, bb_f_x])
    bbs = np.array(bbs)

    return bbs
