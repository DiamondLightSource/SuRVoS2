"""
An entity is Labeled geometric/vector data is stored in 
a dataframe used for object-based image analysis.

The most basic Entity Dataframe has 'z','x','y','class_code'

An example use is to provide labeled patch volumes to classifier.

An Entity 
    Has a ROI
    Has Label(s)
    Has Optional Features
    Has Optional Measurements
        Simple measurement: Single "grade" for ROI
        Set of measurements:

anno.mask
supports converting entities into label volumes

anno.crowd
supports importing of data from zooniverse


"""

import itertools
import hdbscan
from collections import Counter
from statistics import mode, StatisticsError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import h5py
import time
import glob

import collections
import numpy as np
import pandas as pd
from typing import NamedTuple
import itertools
from dataclasses import dataclass


from scipy import ndimage
import torch.utils.data as data
from typing import List

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

from survos2.frontend.nb_utils import summary_stats
from numpy.linalg import LinAlgError

# warnings.filterwarnings("ignore")
# warnings.filterwarnings(action='once')

from survos2.entity.anno.geom import centroid_3d, rescale_3d
from dataclasses import dataclass

from survos2.entity.sampler import crop_vol_and_pts_bb
from survos2.frontend.nb_utils import plot_slice_and_pts


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




@dataclass
class EntityWorkflow:
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    params: dict


def init_entity_workflow(project_file,):
    with open(project_file) as project_file:
        wparams = json.load(project_file)

    proj = wparams["proj"]

    if proj == "vf":
        original_data = h5py.File(
            os.path.join(wparams["input_dir"], wparams["vol_fname"]), "r"
        )
        ds = original_data["dataset"]
        wf1 = ds["workflow_1"]
        wf2 = ds["workflow_2"]
        ds_export = original_data.get("data_export")
        wf1_wrangled = ds_export["workflow1_wrangled_export"]
        vol_shape_x = wf1[0].shape[0]
        vol_shape_y = wf1[0].shape[1]
        vol_shape_z = len(wf1)
        img_volume = wf2

    if proj == "hunt":
        # fname = wparams['vol_fname']
        fname = wparams["vol_fname"]
        original_data = h5py.File(os.path.join(wparams["datasets_dir"], fname), "r")
        img_volume = original_data["data"][:]

    workflow_name = wparams["workflow_name"]
    input_dir = wparams["input_dir"]
    out_dir = wparams["outdir"]
    torch_models_fullpath = wparams["torch_models_fullpath"]
    project_file = wparams["project_file"]
    entity_fpath = wparams["entity_fpath"]
    entity_fnames = wparams["entity_fnames"]
    entity_fname = wparams["entity_fname"]
    datasets_dir = wparams["datasets_dir"]
    entities_offset = wparams["entities_offset"]
    offset = wparams["entities_offset"]
    main_bv_vf = wparams["main_bv_vf"]
    entity_meta = wparams["entity_meta"]
    model_file = wparams["saliency_model_file"]
    roi_name = wparams["main_bv_name"]

    img_volume -= np.min(img_volume)
    img_volume = img_volume / np.max(img_volume)

    entities_df = pd.read_csv(os.path.join(entity_fpath, entity_fname))
    entities_df.drop(
        entities_df.columns[entities_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    entity_pts = np.array(entities_df)
    # e_df = make_entity_df(entity_pts, flipxy=True)
    # entity_pts = np.array(e_df)
    scale_z, scale_x, scale_y = 1.0, 1.0, 1.0
    entity_pts[:, 0] = (entity_pts[:, 0] * scale_z) + offset[0]
    entity_pts[:, 1] = (entity_pts[:, 1] * scale_x) + offset[1]
    entity_pts[:, 2] = (entity_pts[:, 2] * scale_y) + offset[2]

    if proj == "vf":
        main_bv = main_bv_vf
    elif proj == "hunt":
        main_bv = main_bv_hunt

    main_bv = calc_bounding_vols(main_bv)
    bb = main_bv[roi_name]["bb"]
    roi_name = "_".join(map(str, bb))
    print(roi_name)

    precropped_wf1, precropped_pts = crop_vol_and_pts_bb(
        img_volume, entity_pts, bounding_box=bb, debug_verbose=True, offset=True
    )
    precropped_wf2, precropped_pts = crop_vol_and_pts_bb(
        wf1, entity_pts, bounding_box=bb, debug_verbose=True, offset=True
    )

    # from survos2.entity.anno.point_cloud import chip_cluster
    # clustered_pts = chip_cluster(precropped_pts, precropped_wf1, 0, 0,
    #                                     min_cluster_size=2, method='hdbscan',eps=3,
    #                                     debug_verbose=True, plot_all=True)

    combined_clustered_pts, classwise_entities = organize_entities(
        precropped_wf1, precropped_pts, entity_meta
    )

    wf = EntityWorkflow(
        [precropped_wf1, precropped_wf2],
        combined_clustered_pts,
        classwise_entities,
        wparams,
    )

    plt.figure(figsize=(15, 15))
    plt.imshow(wf.vols[0][0, :], cmap="gray")
    plt.title("Input volume")
    plot_slice_and_pts(wf.vols[1], wf.locs, None, (40, 200, 200))

    return wf


def organize_entities(img_vol, clustered_pts, entity_meta, flipxy=False, plot_all=True):

    class_idxs = entity_meta.keys()
    classwise_entities = []

    for c in class_idxs:
        pt_idxs = clustered_pts[:, 3] == int(c)
        classwise_pts = clustered_pts[pt_idxs]
        clustered_df = make_entity_df(classwise_pts, flipxy=flipxy)
        classwise_pts = np.array(clustered_df)
        classwise_entities.append(classwise_pts)
        entity_meta[c]["entities"] = classwise_pts

        if plot_all:
            plt.figure(figsize=(12, 12))
            plt.imshow(img_vol[img_vol.shape[0] // 4, :], cmap="gray")
            plt.scatter(classwise_pts[:, 1], classwise_pts[:, 2])
            plt.title(
                str(entity_meta[c]["name"])
                + " Clustered Locations: "
                + str(len(classwise_pts))
            )

    combined_clustered_pts = np.concatenate(classwise_entities)

    return combined_clustered_pts, entity_meta


def offset_points(pts, patch_pos):
    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]

    print(f"Offset: {offset_x}, {offset_y}, {offset_z}")

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


def make_bounding_vols(ents, patch_size=(14, 14, 14)):
    p_z, p_x, p_y = patch_size

    bbs = []

    for z, x, y, c in ents:
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
