"""
An object/entity is labeled geometric/vector data is stored in 
a dataframe used for object-based image analysis.

A basic Entity Dataframe has 'z','x','y','class_code'

An example use is to provide labeled patch volumes to classifier.

An entity workflow is initialized by the project file, which contains

1. Where the volume image file (hdf5) is
2. The path of the image data within the hdf5 file.
3. A csv of locations
4. Class names
5. Synthetic data generation parameters, such as the parameters for 
the blob or shell to generate for each class type.


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
from survos2.frontend.nb_utils import plot_slice_and_pts, summary_stats
from survos2.entity.sampler import viz_bvols, centroid_to_bvol, offset_points


def make_entity_mask(wf, dets, flipxy=True):
    from survos2.entity.instanceseg.utils import pad_vol

    offset_dets = offset_points(dets, (-64, -64, -64))
    offset_det_bvol = centroid_to_bvol(
        offset_dets, bvol_dim=(32, 32, 32), flipxy=flipxy
    )
    padded_vol = pad_vol(wf.vols[0], (64, 64, 64))
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


@dataclass
class EntityWorkflow:
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    bg_mask: np.ndarray
    params: dict


def init_entity_workflow(project_file, roi_name, plot_all=False):
    with open(project_file) as project_file:
        wparams = json.load(project_file)
    proj = wparams["proj"]

    if proj == "vf":
        original_data = h5py.File(
            os.path.join(wparams["input_dir"], wparams["vol_fname"]), "r"
        )
        ds = original_data[wparams["dataset_name"]]
        # wf1 = ds["workflow_1"]
        wf2 = ds[wparams["workflow_name"]]

        ds_export = original_data.get("data_export")
        # wf1_wrangled = ds_export["workflow1_wrangled_export"]
        vol_shape_x = wf2[0].shape[0]
        vol_shape_y = wf2[0].shape[1]
        vol_shape_z = len(wf2)
        img_volume = wf2
        print(f"Loaded image volume of shape {img_volume.shape}")

    if proj == "hunt":
        # fname = wparams['vol_fname']
        fname = wparams["vol_fname"]
        original_data = h5py.File(os.path.join(wparams["datasets_dir"], fname), "r")
        img_volume = original_data["data"][:]
        wf1 = img_volume

    print(f"Loaded image volume of shape {img_volume.shape}")

    workflow_name = wparams["workflow_name"]
    input_dir = wparams["input_dir"]
    out_dir = wparams["outdir"]
    torch_models_fullpath = wparams["torch_models_fullpath"]
    project_file = wparams["project_file"]
    entity_fpath = wparams["entity_fpath"]
    # entity_fnames = wparams["entity_fnames"]
    entity_fname = wparams["entity_fname"]
    datasets_dir = wparams["datasets_dir"]
    entities_offset = wparams["entities_offset"]
    offset = wparams["entities_offset"]
    entity_meta = wparams["entity_meta"]
    main_bv = wparams["main_bv"]
    bg_mask_fname = wparams["bg_mask_fname"]

    #
    # load object data
    #
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

    print(f"Loaded entities of shape {entities_df.shape}")
    #
    # Load bg mask
    #
    # with h5py.File(os.path.join(wparams["datasets_dir"],bg_mask_fname), "r") as hf:
    #    logger.debug(f"Loaded bg mask file with keys {hf.keys()}")

    bg_mask_fullname = os.path.join(wparams["datasets_dir"], bg_mask_fname)
    bg_mask_file = h5py.File(bg_mask_fullname, "r")
    print(bg_mask_fullname)
    bg_mask = bg_mask_file["mask"][:]

    #
    # Crop main volume
    #
    main_bv = calc_bounding_vols(main_bv)
    bb = main_bv[roi_name]["bb"]
    roi_name = "_".join(map(str, bb))
    logger.debug(roi_name)

    # precropped_wf1, precropped_pts = crop_vol_and_pts_bb(
    #    img_volume, entity_pts, bounding_box=bb, debug_verbose=True, offset=True
    # )
    precropped_wf2, precropped_pts = crop_vol_and_pts_bb(
        img_volume, entity_pts, bounding_box=bb, debug_verbose=True, offset=True
    )
    combined_clustered_pts, classwise_entities = organize_entities(
        precropped_wf2, precropped_pts, entity_meta, plot_all=plot_all
    )
    bg_mask_crop = sample_bvol(bg_mask, bb)
    print(
        f"Cropping background mask of shape {bg_mask.shape} with bounding box: {bb} to shape of {bg_mask_crop.shape}"
    )

    wf = EntityWorkflow(
        [precropped_wf2, precropped_wf2],
        combined_clustered_pts,
        classwise_entities,
        bg_mask_crop,
        wparams,
    )

    if plot_all:
        plt.figure(figsize=(15, 15))
        plt.imshow(wf.vols[0][0, :], cmap="gray")
        plt.title("Input volume")
        plot_slice_and_pts(wf.vols[1], wf.locs, None, (40, 200, 200))

    return wf


def organize_entities(
    img_vol, clustered_pts, entity_meta, flipxy=False, plot_all=False
):

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
            plt.figure(figsize=(9, 9))
            plt.imshow(img_vol[img_vol.shape[0] // 4, :], cmap="gray")
            plt.scatter(classwise_pts[:, 1], classwise_pts[:, 2], c="cyan")
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
