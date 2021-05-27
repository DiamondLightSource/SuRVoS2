import ast
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pprint import pprint
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from skimage import data, measure
from sklearn.model_selection import train_test_split
from survos2 import survos
from survos2.entity.entities import (
    make_bounding_vols,
    make_entity_bvol,
    make_entity_df,
    calc_bounding_vols,
)

from survos2.entity.utils import get_largest_cc, get_surface, pad_vol
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts,
    crop_vol_and_pts_bb,
    offset_points,
    sample_bvol,
    sample_marked_patches,
    viz_bvols,
)
from survos2.frontend.nb_utils import (
    slice_plot,
    show_images,
    view_vols_labels,
    view_vols_points,
    view_volume,
    view_volumes,
)
from survos2.server.features import generate_features, prepare_prediction_features
from survos2.server.filtering import (
    gaussian_blur_kornia,
    ndimage_laplacian,
    spatial_gradient_3d,
)
from survos2.server.filtering.morph import dilate, erode, median
from survos2.server.model import SRData, SRFeatures
from survos2.server.pipeline import Patch, Pipeline
from survos2.server.pipeline_ops import (
    clean_segmentation,
    make_acwe,
    make_bb,
    make_features,
    make_masks,
    make_noop,
    make_sr,
    predict_and_agg,
    predict_sr,
    saliency_pipeline,
)
from survos2.server.state import cfg
from survos2.server.supervoxels import generate_supervoxels
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridAggregator, GridSampler
from torchvision import transforms
from tqdm import tqdm


@dataclass
class PatchWorkflow:
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    bg_mask: np.ndarray
    params: dict
    gold: np.ndarray


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
    gold_fname = wparams["gold_fname"]
    gold_fpath = wparams["gold_fpath"]

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

    #
    # Crop main volume
    #
    main_bv = calc_bounding_vols(main_bv)
    bb = main_bv[roi_name]["bb"]
    print(f"Main bounding box: {bb}")
    roi_name = "_".join(map(str, bb))

    logger.debug(roi_name)

    #
    # load gold data
    #
    gold_df = pd.read_csv(os.path.join(gold_fpath, gold_fname))
    gold_df.drop(
        gold_df.columns[gold_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    gold_pts = np.array(gold_df)
    # gold_df = make_entity_df(gold_pts, flipxy=True)
    # gold_pts = np.array(e_df)
    scale_z, scale_x, scale_y = 1.0, 1.0, 1.0
    gold_pts[:, 0] = (gold_pts[:, 0] * scale_z) + offset[0]
    gold_pts[:, 1] = (gold_pts[:, 1] * scale_x) + offset[1]
    gold_pts[:, 2] = (gold_pts[:, 2] * scale_y) + offset[2]

    # precropped_wf2, gold_pts = crop_vol_and_pts_bb(
    #     img_volume, gold_pts, bounding_box=bb, debug_verbose=True, offset=True
    # )

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
    #bg_mask_crop = bg_mask
    wf = PatchWorkflow(
        [precropped_wf2, precropped_wf2],
        combined_clustered_pts,
        classwise_entities,
        bg_mask_crop,
        wparams,
        gold_pts,
    )

    if plot_all:
        plt.figure(figsize=(15, 15))
        plt.imshow(wf.vols[0][0, :], cmap="gray")
        plt.title("Input volume")
        slice_plot(wf.vols[1], wf.locs, None, (40, 200, 200))

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


def load_patch_vols(train_vols):
    with h5py.File(train_vols[0], "r") as hf:
        print(hf.keys())
        img_vols = hf["data"][:]

    with h5py.File(train_vols[1], "r") as hf:
        print(hf.keys())
        label_vols = hf["data"][:]
    print(img_vols.shape, label_vols.shape)

    return img_vols, label_vols


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


