import os
import sys
import numpy as np
from loguru import logger
import h5py
import json
import time
from typing import List, Dict
from attrdict import AttrDict

from survos2.frontend.frontend import frontend

from survos2.entity.entities import make_entity_df
from survos2 import survos
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.entity.sampler import crop_vol_and_pts_centered
from survos2.model.workspace import WorkspaceException


def preprocess(img_volume):
    img_volume = np.array(img_volume).astype(np.float32)
    img_volume = np.nan_to_num(img_volume)
    img_volume = img_volume - np.min(img_volume)
    img_volume = img_volume / np.max(img_volume)

    return img_volume


def init_ws(workspace_params):
    ws_name = workspace_params["workspace_name"]
    dataset_name = workspace_params["dataset_name"]
    datasets_dir = workspace_params["datasets_dir"]
    fname = workspace_params["vol_fname"]

    image_path = os.path.join(datasets_dir, fname)
    logger.info(f"Initialising workspace {ws_name} with image volume {image_path}")

    original_data = h5py.File(image_path, "r")

    if "group_name" in workspace_params:
        group_name = workspace_params["group_name"]
        logger.info("Extracting dataset and then group")
        img_volume = original_data[dataset_name]
        img_volume = img_volume[group_name]
    else:
        logger.info("Extracting dataset")
        try:
            img_volume = original_data[dataset_name]
        except KeyError as e:
            raise WorkspaceException(
                f"Internal HDF5 dataset: '{dataset_name}' does not exist!"
            ) from e

    logger.info(f"Loaded vol of size {img_volume.shape}")
    if "roi_limits" in workspace_params:
        x_start, x_end, y_start, y_end, z_start, z_end = map(
            int, workspace_params["roi_limits"]
        )
        logger.info(
            f"Cropping data to predefined ROI z:{z_start}-{z_end},"
            f"y:{y_start}-{y_end}, x:{x_start}-{x_end}"
        )
        img_volume = img_volume[z_start:z_end, y_start:y_end, x_start:x_end]
    img_volume = preprocess(img_volume)

    if "precrop_coords" in workspace_params:
        precrop_coords = workspace_params["precrop_coords"]
        if "precrop_vol_size" in workspace_params:
            precrop_vol_size = workspace_params["precrop_vol_size"]

            if workspace_params["entities_name"] is not None:
                entities_name = workspace_params["entities_name"]

            img_volume, entities_df = precrop(
                img_volume, entities_df, precrop_coords, precrop_vol_size
            )

    if "downsample_by" in workspace_params:
        downby = int(workspace_params["downsample_by"])
        logger.info(f"Downsampling data by a factor of {downby}")
        img_volume = img_volume[::downby, ::downby, ::downby]

    tmpvol_fullpath = "tmp\\tmpvol.h5"

    with h5py.File(tmpvol_fullpath, "w") as hf:
        hf.create_dataset("data", data=img_volume)

    survos.run_command("workspace", "create", workspace=ws_name)
    logger.info(f"Created workspace {ws_name}")

    survos.run_command(
        "workspace",
        "add_data",
        workspace=ws_name,
        data_fname=tmpvol_fullpath,
        dtype="float32",
    )

    logger.info(f"Added data to workspace from {os.path.join(datasets_dir, fname)}")

    response = survos.run_command(
        "workspace",
        "add_dataset",
        workspace=ws_name,
        dataset_name=dataset_name,
        dtype="float32",
    )

    DataModel.g.current_workspace = ws_name
    survos.run_command(
        "features", "create", uri=None, workspace=ws_name, feature_type="raw"
    )
    src = DataModel.g.dataset_uri("__data__", None)
    dst = DataModel.g.dataset_uri("001_raw", group="features")
    with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
        print(DM.sources[0].shape)
        orig_dataset = DM.sources[0]
        dst_dataset = DM.out
        src_arr = orig_dataset[:]
        dst_dataset[:] = src_arr

    return response



def roi_ws(img_volume, ws_name):
    tmpvol_fullpath = "tmp\\tmpvol.h5"

    with h5py.File(tmpvol_fullpath, "w") as hf:
        hf.create_dataset("data", data=img_volume)

    survos.run_command("workspace", "create", workspace=ws_name)
    logger.info(f"Created workspace {ws_name}")

    survos.run_command(
        "workspace",
        "add_data",
        workspace=ws_name,
        data_fname=tmpvol_fullpath,
        dtype="float32",
    )

    response = survos.run_command(
        "workspace",
        "add_dataset",
        workspace=ws_name,
        dataset_name=ws_name + "_dataset",
        dtype="float32",
    )

    DataModel.g.current_workspace = ws_name

    survos.run_command(
        "features", "create", uri=None, workspace=ws_name, feature_type="raw"
    )
    src = DataModel.g.dataset_uri("__data__", None)
    dst = DataModel.g.dataset_uri("001_raw", group="features")
    with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
        print(DM.sources[0].shape)
        orig_dataset = DM.sources[0]
        dst_dataset = DM.out
        src_arr = orig_dataset[:]
        dst_dataset[:] = src_arr

    return response


def precrop(img_volume, entities_df, precrop_coord, precrop_vol_size):
    """
    View a ROI from a big volume by creating a temp dataset from a crop.
    Crop both the volume and the associated entities.
    Used for big volumes tha never get loaded into viewer.
    """

    logger.info(f"Preprocess cropping at {precrop_coord} to {precrop_vol_size}")
    img_volume, precropped_pts = crop_vol_and_pts_centered(
        img_volume,
        np.array(entities_df),
        location=precrop_coord,
        patch_size=precrop_vol_size,
        debug_verbose=True,
        offset=True,
    )

    entities_df = make_entity_df(precropped_pts, flipxy=False)
    return img_volume, entities_df

def start_client():
    frontend()
