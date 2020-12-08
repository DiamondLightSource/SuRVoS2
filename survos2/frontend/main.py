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
from survos2.frontend.model import ClientData
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
            raise WorkspaceException(f"Internal HDF5 dataset: '{dataset_name}' does not exist!") from e
        

    logger.info(f"Loaded vol of size {img_volume.shape}")
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

    return survos.run_command(
        "workspace",
        "add_dataset",
        workspace=ws_name,
        dataset_name=dataset_name,
        dtype="float32",
    )


def init_client():
    survos.init_api()
    from survos2.model import Workspace

    # ws = Workspace(DataModel.g.current_workspace)
    # dataset_name = "__data__"
    # ds = ws.get_dataset(dataset_name)
    # img_volume = ds[:]
    # logger.debug(f"Image volume loaded: {img_volume.shape}")

    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]

    DataModel.g.current_workspace_shape = img_volume.shape

    logger.debug(f"DatasetManager loaded volume of shape {img_volume.shape}")

    filtered_layers = [np.array(img_volume).astype(np.float32)]
    layer_names = [
        "Main",
    ]
    opacities = [
        1.0,
    ]

    from survos2.server.config import cfg

    clientData = ClientData(filtered_layers, layer_names, opacities, cfg)
    return clientData


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


def setup_ws(project_file=None):
    with open(project_file) as project_file:
        workspace_params = json.load(project_file)
        workspace_params = AttrDict(workspace_params)
        clientData = init_client(workspace_params)

    return clientData


def start_client():
    clientData = init_client()
    viewer = frontend(clientData)


if __name__ == "__main__":
    start_client()
