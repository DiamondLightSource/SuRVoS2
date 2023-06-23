
import logging
import ntpath
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.api.objects import get_entities
from survos2.api.utils import dataset_repr, get_function_api, save_metadata, pass_through, _unpack_lists
from survos2.api.workspace import auto_create_dataset
from survos2.entity.patches import (
    PatchWorkflow,
    make_patches,
    organize_entities,
    sample_images_and_labels,
    augment_and_save_dataset,
)
from survos2.entity.pipeline_ops import make_proposal
from survos2.entity.sampler import generate_random_points_in_volume
from survos2.entity.train import train_oneclass_detseg
from survos2.entity.utils import pad_vol
from survos2.entity.entities import make_entity_df
from survos2.frontend.components.entity import setup_entity_table
from survos2.frontend.nb_utils import slice_plot
from scipy.ndimage.morphology import binary_erosion
from survos2.entity.utils import pad_vol, get_largest_cc
from survos2.entity.entities import offset_points
from survos2.entity.patches import BoundingVolumeDataset
from survos2.entity.sampler import centroid_to_bvol
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.api.utils import pass_through, _unpack_lists
from fastapi import APIRouter, Body, Query

pipelines = APIRouter()




@pipelines.get("/train_3d_cnn", response_model=None)
@save_metadata
def train_3d_cnn(
    src: str = Body(),
    dst: str = Body(),
    workspace: list = Body(),
    anno_id: list = Body(),
    feature_id: list = Body(),
    objects_id: list = Body(),
    num_samples: int = Body(),  # 100,
    num_epochs: int = Body(),  # 10,
    num_augs: int = Body(),  # 0,
    patch_size: list = Body(),  # = 64,
    patch_overlap: list = Body(),  # 16
    fcn_type: str = Body(),  # "unet3d",
    bce_to_dice_weight: float = Body(),  # = 0.7,
    threshold: float = Body(),  # = 0.5,
    overlap_mode: str = Body(),  # "crop",
    cuda_device: int = Body(),  # 0,
    plot_figures: bool = Body(),
) -> "CNN":
    """3D CNN using eithe FPN or U-net architecture.

    Args:
        src (str): Source Pipeline URI.
        dst (str): Destination Pipeline URI.
        workspace (str): Workspace to use.
        anno_id (str): Annotation label image.
        feature_id (str): Feature to use for training.
        objects_id (str): Point entitites to use as the locations to sample patches for training.
        num_samples (int): If no object_id is provided, sample num_sample locations randomly.
        num_epochs (int): Number of training epochs.
        num_augs (int): Number of flip augmentations to use.
        patch_size (list, optional): Size of patches to sample Defaults to 64.
        patch_overlap (list, optional): Overlap of patches when predicting output volume. Defaults to 16.
        fcn_type (str, optional): Either FPN or Unet3d. Defaults to "unet3d".
        bce_to_dice_weight (float, optional): Balance between Binary Cross-Entropy and Dice for loss. Defaults to 0.7.
        threshold (float, optional): Final segmentation threshold value. Defaults to 0.5.

    """

    workspace = _unpack_lists(workspace)
    anno_id = _unpack_lists(anno_id)
    feature_id = _unpack_lists(feature_id)
    objects_id = _unpack_lists(objects_id)
    current_ws = DataModel.g.current_workspace

    list_image_vols = []
    list_label_vols = []

    # Get datsets from workspaces and sample patches
    for count, (workspace_id, feat_id, label_id, obj_id) in enumerate(
        zip(workspace, feature_id, anno_id, objects_id)
    ):
        logger.info(f"Current workspace: {workspace_id}. Retrieving datasets.")
        DataModel.g.current_workspace = workspace_id
        logger.info(f"Train_3d fcn using anno {anno_id} and feature {feature_id}")

        src = DataModel.g.dataset_uri(feat_id, group="features")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_array = DM.sources[0][:]

        src = DataModel.g.dataset_uri(label_id, group="annotations")
        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            anno_level = src_dataset[:] & 15

        objects_scale = 1.0
        entity_meta = {
            "0": {
                "name": "class1",
                "size": np.array((15, 15, 15)) * objects_scale,
                "core_radius": np.array((7, 7, 7)) * objects_scale,
            },
        }

        # point sampling either by generating random points or loading in a list of points
        padding = np.array(patch_size) // 2
        padded_vol = pad_vol(src_array, padding // 2)
        entity_arr = generate_random_points_in_volume(padded_vol, num_samples, padding * 4)

        if obj_id != "None":
            objects_src = DataModel.g.dataset_uri(obj_id, group="objects")
            result = get_entities(objects_src)
            entity_arr = decode_numpy(result)
            entity_arr = entity_arr[0:num_samples]

        combined_clustered_pts, classwise_entities = organize_entities(
            src_array, entity_arr, entity_meta, plot_all=plot_figures, flipxy=True
        )

        wparams = {}
        wparams["entities_offset"] = (0, 0, 0)
        wparams["entity_meta"] = entity_meta
        wparams["workflow_name"] = "Make_Patches"
        wparams["proj"] = DataModel.g.current_workspace
        wf = PatchWorkflow(
            [src_array],
            combined_clustered_pts,
            classwise_entities,
            src_array,
            wparams,
            combined_clustered_pts,
        )

        # generate patches
        logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")
        logger.debug(f"Making patches in path {src_dataset._path}")
        vol_num = 0
        outdir = src_dataset._path


        max_vols = -1
        img_vols, label_vols, mask_gt = sample_images_and_labels(
            wf,
            entity_arr,
            vol_num,
            (anno_level == 1) * 1.0,
            padding,
            num_augs,
            plot_figures,
            patch_size,
        )

        list_image_vols.append(img_vols)
        list_label_vols.append(label_vols)

    img_vols = np.concatenate(list_image_vols)
    label_vols = np.concatenate(list_label_vols)

    train_v_density = augment_and_save_dataset(
        img_vols, label_vols, wf, outdir, mask_gt, padding, num_augs, plot_figures, max_vols
    )

    # Load in data volume from current workspace
    DataModel.g.current_workspace = current_ws

    # setup model filename
    ws_object = ws.get(current_ws)
    data_out_path = Path(ws_object.path, "fcn")
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    model_fn = f"{dt_string}_trained_fcn_model"
    model_out = str(Path(data_out_path, model_fn).resolve())

    wf_params = {}
    wf_params["torch_models_fullpath"] = model_out
    logger.info(f"Saving fcn model to: {model_out}")

    model_type = fcn_type

    model_file = train_oneclass_detseg(
        train_v_density,
        None,
        wf_params,
        num_epochs=num_epochs,
        model_type=model_type,
        bce_weight=bce_to_dice_weight,
        gpu_id=cuda_device,
    )

    src = DataModel.g.dataset_uri(feature_id[0], group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_array = DM.sources[0][:]

    # use trained model to predict on original feature volume
    proposal = make_proposal(
        src_array,
        os.path.join(wf_params["torch_models_fullpath"], model_file),
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
        gpu_id=cuda_device,
    )

    proposal = proposal.numpy()

    print(f"Made proposal {proposal.shape}")

    if len(proposal.shape) == 4:
        proposal = proposal[0, :]
    # normalize volume and threshold
    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)
    thresholded = (proposal > threshold) * 1.0
    map_blocks(pass_through, thresholded, out=dst, normalize=False)

    # save logit map
    confidence = 1
    if confidence:
        dst = auto_create_dataset(
            DataModel.g.current_workspace, name="logit_map", group="features", dtype="float32"
        )
        dst.set_attr("kind", "raw")
        with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
            DM.out[:] = proposal.copy()


@pipelines.get("/predict_3d_cnn", response_model=None)
@save_metadata
def predict_3d_cnn(
    src: str,
    dst: str,
    workspace: str,
    anno_id: str,
    feature_id: str,
    model_fullname: str,
    patch_size: List[int] = Query(),  # 64,
    patch_overlap: List[int] = Query(),  #  8,
    threshold: float = 0.5,
    model_type: str = "unet3d",
    overlap_mode: str = "crop",
    cuda_device: int = 0,
) -> "CNN":
    """Predict a 3D CNN (U-net or FPN) using a previously trained model.

    Args:
        src (str): Source Pipeline URI.
        dst (str): Destination Pipeline URI.
        workspace (str): Workspace to use.
        anno_id (str): Annotation label image.
        feature_id (str): Feature for prediction.
        model_fullname (str): Full path and filename of pretrained model.
        patch_size (list, optional): Patch size for prediction. Defaults to 64.
        patch_overlap (list, optional): Patch overlap for prediction. Defaults to 8.
        threshold (float, optional): Threshold to binarize image with. Defaults to 0.5.
        model_type (str, optional): Unet3d or Feature Pyramid Network ("fpn3d") or V-Net. ("vnet") Defaults to "unet3d".
        overlap_mode (str, optional): Either "crop" or "average". Defaults to "crop".

    Returns:
        CNN: _description_
    """

    src = DataModel.g.dataset_uri(feature_id, group="features")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Adding feature of shape {src_dataset.shape}")

    proposal = make_proposal(
        src_dataset,
        model_fullname,
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
        gpu_id=cuda_device,
    )

    proposal = proposal.numpy()

    if len(proposal.shape) == 4:
        proposal = proposal[0, :]

    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)

    thresholded = (proposal > threshold) * 1.0
    map_blocks(pass_through, thresholded, out=dst, normalize=False)

    confidence = 1
    if confidence:
        dst = auto_create_dataset(
            DataModel.g.current_workspace, name="logit_map", group="features", dtype="float32"
        )
        dst.set_attr("kind", "raw")
        with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
            DM.out[:] = proposal.copy()
