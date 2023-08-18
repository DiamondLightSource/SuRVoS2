import logging
import ntpath
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.api.annotations import (
    add_label,
    add_level,
    delete_all_labels,
    get_levels,
    rename_level,
    update_label,
)
from survos2.api.utils import pass_through, _unpack_lists
from fastapi import APIRouter, Body

pipelines = APIRouter()

def get_feature_from_id(feat_id):
    src = DataModel.g.dataset_uri(feat_id, group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Feature shape {src_dataset.shape}")
        feature = src_dataset[:]
    return feature

@pipelines.get("/train_multi_axis_cnn", response_model=None)
@save_metadata
def train_multi_axis_cnn(
    src: str = Body(),
    dst: str = Body(),
    workspace: list = Body(),
    anno_id: list = Body(),
    feature_id: list = Body(),
    multi_ax_train_params: dict = Body(),
) -> "CNN":
    # Unpack the list
    workspace = _unpack_lists(workspace)
    anno_id = _unpack_lists(anno_id)
    feature_id = _unpack_lists(feature_id)

    model_type = multi_ax_train_params["model_type"]
    encoder_type = multi_ax_train_params["encoder_type"]
    patience = int(multi_ax_train_params["patience"])
    loss_criterion = multi_ax_train_params["loss_criterion"]
    bce_dice_alpha = float(multi_ax_train_params["bce_dice_alpha"])
    bce_dice_beta = float(multi_ax_train_params["bce_dice_beta"])
    training_axes = multi_ax_train_params["training_axes"]
    logger.info(
        f"Train {model_type} on {training_axes} axis with {encoder_type} encoder using workspaces {workspace} annos {anno_id} and features {feature_id}"
    )
    from volume_segmantics.data import TrainingDataSlicer, get_settings_data
    from volume_segmantics.model import (
        VolSeg2DPredictionManager,
        VolSeg2dTrainer,
    )
    from volume_segmantics.utilities import Quality

    max_label_no = 0
    label_codes = None
    label_values = None
    model_train_label = "Deep Learning Training"
    training_settings_dict = cfg["volume_segmantics"]["train_settings"]
    settings = get_settings_data(training_settings_dict)
    settings.model["type"] = model_type
    settings.model["encoder_name"] = encoder_type
    settings.patience = patience
    settings.loss_criterion = loss_criterion
    settings.alpha = bce_dice_alpha
    settings.beta = bce_dice_beta
    settings.training_axes = training_axes
    current_ws = DataModel.g.current_workspace
    ws_object = ws.get(current_ws)
    data_out_path = Path(ws_object.path, "volseg")

    logger.info(f"Making output folders for slices in {data_out_path}")
    data_slice_path = data_out_path / settings.data_im_dirname
    anno_slice_path = data_out_path / settings.seg_im_out_dirname
    data_slice_path.mkdir(exist_ok=True, parents=True)
    anno_slice_path.mkdir(exist_ok=True, parents=True)

    # Get datsets from workspaces and slice to disk
    for count, (workspace_id, data_id, label_id) in enumerate(zip(workspace, feature_id, anno_id)):
        logger.info(f"Current workspace: {workspace_id}. Retrieving datasets.")
        DataModel.g.current_workspace = workspace_id
        src = DataModel.g.dataset_uri(data_id, group="features")
        feature = get_feature_from_id(data_id)
        src = DataModel.g.dataset_uri(label_id, group="annotations")
        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            labels = src_dataset.get_metadata("labels", {})
            anno_level = src_dataset[:] & 15
        anno_labels = np.unique(anno_level)
        logger.debug(
            f"Obtained annotation level with labels {anno_labels} and shape {anno_level.shape}"
        )
        slicer = TrainingDataSlicer(feature, anno_level, settings)
        data_prefix, label_prefix = f"data{count}", f"seg{count}"
        slicer.output_data_slices(data_slice_path, data_prefix)
        slicer.output_label_slices(anno_slice_path, label_prefix)
        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes = labels
            label_values = anno_labels

    logger.info(f"Creating model Trainer with label codes: {label_codes}")
    trainer = VolSeg2dTrainer(data_slice_path, anno_slice_path, label_codes, settings)
    num_cyc_frozen = multi_ax_train_params["cyc_frozen"]
    num_cyc_unfrozen = multi_ax_train_params["cyc_unfrozen"]
    model_type = settings.model["type"].name
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    model_fn = f"{dt_string}_{model_type}_{settings.model_output_fn}.pytorch"
    model_out = Path(data_out_path, model_fn)
    if num_cyc_frozen > 0:
        trainer.train_model(model_out, num_cyc_frozen, settings.patience, create=True, frozen=True)
    if num_cyc_unfrozen > 0 and num_cyc_frozen > 0:
        trainer.train_model(
            model_out, num_cyc_unfrozen, settings.patience, create=False, frozen=False
        )
    elif num_cyc_unfrozen > 0 and num_cyc_frozen == 0:
        trainer.train_model(
            model_out, num_cyc_unfrozen, settings.patience, create=True, frozen=False
        )
    trainer.output_loss_fig(model_out)
    trainer.output_prediction_figure(model_out)
    # Clean up all the saved slices
    slicer.clean_up_slices()

    # Load in data volume from current workspace
    DataModel.g.current_workspace = current_ws
    # TODO: Add a field in the GUI to choose which data to apply to
    levels = get_levels(current_ws)
    # If there is already a level for U-net prediction output, don't create a new one
    anno_exists = any([model_train_label in x["name"] for x in levels])
    if not anno_exists:
        logger.info("Creating new level for prediction.")
        level_result = add_level(current_ws)
    else:
        level_result = ([x for x in levels if model_train_label in x["name"]] or [None])[0]
        delete_all_labels(current_ws, level_result["id"])
    level_id = level_result["id"]
    logger.info(f"Using labels from level with ID {level_id}, changing level name.")
    rename_level(workspace=current_ws, level=level_id, name=model_train_label)
    create_new_labels_for_level(current_ws, level_id, label_codes)
    feature = get_feature_from_id("001_raw")
    # Now we need to predict a segmentation for the training volume
    # Load in the prediction settings
    predict_settings_dict = cfg["volume_segmantics"]["predict_settings"]
    predict_settings = get_settings_data(predict_settings_dict)
    pred_manager = VolSeg2DPredictionManager(model_out, feature, predict_settings)
    segmentation = pred_manager.predict_volume_to_path(None, Quality.LOW)
    # If more than one annotation label is provided add 1 to the segmentation
    if not np.array_equal(np.array([0, 1]), label_values):
        segmentation += np.ones_like(segmentation)

    map_blocks(pass_through, segmentation, out=dst, normalize=False)


@pipelines.get("/predict_multi_axis_cnn", response_model=None)
@save_metadata
def predict_multi_axis_cnn(
    src: str,
    dst: str,
    workspace: str,
    feature_id: str,
    model_path: str,
    no_of_planes: int,
    cuda_device: int,
    prediction_axis: str,
) -> "CNN":
    model_pred_label = "Deep Learning Prediction"
    if feature_id:
        logger.debug(f"Predict_multi_axis_cnn with feature {feature_id} in {no_of_planes} planes")

        src = DataModel.g.dataset_uri(feature_id, group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            logger.debug(f"Adding feature of shape {src_dataset.shape}")
            feature = src_dataset[:]

        logger.info(
            f"Predict_multi_axis_cnn with feature shape {feature.shape} using model {model_path}"
        )
    else:
        logging.error("No feature selected!")
        return

    levels = get_levels(workspace)
    # If there is already a level for U-net prediction output, don't create a new one
    anno_exists = any([model_pred_label in x["name"] for x in levels])
    logging.info(f"Previous prediction exists: {anno_exists}")
    if not anno_exists:
        level_result = add_level(workspace)
    else:
        level_result = ([x for x in levels if model_pred_label in x["name"]] or [None])[0]
        delete_all_labels(workspace, level_result["id"])
    if level_result:
        import torch
        from volume_segmantics.data import get_settings_data
        from volume_segmantics.model import VolSeg2DPredictionManager
        from volume_segmantics.utilities import Quality

        predict_settings_dict = cfg["volume_segmantics"]["predict_settings"]
        predict_settings = get_settings_data(predict_settings_dict)
        predict_settings.cuda_device = cuda_device
        predict_settings.prediction_axis = prediction_axis
        level_id = level_result["id"]
        logger.info(f"Using level with ID {level_id}, changing level name.")
        rename_level(workspace=workspace, level=level_id, name=model_pred_label)
        logger.info(f"Unpacking model and labels")
        ws_object = ws.get(workspace)
        root_path = Path(ws_object.path, "volseg")
        root_path.mkdir(exist_ok=True, parents=True)
        # Create prediction manager and get label codes
        pred_manager = VolSeg2DPredictionManager(model_path, feature, predict_settings)
        label_codes = pred_manager.get_label_codes()
        logger.info(f"Labels found: {label_codes}")
        create_new_labels_for_level(workspace, level_id, label_codes)
        quality = Quality(no_of_planes)
        segmentation = pred_manager.predict_volume_to_path(None, quality)
        segmentation += np.ones_like(segmentation)
        logger.info("Freeing GPU memory.")
        del pred_manager
        torch.cuda.empty_cache()

        map_blocks(pass_through, segmentation, out=dst, normalize=False)
    else:
        logging.error("Unable to add level for output!")


def create_new_labels_for_level(workspace, level_id, label_codes):
    # New style codes are in a dictionary
    if isinstance(label_codes, dict):
        for key in label_codes:
            label_result = add_label(workspace=workspace, level=level_id)
            if label_result:
                update_result = update_label(
                    workspace=workspace, level=level_id, **label_codes[key]
                )
                logger.info(f"Label created: {update_result}")
        # Old style codes are in a list
    elif isinstance(label_codes, list):
        r = lambda: random.randint(0, 255)
        for idx, code in enumerate(label_codes, start=2):
            label_result = add_label(workspace=workspace, level=level_id)
            if label_result:
                codes_dict = {
                    "color": f"#{r():02X}{r():02X}{r():02X}",
                    "idx": idx,
                    "name": code,
                    "visible": True,
                }
                update_result = update_label(workspace=workspace, level=level_id, **codes_dict)
                logger.info(f"Label created: {update_result}")
