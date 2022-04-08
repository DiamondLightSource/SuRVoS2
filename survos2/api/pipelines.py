from enum import auto
import logging
import ntpath
import os
from typing import List
from dataclasses import dataclass
from datetime import datetime
import random
import ast


import dask.array as da
import hug
#from numba.core.types.scalars import Integer
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger
from scipy import ndimage
from skimage.morphology.selem import octahedron
from pathlib import Path

from survos2.api import workspace as ws
from survos2.api.types import (
    DataURI,
    DataURIList,
    Float,
    FloatList,
    FloatOrVector,
    IntOrVector,
    Int,
    IntList,
    SmartBoolean,
    String,
    StringList
)
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.utils import decode_numpy    
from survos2.api.objects import get_entities

from survos2.entity.anno.pseudo import make_pseudomasks

from survos2.api.workspace import add_dataset, auto_create_dataset
from survos2.api.annotations import (add_level, rename_level, get_levels,
                                     add_label, update_label, delete_all_labels)
from survos2.entity.patches import PatchWorkflow, organize_entities, make_patches
from survos2.entity.pipeline_ops import make_proposal

from survos2.entity.sampler import grid_of_points, generate_random_points_in_volume
from survos2.entity.utils import pad_vol
from survos2.entity.train import train_oneclass_detseg
from survos2.entity.pipeline_ops import make_proposal

__pipeline_group__ = "pipelines"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0


def pass_through(x):
    return x

@dataclass
class PatchWorkflow:
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    bg_mask: np.ndarray
    params: dict
    gold: np.ndarray


@hug.get()
def cleaning(
    # object_id : DataURI,
    feature_id: DataURI,
    dst: DataURI,
    min_component_size: Int = 100,
):
    from survos2.entity.saliency import (
        single_component_cleaning,
        filter_small_components,
    )

    # src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    # logger.debug(f"Getting objects {src}")
    # with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
    #     ds_objects = DM.sources[0]
    # entities_fullname = ds_objects.get_metadata("fullname")
    # tabledata, entities_df = setup_entity_table(entities_fullname)
    # selected_entities = np.array(entities_df)

    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature_dataset_arr = DM.sources[0][:]

    seg_cleaned, tables, labeled_images = filter_small_components(
        [feature_dataset_arr], min_component_size=min_component_size
    )
    # seg_cleaned = single_component_cleaning(selected_entities, feature_dataset_arr, bvol_dim=(42,42,42))

    # map_blocks(pass_through, (seg_cleaned > 0) * 1.0, out=dst, normalize=False)

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = (seg_cleaned > 0) * 1.0


@hug.get()
def feature_postprocess(
    feature_A: DataURI,
    feature_B: DataURI,
    dst: DataURI,
):
    src_A = DataModel.g.dataset_uri(feature_A, group="features")
    with DatasetManager(src_A, out=None, dtype="uint16", fillvalue=0) as DM:
        src_A_dataset = DM.sources[0]
        src_A_arr = src_A_dataset[:]
        logger.info(f"Obtained src A with shape {src_A_arr.shape}")

    src_B = DataModel.g.dataset_uri(feature_B, group="features")    
    with DatasetManager(src_B, out=None, dtype="uint16", fillvalue=0) as DM:
        src_B_dataset = DM.sources[0]
        src_B_arr = src_B_dataset[:]
        logger.info(f"Obtained src B with shape {src_B_arr.shape}")
    
    result = src_A_arr * src_B_arr        
    map_blocks(pass_through, result, out=dst, normalize=False)


@hug.get()
def label_postprocess(
    level_over: DataURI,
    level_base: DataURI,
    selected_label: Int,
    offset: Int,
    dst: DataURI,
):
    if level_over != 'None':
        src1 = DataModel.g.dataset_uri(level_over, group="annotations")
        with DatasetManager(src1, out=None, dtype="uint16", fillvalue=0) as DM:
            src1_dataset = DM.sources[0]
            anno1_level = src1_dataset[:] & 15
            logger.info(f"Obtained over annotation level with labels {np.unique(anno1_level)}")

    src_base = DataModel.g.dataset_uri(level_base, group="annotations")
    with DatasetManager(src_base, out=None, dtype="uint16", fillvalue=0) as DM:
        src_base_dataset = DM.sources[0]
        anno_base_level = src_base_dataset[:] & 15
        logger.info(f"Obtained base annotation level with labels {np.unique(anno_base_level)}")

    print(f"Selected label {selected_label}")
    result = anno_base_level
    
    if level_over != 'None':
        result = anno_base_level * (1.0 - ((anno1_level > 0) * 1.0))
        anno1_level[anno1_level == selected_label] += offset

        result += anno1_level
        
    map_blocks(pass_through, result, out=dst, normalize=False)


@hug.get()
def watershed(src: DataURI, anno_id: DataURI, dst: DataURI):
    from ..server.filtering import watershed

    # get marker anno
    anno_uri = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(anno_uri, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_level = src_dataset[:] & 15
        logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

    logger.debug(f"Calculating watershed")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    filtered = watershed(src_dataset_arr, anno_level)

    dst = DataModel.g.dataset_uri(dst, group="pipelines")

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = filtered


@hug.get()
@save_metadata
def rasterize_points(
    src: DataURI,
    dst: DataURI,
    workspace: String,
    feature_id: DataURI,
    object_id: DataURI,
    acwe: SmartBoolean,
    size: FloatOrVector,
    balloon: Float,
    threshold: Float,
    iterations: Int,
    smoothing: Int,
):

    from survos2.entity.anno.pseudo import (
        organize_entities,
    )

    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src}")

    features = []
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Adding feature of shape {src_dataset.shape}")
        features.append(src_dataset[:])

    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]

    objects_fullname = ds_objects.get_metadata("fullname")
    objects_scale = ds_objects.get_metadata("scale")
    objects_offset = ds_objects.get_metadata("offset")
    objects_crop_start = ds_objects.get_metadata("crop_start")
    objects_crop_end = ds_objects.get_metadata("crop_end")

    logger.debug(f"Getting objects from {src} and file {objects_fullname} with scale {objects_scale}")
    from survos2.frontend.components.entity import make_entity_df, setup_entity_table

    tabledata, entities_df = setup_entity_table(
        objects_fullname,
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
        flipxy=False,
    )

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=False))

    # default params TODO make generic, allow editing
    entity_meta = {
        "0": {
            "name": "class1",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
    }

    combined_clustered_pts, classwise_entities = organize_entities(
        features[0], entities, entity_meta, plot_all=False
    )

    wparams = {}
    wparams["entities_offset"] = (0, 0, 0)


    wf = PatchWorkflow(
        features,
        combined_clustered_pts,
        classwise_entities,
        features[0],
        wparams,
        combined_clustered_pts,
    )

    combined_clustered_pts, classwise_entities = organize_entities(
        wf.vols[0], wf.locs, entity_meta
    )

    wf.params["entity_meta"] = entity_meta

    anno_masks, anno_acwe = make_pseudomasks(
        wf,
        classwise_entities,
        acwe=acwe,
        padding=(128, 128, 128),
        core_mask_radius=size,
        balloon=balloon,
        threshold=threshold,
        iterations=iterations,
        smoothing=smoothing,
    )

    if acwe:
        combined_anno = anno_acwe["0"]
    else:
        combined_anno = anno_masks["0"]["mask"]

    combined_anno = (combined_anno > 0.1) * 1.0

    # store in dst
    # dst = DataModel.g.dataset_uri(dst, group="pipelines")

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = combined_anno


@hug.get()
@save_metadata
def superregion_segment(
    src: DataURI,
    dst: DataURI,
    workspace: String,
    anno_id: DataURI,
    constrain_mask: DataURI,
    region_id: DataURI,
    feature_ids: DataURIList,
    lam: float,
    refine: SmartBoolean,
    classifier_type: String,
    projection_type: String,
    classifier_params: dict,
    confidence : bool
):
    logger.debug(
        f"superregion_segment using anno {anno_id} and superregions {region_id} and features {feature_ids}"
    )

    # get anno
    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_level = src_dataset[:] & 15

        logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

    # get superregions
    src = DataModel.g.dataset_uri(region_id, group="superregions")
    with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        supervoxel_image = src_dataset[:]

    # get features
    features = []

    for feature_id in feature_ids:
        src = DataModel.g.dataset_uri(feature_id, group="features")
        logger.debug(f"Getting features {src}")

        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            logger.debug(f"Adding feature of shape {src_dataset.shape}")
            features.append(src_dataset[:])

    logger.debug(
        f"sr_predict with {len(features)} features and anno of shape {anno_level.shape} and sr of shape {supervoxel_image.shape}"
    )

    # run predictions
    from survos2.server.superseg import sr_predict

    superseg_cfg = cfg.pipeline
    superseg_cfg["predict_params"] = classifier_params
    superseg_cfg["predict_params"]["clf"] = classifier_type
    superseg_cfg["predict_params"]["type"] = classifier_params["type"]
    superseg_cfg["predict_params"]["proj"] = projection_type
    logger.debug(f"Using superseg_cfg {superseg_cfg}")

    if constrain_mask != "None":
        import ast

        constrain_mask = ast.literal_eval(constrain_mask)
        print(constrain_mask)
        constrain_mask_id = ntpath.basename(constrain_mask["level"])
        label_idx = constrain_mask["idx"]

        src = DataModel.g.dataset_uri(constrain_mask_id, group="annotations")

        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            constrain_mask_level = src_dataset[:] & 15

        logger.debug(
            f"Constrain mask {constrain_mask_id}, label {label_idx} level shape {constrain_mask_level.shape} with unique labels {np.unique(constrain_mask_level)}"
        )

        mask = constrain_mask_level == label_idx - 1

    else:
        mask = None

    segmentation, conf_map = sr_predict(
        supervoxel_image,
        anno_level,
        features,
        mask,
        superseg_cfg,
        refine,
        lam,
    )
    conf_map = conf_map[:,:,:,1]
    logger.info(f"Obtained conf map of shape {conf_map.shape}")

    def pass_through(x):
        return x

    map_blocks(pass_through, segmentation, out=dst, normalize=False)

    if confidence:
        dst = auto_create_dataset(DataModel.g.current_workspace,name="confidence_map", group="features",  dtype="float32")
        dst.set_attr("kind", "raw")
        with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
            DM.out[:] = conf_map



def get_feature_from_id(feat_id):
    src = DataModel.g.dataset_uri(feat_id, group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Feature shape {src_dataset.shape}")
        feature = src_dataset[:]
    return feature

def _unpack_lists(input_list):
    """Unpacks a list of strings containing sublists of strings
    such as provided by the data table in the 2d U-net training pipeline card

    Args:
        input_list (list): list of strings which contain sublists

    Returns:
        list: list of hidden items from the table (first member in sublist)
    """
    return [ast.literal_eval(item)[0] for item in input_list]

@hug.get()
@save_metadata
def train_2d_unet(
    src: DataURI,
    dst: DataURI,
    workspace: StringList,
    anno_id: DataURIList,
    feature_id: DataURIList,
    unet_train_params: dict
):
    # Unpack the list
    workspace = _unpack_lists(workspace)
    anno_id = _unpack_lists(anno_id)
    feature_id = _unpack_lists(feature_id)
    logger.info(
        f"Train_2d_unet using workspaces {workspace} annos {anno_id} and features {feature_id}"
    )
    from survos2.server.unet2d.data_utils import TrainingDataSlicer
    from survos2.server.unet2d.unet2d import Unet2dTrainer
    max_label_no = 0
    label_codes = None
    label_values = None
    unet_train_label = "U-Net Training"
    current_ws = DataModel.g.current_workspace
    ws_object = ws.get(current_ws)
    data_out_path = Path(ws_object.path, "unet2d")
    logger.info(f"Making output folders for slices in {data_out_path}")
    data_slice_path = data_out_path / "data"
    anno_slice_path = data_out_path / "seg"
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
        logger.debug(f"Obtained annotation level with labels {anno_labels} and shape {anno_level.shape}")
        slicer = TrainingDataSlicer(feature, anno_level, clip_data=True)
        data_prefix, label_prefix = f"data{count}", f"seg{count}"
        slicer.output_data_slices(data_slice_path, data_prefix)
        slicer.output_label_slices(anno_slice_path, label_prefix)
        if slicer.num_seg_classes > max_label_no:
            max_label_no = slicer.num_seg_classes
            label_codes = labels
            label_values = anno_labels
    # Create the trainer and pass in the dictionary of label metadata
    logger.info(f"Creating U-Net Trainer with label codes: {label_codes}")
    trainer = Unet2dTrainer(data_slice_path, anno_slice_path, label_codes)
    cyc_frozen = unet_train_params["cyc_frozen"]
    cyc_unfrozen = unet_train_params["cyc_unfrozen"]
    trainer.train_model(num_cyc_frozen=cyc_frozen, num_cyc_unfrozen=cyc_unfrozen)
    # Save the model
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    model_fn = f"{dt_string}_trained_2dUnet_model"
    model_out = Path(data_out_path, model_fn)
    trainer.save_model_weights(model_out)
    # Save a figure showing the predictions
    trainer.output_prediction_figure(model_out)
    # Clean up all the saved slices
    slicer.clean_up_slices(dt_string)
    # Load in data volume from current workspace
    DataModel.g.current_workspace = current_ws
    # TODO: Add a field in the GUI to choose which data to apply to
    levels = get_levels(current_ws)
    # If there is already a level for U-net prediction output, don't create a new one
    anno_exists = any([unet_train_label in x["name"] for x in levels])
    if not anno_exists:
        logger.info("Creating new level for prediction.")
        level_result = add_level(current_ws)
    else:
        level_result = ([x for x in levels if unet_train_label in x["name"]] or [None])[0]
        delete_all_labels(current_ws, level_result["id"])
    level_id = level_result["id"]
    logger.info(f"Using labels from level with ID {level_id}, changing level name.")
    rename_level(workspace=current_ws, level=level_id, name=unet_train_label)
    create_new_labels_for_level(current_ws, level_id, label_codes)
    feature = get_feature_from_id("001_raw")
    slicer = TrainingDataSlicer(feature, anno_level, clip_data=True)
    segmentation = trainer.return_fast_prediction_volume(slicer.data_vol)
    # If more than one annotation label is provided add 1 to the segmentation
    if not np.array_equal(np.array([0, 1]), label_values):
        segmentation += np.ones_like(segmentation)

    def pass_through(x):
        return x

    map_blocks(pass_through, segmentation, out=dst, normalize=False)

@hug.get()
@save_metadata
def predict_2d_unet(
    src: DataURI,
    dst: DataURI,
    workspace: String,
    feature_id: DataURI,
    model_path: str,
    no_of_planes: int
):
    unet_pred_label = "U-Net prediction"
    if feature_id:
        logger.debug(
            f"Predict_2d_unet with feature {feature_id} in {no_of_planes} planes"
        )

        src = DataModel.g.dataset_uri(feature_id, group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            logger.debug(f"Adding feature of shape {src_dataset.shape}")
            feature = src_dataset[:]

        logger.info(
            f"Predict_2d_unet with feature shape {feature.shape} using model {model_path}"
        )
    else:
        logging.error("No feature selected!")
        return 
    
    levels = get_levels(workspace)
    # If there is already a level for U-net prediction output, don't create a new one
    anno_exists = any([unet_pred_label in x["name"] for x in levels])
    logging.info(f"Previous U-Net prediction exists: {anno_exists}")
    if not anno_exists:
        level_result = add_level(workspace)
    else:
        level_result = ([x for x in levels if unet_pred_label in x["name"]] or [None])[0]
        delete_all_labels(workspace, level_result["id"])
    if level_result:
        import torch
        from survos2.server.unet2d.unet2d import Unet2dPredictor
        from survos2.server.unet2d.data_utils import PredictionHDF5DataSlicer
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        level_id = level_result["id"]
        logger.info(f"Using level with ID {level_id}, changing level name.")
        rename_level(workspace=workspace, level=level_id, name=unet_pred_label)
        logger.info(f"Unpacking model and labels")
        ws_object = ws.get(workspace)
        root_path = Path(ws_object.path, "unet2d")
        root_path.mkdir(exist_ok=True, parents=True)
        predictor = Unet2dPredictor(root_path)
        label_codes = predictor.create_model_from_zip(Path(model_path))
        logger.info(f"Labels found: {label_codes}")
        create_new_labels_for_level(workspace, level_id, label_codes)
        slicer = PredictionHDF5DataSlicer(predictor, feature, clip_data=True)
        if no_of_planes == 1:
            segmentation = slicer.predict_1_way(root_path, output_prefix=dt_string)
        elif no_of_planes == 3:
            segmentation = slicer.predict_3_ways(root_path, output_prefix=dt_string)
        segmentation += np.ones_like(segmentation)
        logger.info("Freeing GPU memory.")
        del slicer
        del predictor
        torch.cuda.empty_cache()

        def pass_through(x):
            return x

        map_blocks(pass_through, segmentation, out=dst, normalize=False)
    else:
        logging.error("Unable to add level for output!")

def create_new_labels_for_level(workspace, level_id, label_codes):
    # New style codes are in a dictionary
    if isinstance(label_codes, dict):
        for key in label_codes:
            label_result = add_label(workspace=workspace,level=level_id)
            if label_result:
                update_result = update_label(workspace=workspace, level=level_id, **label_codes[key])
                logger.info(f"Label created: {update_result}")
        # Old style codes are in a list
    elif isinstance(label_codes, list):
        r = lambda: random.randint(0,255)
        for idx, code in enumerate(label_codes, start=2):
            label_result = add_label(workspace=workspace,level=level_id)
            if label_result:
                codes_dict = {"color": f"#{r():02X}{r():02X}{r():02X}", "idx": idx, "name": code, "visible": True}
                update_result = update_label(workspace=workspace, level=level_id, **codes_dict)
                logger.info(f"Label created: {update_result}")


@hug.get()
@save_metadata
def train_3d_fcn(
    src: DataURI,
    dst: DataURI,
    workspace: String,
    anno_id: DataURI,
    feature_id: DataURI,
    objects_id: DataURI,
    fpn_train_params: dict,
    num_samples: Int,
    num_epochs : Int,
    num_augs : Int,
    padding: IntOrVector = 32,
    grid_dim: IntOrVector = 4,
    patch_size: IntOrVector = 64,
    patch_overlap: IntOrVector = 16,
    fcn_type: String = 'fpn3d'
):
    logger.debug(
        f"Train_3d fcn using anno {anno_id} and feature {feature_id}"
    )

    src = DataModel.g.dataset_uri(feature_id, group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_array = DM.sources[0][:]

    objects_scale = 1.0
    entity_meta = {
        "0": {
            "name": "class1",
            "size": np.array((15, 15, 15)) * objects_scale,
            "core_radius": np.array((7, 7, 7)) * objects_scale,
        },
    }

    
    padded_vol = pad_vol(src_array, padding )
    
    entity_arr = generate_random_points_in_volume(padded_vol, num_samples, padding)
    
    if objects_id != "None":    
        objects_src = DataModel.g.dataset_uri(objects_id, group="objects")
        result = get_entities(objects_src)
        entity_arr = decode_numpy(result)

    combined_clustered_pts, classwise_entities = organize_entities(
        src_array, entity_arr, entity_meta, plot_all=False
    )

    wparams = {}
    wparams["entities_offset"] = (0, 0, 0)
    wparams["entity_meta"] = entity_meta
    wparams["workflow_name"] = "Make_Patches"
    wparams["proj"] = DataModel.g.current_workspace
    wf = PatchWorkflow(
        [src_array], combined_clustered_pts, classwise_entities, src_array, wparams, combined_clustered_pts
    )

    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_level = src_dataset[:] & 15

    logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")
    logger.debug(f"Making patches in path {src_dataset._path}")
    train_v_density = make_patches(wf, entity_arr, src_dataset._path, 
        proposal_vol=(anno_level == 1)* 1.0, 
        padding=padding, num_augs=num_augs, max_vols=-1)

    ws_object = ws.get(workspace)
    data_out_path = Path(ws_object.path, "fcn")
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    model_fn = f"{dt_string}_trained_fcn_model"
    model_out = str(Path(data_out_path, model_fn).resolve())
    
    wf_params = {}
    wf_params["torch_models_fullpath"] = model_out
    logger.info(f"Saving fcn model to: {model_out}")

    overlap_mode="crop"
    model_type=fcn_type

    model_file = train_oneclass_detseg(train_v_density, None, wf_params, num_epochs=num_epochs, model_type=model_type)
    
    src = DataModel.g.dataset_uri(feature_id, group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_array = DM.sources[0][:]
    
    proposal = make_proposal(
    src_array,
    os.path.join(wf_params["torch_models_fullpath"],model_file),
    model_type=model_type,
    patch_size=patch_size,
    patch_overlap=patch_overlap,
    overlap_mode=overlap_mode,
    )

    final_seg = (proposal > 0) * 1.0
    map_blocks(pass_through, final_seg, out=dst, normalize=False)
    

@hug.get()
def predict_3d_fcn(
    feature_id: DataURI,
    model_fullname: String,
    dst: DataURI,
    patch_size: IntOrVector = 64,
    patch_overlap: IntOrVector = 8,
    threshold: Float = 0.5,
    model_type: String = "unet3d",
    overlap_mode: String = "crop"
):
    
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
    )

    if model_type=='unet3d':
        proposal -= np.min(proposal)
        proposal = proposal / np.max(proposal)
    
    #
    #    proposal = 1.0 - proposal
    #proposal = ((proposal < threshold) * 1) + 1
    
    proposal = (proposal > threshold) * 1.0

    # store resulting segmentation in dst
    dst = DataModel.g.dataset_uri(dst, group="pipelines")
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = proposal


@hug.get()
def predict_segmentation_fcn(
    feature_id: DataURI,
    model_fullname: String,
    dst: DataURI,
    patch_size: IntOrVector = 64,
    patch_overlap: IntOrVector = 8,
    threshold: Float = 0.5,
    model_type: String = "unet3d",
):
    from survos2.entity.pipeline_ops import make_proposal

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
    )

    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)
    proposal = ((proposal < threshold) * 1) + 1

    # store resulting segmentation in dst
    dst = DataModel.g.dataset_uri(dst, group="pipelines")
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = proposal

from survos2.frontend.nb_utils import slice_plot
from scipy.ndimage.morphology import binary_erosion
from survos2.entity.utils import pad_vol, get_largest_cc, get_surface
from survos2.entity.sampler import centroid_to_bvol
from survos2.frontend.components.entity import setup_entity_table
from survos2.entity.patches import BoundingVolumeDataset
from survos2.entity.entities import make_entity_bvol, make_bounding_vols, make_entity_df
from survos2.entity.entities import offset_points, get_entities

from matplotlib import pyplot as plt


@hug.get()
def per_object_cleaning(
    dst: DataURI,
    feature_id: DataURI, 
    object_id: DataURI,
):
    """_summary_

    Parameters
    ----------
    feature_id : DataURI
        _description_
    object_id : DataURI
        _description_
    """

    # get image feature
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature = DM.sources[0][:]
        logger.debug(f"Feature shape {feature.shape}")

    # get object entities
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=False)
    entities_arr = np.array(entities_df)
    entities_arr[:,3] = np.array([[1] * len(entities_arr)])
    entities = np.array(make_entity_df(entities_arr, flipxy=False))
    print(entities)

    target = per_object_cleaning(entities, feature, display_plots=False)
    #dst = DataModel.g.dataset_uri(dst, group="pipelines")
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = target



def per_object_cleaning(entities, seg, bvol_dim=(32,32,32), offset=(0,0,0), flipxy=True, display_plots=False):    
    patch_size = tuple(1 * np.array(bvol_dim))
    seg = pad_vol(seg, np.array(patch_size))
    target = np.zeros_like(seg)
    
    entities = np.array(make_entity_df(np.array(entities), flipxy=flipxy))
    entities = offset_points(entities, offset)
    entities = offset_points(entities, -np.array(bvol_dim))

    if display_plots:    
        slice_plot(seg, np.array(make_entity_df(entities, flipxy=flipxy)), seg, (60,300,300))
    bvol = centroid_to_bvol(np.array(entities), bvol_dim=bvol_dim)
    bvol_seg = BoundingVolumeDataset(seg, bvol, labels=[1] * len(bvol), patch_size=patch_size)
    c = bvol_dim[0], bvol_dim[1], bvol_dim[2]
    
    for i, p in enumerate(bvol_seg):
        print(bvol_seg.bvols[i])
        seg, _ = bvol_seg[i]
        cleaned_patch = (get_largest_cc(seg) > 0) * 1.0
        # try:
        #     res = get_surface((cleaned_patch > 0) * 1.0, plot3d=False)
        mask = ((binary_erosion((cleaned_patch), iterations=1) > 0) * 1.0)
        
        if display_plots:
            plt.figure()
            plt.imshow(seg[c[0],:])
            plt.figure()
            plt.imshow(mask[c[0],:])
            
        
        z_st, y_st, x_st, z_end, y_end, x_end = bvol_seg.bvols[i]
        target[z_st:z_end, x_st:x_end, y_st:y_end,] = mask

    target = target[patch_size[0]:target.shape[0]-patch_size[0],
                    patch_size[1]:target.shape[1]-patch_size[1],
                    patch_size[2]:target.shape[2]-patch_size[2]]
        
    return target




@hug.get()
def create(workspace: String, pipeline_type: String):
    ds = ws.auto_create_dataset(
        workspace,
        pipeline_type,
        __pipeline_group__,
        __pipeline_dtype__,
        fill=__pipeline_fill__,
    )
    ds.set_attr("kind", pipeline_type)

    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(
    workspace: String, full: SmartBoolean = False, filter: SmartBoolean = True
):
    datasets = ws.existing_datasets(workspace, group=__pipeline_group__)

    if full:
        datasets = {
            "{}/{}".format(__pipeline_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}

    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}

    return datasets


@hug.get()
def remove(workspace: String, pipeline_id: String):
    ws.delete_dataset(workspace, pipeline_id, group=__pipeline_group__)


@hug.get()
def rename(workspace: String, pipeline_id: String, new_name: String):
    ws.rename_dataset(workspace, pipeline_id, __pipeline_group__, new_name)


@hug.get()
def group():
    return __pipeline_group__


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[""].items():
        if name[1:] in ["available", "create", "existing", "remove", "rename", "group"]:
            continue
        name = name[1:]
        func = method["GET"][None].interface.spec
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
