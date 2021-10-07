from enum import auto
import logging
import ntpath
import os
from typing import List
from dataclasses import dataclass
from datetime import datetime


import dask.array as da
import hug
#from numba.core.types.scalars import Integer
import numpy as np
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
)
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager, dask_relabel_chunks
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import encode_numpy
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.entity.anno.pseudo import make_pseudomasks
from torch.utils.data import DataLoader
from survos2.frontend.components.entity import setup_entity_table
from survos2.api.workspace import add_dataset, auto_create_dataset


__pipeline_group__ = "pipelines"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0


def pass_through(x):
    return x

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
    
    #if int(selected_label) != -1:
    #    anno_base_level = (anno_base_level == int(selected_label)) * 1.0

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
    )

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=True))

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

    @dataclass
    class PatchWorkflow:
        vols: List[np.ndarray]
        locs: np.ndarray
        entities: dict
        bg_mask: np.ndarray
        params: dict
        gold: np.ndarray

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


@hug.get()
@save_metadata
def train_2d_unet(
    src: DataURI,
    dst: DataURI,
    workspace: String,
    anno_id: DataURI,
    feature_id: DataURI,
    unet_train_params: dict
):
    logger.debug(
        f"Train_2d_unet using anno {anno_id} and feature {feature_id}"
    )

    # get anno
    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_level = src_dataset[:] & 15
    anno_labels = np.unique(anno_level)
    logger.debug(f"Obtained annotation level with labels {anno_labels}")
    
    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Adding feature of shape {src_dataset.shape}")
        feature = src_dataset[:]

    logger.info(
        f"Train_2d_unet with feature shape {feature.shape} and anno shape {anno_level.shape}"
    )
    from survos2.server.unet2d.data_utils import TrainingDataSlicer
    from survos2.server.unet2d.unet2d import Unet2dTrainer
    slicer = TrainingDataSlicer(feature, anno_level, clip_data=True)
    ws_object = ws.get(workspace)
    data_out_path = Path(ws_object.path, "unet2d")
    data_slice_path = data_out_path / "data"
    anno_slice_path = data_out_path / "seg"
    logger.info(f"Making output folders for slices in {data_out_path}")
    data_slice_path.mkdir(exist_ok=True, parents=True)
    anno_slice_path.mkdir(exist_ok=True, parents=True)
    slicer.output_data_slices(data_slice_path, "data")
    slicer.output_label_slices(anno_slice_path, "seg")
    trainer = Unet2dTrainer(data_slice_path, anno_slice_path, slicer.codes)
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
    segmentation = trainer.return_fast_prediction_volume(slicer.data_vol)
    # If more than one annotation label is provided add 1 to the segmentation
    if not np.array_equal(np.array([0, 1]), anno_labels):
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
    anno_id: DataURI,
    feature_id: DataURI,
    model_path: str,
    no_of_planes: int
):
    logger.debug(
        f"Predict_2d_unet with feature {feature_id} in {no_of_planes} planes"
    )

    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_level = src_dataset[:] & 15
    logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Adding feature of shape {src_dataset.shape}")
        feature = src_dataset[:]

    logger.info(
        f"Predict_2d_unet with feature shape {feature.shape} using model {model_path}"
    )
    from survos2.server.unet2d.unet2d import Unet2dPredictor
    from survos2.server.unet2d.data_utils import PredictionHDF5DataSlicer
    ws_object = ws.get(workspace)
    root_path = Path(ws_object.path, "unet2d")
    root_path.mkdir(exist_ok=True, parents=True)
    predictor = Unet2dPredictor(root_path)
    predictor.create_model_from_zip(Path(model_path))
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H_%M_%S")
    slicer = PredictionHDF5DataSlicer(predictor, feature, clip_data=True)
    if no_of_planes == 1:
        segmentation = slicer.predict_1_way(root_path, output_prefix=dt_string)
    elif no_of_planes == 3:
        segmentation = slicer.predict_3_ways(root_path, output_prefix=dt_string)
    segmentation += np.ones_like(segmentation)
    def pass_through(x):
        return x

    map_blocks(pass_through, segmentation, out=dst, normalize=False)
    

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
