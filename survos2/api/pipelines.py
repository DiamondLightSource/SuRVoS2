import logging
import ntpath
import os
from typing import List

import dask.array as da
import hug
import numpy as np
from loguru import logger

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
from survos2.api.annotations import get_label_parent

from survos2.frontend.components.entity import make_entity_df, setup_entity_table
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager, dask_relabel_chunks
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.server.features import features_factory
from survos2.server.state import cfg
from survos2.utils import encode_numpy


__pipeline_group__ = "pipelines"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0



@hug.get()
@save_metadata
def superregion_segment(
    src: DataURI,
    dst: DataURI,
    workspace : String,
    anno_id: DataURI,
    constrain_mask_id: DataURI,
    region_id: DataURI,
    feature_ids: DataURIList,
    lam: float,
    refine: SmartBoolean,
    classifier_type: String,
    projection_type: String,
    classifier_params : dict,
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
    src = DataModel.g.dataset_uri(region_id, group="regions")
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
        f"sr_predict with {len(features)} features and anno of shape {anno_level.shape} and sr of shape {supervoxel_image.shape}  using classifier type {classifier_type} and params {classifier_params}, projection type {projection_type} "
    )

    # run predictions
    from survos2.server.superseg import sr_predict

    superseg_cfg = cfg.pipeline
    superseg_cfg["type"] = classifier_type
    superseg_cfg["predict_params"]["proj"] = projection_type
    superseg_cfg["classifier_params"] = classifier_params

    
    if constrain_mask_id != "None":
        label_idx = 2
        parent_level, parent_label_idx = get_label_parent(workspace, anno_id, label_idx)
        logger.debug(f"Parent for {anno_id} is {parent_level} with labels {parent_label_idx}")
        src = DataModel.g.dataset_uri(constrain_mask_id, group="annotations")
        
        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            constrain_mask_level = src_dataset[:] & 15

        logger.debug(
            f"Constrain mask level shape {constrain_mask_level.shape} with labels {np.unique(constrain_mask_level)}"
        )
        
        logger.debug(f"Masking with parent label {parent_label_idx}")
        mask = constrain_mask_level == parent_label_idx
    
    else:
        mask = None

    segmentation = sr_predict(
        supervoxel_image,
        anno_level,
        features,
        mask,
        superseg_cfg,
        refine,
        lam,
    )

    # store in dst
    logger.debug(f"Storing segmentation in dst {dst}")
    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = segmentation.astype(np.uint32)


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
