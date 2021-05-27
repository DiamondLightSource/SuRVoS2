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


from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager, dask_relabel_chunks
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import encode_numpy

from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.entity.anno.pseudo import make_pseudomasks


__pipeline_group__ = "pipelines"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0



@hug.get()
def generate_blobs(
    workspace: String,
    feature_ids: DataURIList,
    object_id: DataURI,
    acwe : SmartBoolean,
    dst: DataURI,
):
    logger.debug(
        f"object detection with workspace {workspace}, with features {feature_ids} and {object_id}"
    )

    from survos2.entity.patches import (
        PatchWorkflow,
        init_entity_workflow,
        organize_entities,
    )

    # get features
    features = []
    for feature_id in feature_ids:
        src = DataModel.g.dataset_uri(feature_id, group="features")
        logger.debug(f"Getting features {src}")

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

    logger.debug(
        f"Getting objects from {src} and file {objects_fullname}"
    )
    from survos2.frontend.components.entity import make_entity_df, setup_entity_table
    tabledata, entities_df = setup_entity_table(
        objects_fullname, scale=objects_scale, offset=objects_offset, crop_start=objects_crop_start, crop_end=objects_crop_end 
    )
    print(entities_df)
    
    entities = np.array(make_entity_df(np.array(entities_df), flipxy=True))

    #default params TODO make generic, allow editing 
    entity_meta = {
        "0": {
            "name": "class1",
            "size": np.array((15, 15, 15)) * objects_scale,
            "core_radius": np.array((7, 7, 7)) * objects_scale,
        },
        "1": {
            "name": "class2",
            "size": np.array((17, 17, 17)) * objects_scale,
            "core_radius": np.array((9, 9, 9)) * objects_scale,
        },
        "2": {
            "name": "class3",
            "size": np.array((22, 22, 22)) * objects_scale,
            "core_radius": np.array((11, 11, 11)) * objects_scale,
        },
        "5": {
            "name": "class4",
            "size": np.array((12, 12, 12)) * objects_scale,
            "core_radius": np.array((5, 5, 5)) * objects_scale,
        },
    }

    combined_clustered_pts, classwise_entities = organize_entities(
        features[0], entities, entity_meta, plot_all=False
    )

    wparams = {}
    wparams["entities_offset"] = (0, 0, 0)

    wf = PatchWorkflow(
        features, combined_clustered_pts, classwise_entities, features[0], wparams,combined_clustered_pts
    )

    gt_proportion = 0.5
    wf_sel = np.random.choice(range(len(wf.locs)), int(gt_proportion * len(wf.locs)))
    gt_entities = wf.locs[wf_sel]
    logger.debug(f"Produced {len(gt_entities)} ground truth entities.")

    combined_clustered_pts, classwise_entities = organize_entities(
        wf.vols[0], gt_entities, entity_meta
    )

    wf.params["entity_meta"] = entity_meta

    anno_masks, anno_all = make_pseudomasks(
        wf,
        classwise_entities,
        acwe=acwe,
        padding=(128, 128, 128),
        core_mask_radius=(8, 8, 8),
    )

    if acwe:
        combined_anno = anno_masks["acwe"]
    else:
        combined_anno = (
            anno_masks["0"]["mask"]
            + anno_masks["1"]["mask"]
            + anno_masks["2"]["mask"]
            + anno_masks["5"]["mask"]
    )

    combined_anno = (combined_anno > 0.1) * 1.0
    # store in dst
    dst = DataModel.g.dataset_uri(dst, group="pipelines")

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = combined_anno


@hug.get()
@save_metadata
def superregion_segment(
    src: DataURI,
    dst: DataURI,
    workspace : String,
    anno_id: DataURI,
    constrain_mask: DataURI,
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
