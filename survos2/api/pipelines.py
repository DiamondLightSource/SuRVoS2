import os
import hug
import logging
import numpy as np
import dask.array as da
from loguru import logger

from survos2.utils import encode_numpy
from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.api.utils import save_metadata, dataset_repr
from survos2.api.types import (
    DataURI,
    Float,
    SmartBoolean,
    FloatOrVector,
    IntList,
    String,
    Int,
    FloatList,
    DataURIList,
)
from survos2.api import workspace as ws
from survos2.server.features import features_factory

from survos2.improc.utils import DatasetManager, dask_relabel_chunks
from survos2.model import DataModel
from survos2.api.utils import get_function_api, save_metadata, dataset_repr
from typing import List

from survos2.server.config import cfg


__pipeline_group__ = "pipeline"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0


# @hug.get()
# def run_pipeline():
#    pipeline = Pipeline(cData.cfg.pipeline_ops)
#    pipeline.init_payload(p)
#    process_pipeline(pipeline)
#    result = pipeline.get_result()


@hug.get()
def superregion_segment(
    workspace: String,
    anno_id: String,
    region_id: String,
    feature_ids: DataURIList,
    lam: float,
    dst: DataURI,
):
    logger.debug(
        f"superregion_segment with workspace {workspace}, anno {anno_id} and superregions {region_id} and features {feature_ids}"
    )
    # get anno
    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_image = src_dataset[:]

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
        f"sr_predict with {len(features)} features and anno of shape {anno_image.shape} and sr of shape {supervoxel_image.shape}"
    )

    # run predictions
    from survos2.server.superseg import sr_predict

    segmentation = sr_predict(supervoxel_image, anno_image, features, lam)

    # store in dst
    dst = DataModel.g.dataset_uri(dst, group="pipeline")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = segmentation


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
