"""
Pipelines

Pipelines are for segmentation, postprocessing and synthesis tasks. Pipelines output an integer image.  

Generally a pipeline function has no return value as it follows the convention that the 
destination pipeline is used as output.

"""
import logging
import os
import sys

import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.api.utils import dataset_repr, get_function_api, save_metadata, pass_through, _unpack_lists


from survos2.api._pipelines.rasterize_points import pipelines as rasterize_points
from survos2.api._pipelines.superregion_segment import pipelines as superregion_segment
from survos2.api._pipelines.multiaxis_cnn import pipelines as multiaxis_cnn
from survos2.api._pipelines.cnn3d import pipelines as cnn3d
from survos2.api._pipelines.cleaning import pipelines as cleaning
from survos2.api._pipelines.watershed import pipelines as watershed
from survos2.api._pipelines.postprocess import pipelines as postprocess

__pipeline_group__ = "pipelines"
__pipeline_dtype__ = "float32"
__pipeline_fill__ = 0

from fastapi import APIRouter, Body, Query

pipelines = APIRouter()
pipelines.include_router(rasterize_points)
pipelines.include_router(superregion_segment)
pipelines.include_router(multiaxis_cnn)
pipelines.include_router(cnn3d)
pipelines.include_router(postprocess)
pipelines.include_router(watershed)
pipelines.include_router(cleaning)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)





@pipelines.get("/create")
def create(workspace: str, pipeline_type: str):
    ds = ws.auto_create_dataset(
        workspace,
        pipeline_type,
        __pipeline_group__,
        __pipeline_dtype__,
        fill=__pipeline_fill__,
    )
    ds.set_attr("kind", pipeline_type)

    return dataset_repr(ds)


# @hug.local()
@pipelines.get("/existing")
def existing(workspace: str, full: bool = False, filter: bool = True):
    datasets = ws.existing_datasets(workspace, group=__pipeline_group__)

    if full:
        datasets = {
            "{}/{}".format(__pipeline_group__, k): dataset_repr(v) for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}

    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}

    return datasets


@pipelines.get("/remove")
def remove(workspace: str, pipeline_id: str):
    ws.delete_dataset(workspace, pipeline_id, group=__pipeline_group__)
    return {"done": "ok"}


@pipelines.get("/rename")
def rename(workspace: str, pipeline_id: str, new_name: str):
    ws.rename_dataset(workspace, pipeline_id, __pipeline_group__, new_name)
    return {"done": "ok"}


@pipelines.get("/group")
def group():
    return __pipeline_group__


@pipelines.get("/available")
def available():
    h = pipelines  # hug.API(__name__)
    all_features = []
    for r in h.routes:
        name = r.name
        if name in ["available", "create", "existing", "remove", "rename", "group"]:
            continue
        func = r.endpoint
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
