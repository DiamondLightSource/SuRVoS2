import ntpath
from typing import List
import os
import numpy as np
from loguru import logger

from survos2.api import workspace as ws
from survos2.api.utils import dataset_repr, get_function_api

from survos2.api._analyzer.label_analysis import analyzer as label_analysis
from survos2.api._analyzer.geometry import analyzer as geometry
from survos2.api._analyzer.remove_masked_objects import analyzer as remove_masked_objects
from survos2.api._analyzer.image_stats import analyzer as image_stats
from survos2.api._analyzer.patch_clusterer import analyzer as patch_clusterer
from survos2.api._analyzer.spatial_clustering import analyzer as spatial_clustering
from survos2.api._analyzer.connected_components import analyzer as connected_components

from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()
analyzer.include_router(label_analysis)
analyzer.include_router(geometry)
analyzer.include_router(remove_masked_objects)
analyzer.include_router(image_stats)
analyzer.include_router(patch_clusterer)
analyzer.include_router(spatial_clustering)
analyzer.include_router(connected_components)


__analyzer_fill__ = 0
__analyzer_dtype__ = "uint32"
__analyzer_group__ = "analyzer"
__analyzer_names__ = [
    "find_connected_components",
    "patch_stats",
    "object_detection_stats",
    "segmentation_stats",
    "label_analyzer",
    "label_splitter",
    "binary_image_stats",
    "spatial_clustering",
    "remove_masked_objects",
    "object_analyzer",
    "binary_classifier",
    "point_generator",
]



@analyzer.get("/create")
def create(workspace: str, order: int = 0):
    analyzer_type = __analyzer_names__[order]

    ds = ws.auto_create_dataset(
        workspace,
        analyzer_type,
        __analyzer_group__,
        __analyzer_dtype__,
        fill=__analyzer_fill__,
    )
    ds.set_attr("kind", analyzer_type)
    return dataset_repr(ds)

@analyzer.get("/existing")
def existing(workspace: str, full: bool = False, order: int = 0):
    filter = __analyzer_names__[order]
    datasets = ws.existing_datasets(workspace, group=__analyzer_group__)  # , filter=filter)
    if full:
        return {"{}/{}".format(__analyzer_group__, k): dataset_repr(v) for k, v in datasets.items()}
    return {k: dataset_repr(v) for k, v in datasets.items()}


@analyzer.get("/remove")
def remove(workspace: str, analyzer_id: str):
    ws.delete_dataset(workspace, analyzer_id, group=__analyzer_group__)
    return {"done": "ok"}


@analyzer.get("/rename")
def rename(workspace: str, analyzer_id: str, new_name: str):
    ws.rename_dataset(workspace, analyzer_id, __analyzer_group__, new_name)
    return {"done": "ok"}


@analyzer.get("/available")
def available():
    h = analyzer  # hug.API(__name__)
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
