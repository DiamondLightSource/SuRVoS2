import ntpath
import os
import numpy as np
from loguru import logger

from survos2.api import workspace as ws

from survos2.frontend.components.entity import setup_entity_table
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel


from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()


@analyzer.get("/spatial_clustering", response_model=None)
def spatial_clustering(
    src: str = Body(),
    feature_id: str = Body(),
    object_id: str = Body(),
    workspace: str = Body(),
    params: dict = Body(),
) -> "OBJECTS":
    """Cluster points using HDBSCAN or DBSCAN.

    Args:
        src (str): Source Pipelines URI (the current image.)
        feature_id (str): Feature URI to constrain the point locations to.
        object_id (str): Object URI as source points.
        workspace (str): Workspace to use.
        params (dict): Clustering parameters.

    Returns:
        list of lists: List of lists of entities.
    """
    DataModel.g.current_workspace = workspace

    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    scale = ds_objects.get_metadata("scale")

    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname)

    logger.debug(f"Spatial clustering using feature as reference image: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    from survos2.entity.anno.crowd import aggregate

    refined_entity_df = aggregate(entities_df, src_dataset_arr.shape, params=params)

    result_list = []
    for i in range(len(refined_entity_df)):
        result_list.append(
            [
                refined_entity_df.iloc[i]["class_code"],
                refined_entity_df.iloc[i]["z"],
                refined_entity_df.iloc[i]["y"],
                refined_entity_df.iloc[i]["x"],
            ]
        )

    return result_list
