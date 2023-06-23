import ntpath
from typing import List
import os
import matplotlib.patheffects as PathEffects
import numpy as np
from loguru import logger
from survos2.api import workspace as ws

from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel

from survos2.api._analyzer.utils import remove_masked_entities
from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()



@analyzer.get("/remove_masked_objects", response_model=None)
def remove_masked_objects(
    src: str,
    dst: str,
    workspace: str,
    feature_id: str,
    object_id: str,
    invert: bool,
) -> "OBJECTS":
    """Remove objects that are masked by a background mask.

    Args:
        src (str): Source image URI.
        dst (str): Destination image URI.
        feature_id (str): Feature image to constrain the point locations to.
        object_id (str): Object URI to use as source points.
        invert (Boolean): Whether to invert the mask.

    Returns:
        list of lists: List of lists of entities.
    """

    DataModel.g.current_workspace = workspace

    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")

    logger.debug(f"Getting objects {src}")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    
    
    objects_fullname = ds_objects.get_metadata("fullname")
    objects_scale = ds_objects.get_metadata("scale")
    objects_offset = ds_objects.get_metadata("offset")
    objects_crop_start = ds_objects.get_metadata("crop_start")
    objects_crop_end = ds_objects.get_metadata("crop_end")

    logger.debug(f"Getting objects from {src} and file {objects_fullname}")
    from survos2.frontend.components.entity import make_entity_df, setup_entity_table
    objects_path = ds_objects._path
    
    
    tabledata, entities_df = setup_entity_table(
        os.path.join(objects_path, objects_fullname),
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
    )

    logger.debug(f"Removing entities using feature as mask: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        mask = DM.sources[0][:]

    if invert:
        mask = 1.0 - mask

    logger.debug(f"Initial number of objects: {len(entities_df)}")
    refined_entity_df = make_entity_df(
        remove_masked_entities((mask == 0) * 1.0, np.array(entities_df))
    )

    logger.debug(f"Removing entities using mask with shape {mask.shape}")
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
    logger.debug(f"Total number of entities after masking {len(refined_entity_df)}")
    print(result_list)

    result_list = np.array(result_list)
    result_list = result_list.tolist()
    
    return result_list

