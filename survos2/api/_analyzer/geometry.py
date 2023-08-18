import ntpath
import os

import numpy as np
import torch
from loguru import logger
from survos2.entity.entities import make_entity_bvol, make_entity_df
from survos2.entity.sampler import (
    generate_random_points_in_volume,

)
from survos2.entity.utils import get_surface
from survos2.frontend.components.entity import setup_entity_table
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel


from fastapi import APIRouter

from survos2.api._analyzer.utils import remove_masked_entities

analyzer = APIRouter()

@analyzer.get("/point_generator", response_model=None)
def point_generator(
    bg_mask_id: str,
    num_before_masking: int,
):
    """Generate a set of random points and then mask out regions based on the background mask.

    Args:
        bg_mask_id (str): Background mask to use
        num_before_masking (int): Number of points to start with, before masking.

    Returns:
        (list of entities): Resultant set of masked points
    """
    mask_name = ntpath.basename(bg_mask_id)

    if mask_name == "None":
        src = DataModel.g.dataset_uri("001_raw", group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            bg_mask = np.zeros_like(DM.sources[0][:])
            logger.debug(f"Feature shape {bg_mask.shape}")

    else:
        # get feature for background mask
        src = DataModel.g.dataset_uri(ntpath.basename(bg_mask_id), group="features")

        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            bg_mask = DM.sources[0][:]
            logger.debug(f"Feature shape {bg_mask.shape}")

    random_entities = generate_random_points_in_volume(
        bg_mask, num_before_masking, border=(0, 0, 0)
    ).astype(np.uint32)

    if mask_name != "None":
        from survos2.entity.utils import remove_masked_entities

        logger.debug(f"Before masking random entities generated of shape {random_entities.shape}")
        result_entities = remove_masked_entities(bg_mask, random_entities)
        logger.debug(f"After masking: {random_entities.shape}")
    else:
        result_entities = random_entities

    result_entities[:, 3] = np.array([6] * len(result_entities))
    result_entities = np.array(make_entity_df(result_entities, flipxy=True))
    result_entities = result_entities.tolist()

    return result_entities