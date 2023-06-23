import logging
import ntpath
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from fastapi import APIRouter, Body



pipelines = APIRouter()



@pipelines.get("/watershed", response_model=None)
def watershed(src: str, anno_id: str, dst: str) -> "CLASSICAL":
    """Simple wrapper around skimage watershed algorithm.

    Args:
        src (str): Source Pipeline URI. Feature image to be segmented.
        anno_id (str): Annotation label image to use as seed points.
        dst (str): Destination Pipeline URI.

    Returns:
    """
    from survos2.server.filtering import watershed

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
