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
from survos2.api.objects import get_entities
from survos2.api.utils import dataset_repr, get_function_api, save_metadata, pass_through, _unpack_lists
from survos2.api.workspace import auto_create_dataset
from survos2.entity.patches import (
    PatchWorkflow,
    make_patches,
    organize_entities,
    sample_images_and_labels,
    augment_and_save_dataset,
)
from survos2.entity.pipeline_ops import make_proposal
from survos2.entity.sampler import generate_random_points_in_volume
from survos2.entity.train import train_oneclass_detseg
from survos2.entity.utils import pad_vol
from survos2.entity.entities import make_entity_df
from survos2.frontend.components.entity import setup_entity_table
from survos2.frontend.nb_utils import slice_plot
from scipy.ndimage.morphology import binary_erosion
from survos2.entity.utils import pad_vol, get_largest_cc
from survos2.entity.entities import offset_points
from survos2.entity.patches import BoundingVolumeDataset
from survos2.entity.sampler import centroid_to_bvol
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.api.utils import pass_through, _unpack_lists
from fastapi import APIRouter, Body, Query

pipelines = APIRouter()



@pipelines.get("/label_postprocess", response_model=None)
@save_metadata
def label_postprocess(
    src: str,
    dst: str,
    workspace: str,
    level_over: str,
    level_base: str,
    selected_label_for_over: int,
    offset: int,
    base_offset: int,
) -> "POSTPROCESSING":
    """Takes two label (integer) image and performs an operation to combine them.
    Args:
        src (str): stub
        dst (str): Destination pipeline to save into
        workspace (str): workspace id
        level_over (str): Image B
        level_base (str): Image A
        selected_label (int):
        offset (int): Integer value to offset the label
    """
    if level_over != "None":
        src1 = DataModel.g.dataset_uri(level_over, group="annotations")
        with DatasetManager(src1, out=None, dtype="uint16", fillvalue=0) as DM:
            src1_dataset = DM.sources[0]
            anno_over_level = src1_dataset[:] & 15

    src_base = DataModel.g.dataset_uri(level_base, group="annotations")
    with DatasetManager(src_base, out=None, dtype="uint16", fillvalue=0) as DM:
        src_base_dataset = DM.sources[0]
        anno_base_level = src_base_dataset[:] & 15

    result = anno_base_level + base_offset

    if level_over != "None":
        # zero out everything but the selected level in the over image
        anno_over_level[anno_over_level != selected_label_for_over] = 0
        anno_over_level[anno_over_level == selected_label_for_over] = (
            selected_label_for_over + offset
        )
        # mask out those voxels that are in the over image, in the base image
        result = result * (1.0 - (anno_over_level > 0) * 1.0)
        result += anno_over_level

    map_blocks(pass_through, result, out=dst, normalize=False)

