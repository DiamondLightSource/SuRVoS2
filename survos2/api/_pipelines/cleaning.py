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



@pipelines.get("/per_object_cleaning", response_model=None)
@save_metadata
def per_object_cleaning(
    src: str,
    dst: str,
    feature_id: str,
    object_id: str,
    patch_size: List[int] = Query(),  # 64
) -> "POSTPROCESSING":
    # get image feature
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature = DM.sources[0][:]
        logger.debug(f"Feature shape {feature.shape}")

    # get object entities
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]

    objects_name = ds_objects.get_metadata("fullname")
    fname = os.path.basename(objects_name)
    objects_dataset_fullpath = ds_objects._path
    entities_fullname = os.path.join(objects_dataset_fullpath, fname)

    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=False)
    entities_arr = np.array(entities_df)
    entities_arr[:, 3] = np.array([[1] * len(entities_arr)])
    entities = np.array(make_entity_df(entities_arr, flipxy=False))

    target = _per_object_cleaning(entities, feature, display_plots=False, bvol_dim=patch_size)
    # dst = DataModel.g.dataset_uri(dst, group="pipelines")
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = target


def _per_object_cleaning(
    entities,
    seg,
    bvol_dim=(32, 32, 32),
    offset=(0, 0, 0),
    flipxy=True,
    display_plots=False,
):
    patch_size = tuple(1 * np.array(bvol_dim))
    seg = pad_vol(seg, np.array(patch_size))
    target = np.zeros_like(seg)

    entities = np.array(make_entity_df(np.array(entities), flipxy=flipxy))
    entities = offset_points(entities, offset)
    entities = offset_points(entities, -np.array(bvol_dim))

    if display_plots:
        slice_plot(seg, np.array(make_entity_df(entities, flipxy=flipxy)), seg, (60, 300, 300))
    bvol = centroid_to_bvol(np.array(entities), bvol_dim=bvol_dim)
    bvol_seg = BoundingVolumeDataset(seg, bvol, labels=[1] * len(bvol), patch_size=patch_size)
    c = bvol_dim[0], bvol_dim[1], bvol_dim[2]

    for i, p in enumerate(bvol_seg):
        seg, _ = bvol_seg[i]
        cleaned_patch = (get_largest_cc(seg) > 0) * 1.0
        # try:
        #     res = get_surface((cleaned_patch > 0) * 1.0, plot3d=False)
        mask = (binary_erosion((cleaned_patch), iterations=1) > 0) * 1.0

        if display_plots:
            plt.figure()
            plt.imshow(seg[c[0], :])
            plt.figure()
            plt.imshow(mask[c[0], :])

        z_st, y_st, x_st, z_end, y_end, x_end = bvol_seg.bvols[i]
        target[
            z_st:z_end,
            x_st:x_end,
            y_st:y_end,
        ] = (
            (
                mask
                + target[
                    z_st:z_end,
                    x_st:x_end,
                    y_st:y_end,
                ]
            )
            > 0
        ) * 1

    target = target[
        patch_size[0] : target.shape[0] - patch_size[0],
        patch_size[1] : target.shape[1] - patch_size[1],
        patch_size[2] : target.shape[2] - patch_size[2],
    ]

    return target




@pipelines.get("/cleaning", response_model=None)
def cleaning(
    # object_id : str,
    feature_id: str,
    dst: str,
    min_component_size: int = 100,
) -> "POSTPROCESSING":
    """Clean components smaller than a given size from a binary feature volume.

    Args:
        feature_id (str): Feature URI.
        dst (str): Destination Pipeline URI.
        min_component_size (int, optional): Minimum component size. Defaults to 100.
    """
    from survos2.entity.components import (
        filter_small_components,
    )

    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature = DM.sources[0][:]

    from skimage.morphology import remove_small_objects
    from skimage.measure import label

    feature = feature > 0

    seg_cleaned = remove_small_objects(feature, min_component_size)

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = seg_cleaned  # (seg_cleaned > 0) * 1.0
