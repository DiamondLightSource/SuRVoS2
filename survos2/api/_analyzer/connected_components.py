import ntpath
from typing import List
import os
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
from loguru import logger

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from scipy.ndimage import measurements as measure
from skimage.morphology import ball, octahedron
from sklearn.decomposition import PCA

from survos2.api import workspace as ws
from survos2.entity.components import measure_components
from survos2.entity.entities import make_entity_bvol, make_entity_df

from survos2.entity.utils import get_surface
from survos2.frontend.components.entity import setup_entity_table
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.api.utils import pass_through

from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()



def component_bounding_boxes(images):
    bbs_tables = []
    bbs_arrs = []

    for image in images:
        bbs_arr = measure_components(image)
        bbs_arrs.append(bbs_arr)
        bbs_table = make_entity_bvol(bbs_arr)
        bbs_tables.append(bbs_table)

    return bbs_tables, bbs_arrs


def detect_blobs(
    padded_proposal,
    area_min=0,
    area_max=1e12,
    plot_all=False,
):
    images = [padded_proposal]
    bbs_tables, bbs_arrs = component_bounding_boxes(images)
    logger.debug(f"Detecting blobs on image of shape {padded_proposal.shape}")
    zidx = padded_proposal.shape[0] // 2

    from survos2.frontend.nb_utils import summary_stats

    logger.debug("Component stats: ")
    logger.debug(f"{summary_stats(bbs_tables[0]['area'])}")

    if plot_all:
        for idx in range(len(bbs_tables)):
            logger.debug(idx)
            plt.figure(figsize=(5, 5))
            plt.imshow(images[idx][zidx, :], cmap="gray")
            plt.scatter(bbs_arrs[idx][:, 4], bbs_arrs[idx][:, 3])

    selected_entities = bbs_tables[0][
        (bbs_tables[0]["area"] > area_min) & (bbs_tables[0]["area"] < area_max)
    ]
    logger.debug(f"Number of selected entities {len(selected_entities)}")

    return bbs_tables, selected_entities


@analyzer.get("/find_connected_components", response_model=None)
def find_connected_components(
    src: str,
    dst: str,
    workspace: str,
    label_index: int,
    area_min: int,
    area_max: int,
    mode: str,
    pipelines_id: str,
    analyzers_id: str,
    annotations_id: str,
) -> "SEGMENTATION":
    """Find connected components and calculate a table of stats for each component. Filter the components by the
    area min and max. The mode can be 'largest' or 'smallest'.

    Args:
        src (str): Source image URI.
        dst (str): Destination image URI.
        workspace (str): Workspace to get images from.
        label_index (int): Index of the label to find connected components for.
        area_min (int): Minimumm component area.
        area_max (int): Maximum component area.
        mode (str): Select either Pipelines, Annotations or Analyzers.
        pipelines_id (str): Pipelines URI if mode is 'pipelines'.
        analyzers_id (str): Analyzers URI if mode is 'analyzers'.
        annotations_id (str): Annotations URI if mode is 'annotations'.

    Returns:
        list of dict: List of dictionaries with the stats for each component.
    """

    DataModel.g.current_workspace = workspace

    logger.debug(f"Finding connected components on segmentation: {pipelines_id}")

    if mode == "1":
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif mode == "2":
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif mode == "3":
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")

    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        seg = DM.sources[0][:]
        logger.debug(f"src_dataset shape {seg[:].shape}")

    src_dataset_arr = seg.astype(np.uint32) & 15

    single_label_level = (src_dataset_arr == label_index) * 1.0

    bbs_tables, selected_entities = detect_blobs(single_label_level)

    # iterate through component and create result list
    result_list = []
    for i in range(len(bbs_tables[0])):
        if (bbs_tables[0].iloc[i]["area"] > area_min) & (bbs_tables[0].iloc[i]["area"] < area_max):
            result_list.append(
                [
                    bbs_tables[0].iloc[i]["z"],
                    bbs_tables[0].iloc[i]["x"],
                    bbs_tables[0].iloc[i]["y"],
                    bbs_tables[0].iloc[i]["area"],
                ]
            )

    result_list = np.array(result_list)
    result_list = result_list.tolist()
    return result_list



@analyzer.get("/binary_image_stats", response_model=None)
def binary_image_stats(
    src: str,
    dst: str,
    workspace: str,
    feature_id: str,
    threshold: float = 0.5,
    area_min: int = 0,
    area_max: int = 1e12,
) -> "IMAGE":
    DataModel.g.current_workspace = workspace

    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    src_thresh = (src_dataset_arr > threshold) * 1.0
    bbs_tables, selected_entities = detect_blobs(src_thresh, area_min=area_min, area_max=area_max)

    result_list = []

    for i in range(len(bbs_tables[0])):
        if (bbs_tables[0].iloc[i]["area"] < area_max) & (bbs_tables[0].iloc[i]["area"] > area_min):
            result_list.append(
                [
                    bbs_tables[0].iloc[i]["z"],
                    bbs_tables[0].iloc[i]["x"],
                    bbs_tables[0].iloc[i]["y"],
                    bbs_tables[0].iloc[i]["area"],
                ]
            )

    return result_list
