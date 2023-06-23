import ntpath
from typing import List
import os
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from loguru import logger
from matplotlib import offsetbox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import ndimage
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.api.utils import pass_through
from survos2.api._analyzer.label_analysis import analyzer as label_analyzer

from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()




@analyzer.get("/segmentation_stats", response_model=None)
def segmentation_stats(
    src: str,
    dst: str,
    modeA: str,
    modeB: str,
    workspace: str,
    pipelines_id_A: str,
    analyzers_id_A: str,
    annotations_id_A: str,
    pipelines_id_B: str,
    analyzers_id_B: str,
    annotations_id_B: str,
    label_index_A: int,
    label_index_B: int,
) -> "IMAGE":
    """Calculate segmentation statistics to compare two label images.
    Dice and IOU are returned.

    Args:
        src (str): Source image URI.
        dst (str): Destination image URI.
        modeA (str): Pipeline, Analyzer or Annotation.
        modeB (str): Pipeline, Analyzer or Annotation.
        workspace (str): Workspace to get images from.
        pipelines_id_A (str): Pipelines URI if mode A is 'pipelines'.
        analyzers_id_A (str): Analyzers URI if mode A is 'analyzers'.
        annotations_id_A (str): Annotations URI if mode A is 'annotations'.
        pipelines_id_B (str): Pipelines URI if mode B is 'pipelines'.
        analyzers_id_B (str): Analyzers URI if mode B is 'analyzers'.
        annotations_id_B (str): Annotations URI if mode B is 'annotations'.
        label_index_A (int): Label value to use as foreground.
        label_index_B (int): Label value to use as foreground.

    Returns:
        list of float: Dice and IOU.
    """
    DataModel.g.current_workspace = workspace

    if modeA == "1":
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id_A), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif modeA == "2":
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id_A), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif modeA == "3":
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id_A), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")

    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        print(src)
        print(DM.sources)
        print(DM.sources[0][:])
        seg = DM.sources[0][:]
        logger.debug(f"src_dataset shape {seg[:].shape}")

    src_dataset_arr_A = seg.astype(np.uint32) & 15

    if modeB == "1":
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id_B), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif modeB == "2":
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id_B), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif modeB == "3":
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id_B), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")

    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        segB = DM.sources[0][:]
        logger.debug(f"src_dataset shape {segB[:].shape}")

    src_dataset_arr_B = segB.astype(np.uint32) & 15

    single_label_level_A = (src_dataset_arr_A == label_index_A) * 1.0
    single_label_level_B = (src_dataset_arr_B == label_index_B) * 1.0

    logger.debug(f"Count: {np.sum(single_label_level_A * single_label_level_B)}")

    # from survos2.entity.trainer import score_dice
    # logger.debug(f"Dice loss {score_dice(single_label_level_A, single_label_level_B)}")

    from torchmetrics import Dice, JaccardIndex

    dice = Dice(average="micro")
    jaccard = JaccardIndex(task="binary", num_classes=2)

    A_t = torch.IntTensor(single_label_level_A)
    B_t = torch.IntTensor(single_label_level_B)

    dice_score = dice(A_t, B_t)
    iou_score = jaccard(A_t, B_t)

    result_list = [float(dice_score.numpy()), float(iou_score.numpy())]

    return result_list