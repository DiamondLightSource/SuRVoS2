
import ntpath
from typing import List
import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.api.objects import get_entities
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.api.workspace import auto_create_dataset

from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from fastapi import APIRouter, Body

pipelines = APIRouter()


@pipelines.get("/superregion_segment", response_model=None)
@save_metadata
def superregion_segment(
    src: str = Body(),
    dst: str = Body(),
    workspace: str = Body(),
    anno_id: str = Body(),
    constrain_mask: str = Body(),
    region_id: str = Body(),
    lam: float = Body(),
    refine: bool = Body(),
    classifier_type: str = Body(),
    projection_type: str = Body(),
    confidence: bool = Body(),
    classifier_params: dict = Body(),
    feature_ids: List[str] = Body(),
) -> "CLASSICAL":
    """Classical ML for superregion segmentation. Multiple feature images can be used to calculate per-region features, which
    are then used to train a model that can classify superregions. Usually a subset of the superregions is used, corresponding to
    those superregions for which scribbles have been painted. The entire volume of superregions is then predicted using this model.

    Args:
        src (str): Source pipeline URI.
        dst (str): Destination pipeline URI.
        workspace (str): Workspace to use.
        anno_id (str): Annotation URI to use as the label image.
        constrain_mask (str): Mask to constrain the prediction to.
        region_id (str): Region URI to use as the super-regions.
        feature_ids (DataURIList): Feature URIs to use as features.
        lam (float): Lambda for the MRF smoothing model.
        refine (bool): Boolean to use the refinement step.
        classifier_type (str): One of Ensemble or SVM.
        projection_type (str): Which type of projection to use to compute features.
        classifier_params (dict): Classifier-specific parameters
        confidence (bool): Whether to generate a confidence map as a feature image.

    """
    logger.debug(
        f"superregion_segment using anno {anno_id} and superregions {region_id} and features {feature_ids}"
    )

    # get anno
    src = DataModel.g.dataset_uri(anno_id, group="annotations")

    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]

        anno_level = src_dataset[:] & 15

        logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

    # get superregions
    src = DataModel.g.dataset_uri(region_id, group="superregions")
    with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        supervoxel_image = src_dataset[:]

    # get features
    features = []

    for feature_id in feature_ids:
        src = DataModel.g.dataset_uri(feature_id, group="features")
        logger.debug(f"Getting features {src}")

        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            logger.debug(f"Adding feature of shape {src_dataset.shape}")
            features.append(src_dataset[:])

    logger.debug(
        f"sr_predict with {len(features)} features and anno of shape {anno_level.shape} and sr of shape {supervoxel_image.shape}"
    )

    # run predictions
    from survos2.server.superseg import sr_predict

    superseg_cfg = cfg.pipeline
    superseg_cfg["predict_params"] = classifier_params
    superseg_cfg["predict_params"]["clf"] = classifier_type
    superseg_cfg["predict_params"]["type"] = classifier_params["type"]
    superseg_cfg["predict_params"]["proj"] = projection_type
    logger.debug(f"Using superseg_cfg {superseg_cfg}")

    if constrain_mask != "None":
        import ast

        constrain_mask = ast.literal_eval(constrain_mask)
        constrain_mask_id = ntpath.basename(constrain_mask["level"])
        label_idx = constrain_mask["idx"]
        src = DataModel.g.dataset_uri(constrain_mask_id, group="annotations")
        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            constrain_mask_level = src_dataset[:] & 15
        logger.debug(
            f"Constrain mask {constrain_mask_id}, label {label_idx} level shape {constrain_mask_level.shape} with unique labels {np.unique(constrain_mask_level)}"
        )
        mask = constrain_mask_level == label_idx - 1
    else:
        mask = None

    segmentation, conf_map = sr_predict(
        supervoxel_image,
        anno_level,
        features,
        mask,
        superseg_cfg,
        refine,
        lam,
    )
    conf_map = conf_map[:, :, :, 1]
    logger.info(f"Obtained conf map of shape {conf_map.shape}")

    def pass_through(x):
        return x

    map_blocks(pass_through, segmentation, out=dst, normalize=False)

    if confidence:
        dst = auto_create_dataset(
            DataModel.g.current_workspace,
            name="confidence_map",
            group="features",
            dtype="float32",
        )
        dst.set_attr("kind", "raw")
        with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
            DM.out[:] = conf_map




