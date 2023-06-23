
import ntpath
import os
from typing import List
import numpy as np
from loguru import logger
from survos2.api import workspace as ws
from survos2.api.utils import save_metadata
from survos2.entity.patches import (
    PatchWorkflow,
)
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from fastapi import APIRouter, Query

pipelines = APIRouter()


@pipelines.get("/rasterize_points", response_model=None)
@save_metadata
def rasterize_points(
    src: str,
    dst: str,
    workspace: str,
    feature_id: str,
    object_id: str,
    acwe: bool,
    balloon: float,
    threshold: float,
    iterations: int,
    smoothing: int,
    selected_class: int,
    size: List[float] = Query(),
) -> "SYNTHESIS":
    from survos2.entity.anno.pseudo import organize_entities

    src = DataModel.g.dataset_uri(feature_id, group="features")
    logger.debug(f"Getting features {src}")

    features = []
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"Adding feature of shape {src_dataset.shape}")
        features.append(src_dataset[:])

    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]

    objects_fullname = ntpath.basename(ds_objects.get_metadata("fullname"))
    objects_scale = ds_objects.get_metadata("scale")
    objects_offset = ds_objects.get_metadata("offset")
    objects_crop_start = ds_objects.get_metadata("crop_start")
    objects_crop_end = ds_objects.get_metadata("crop_end")

    logger.debug(
        f"Getting objects from {src} and file {objects_fullname} with scale {objects_scale}"
    )
    from survos2.frontend.components.entity import make_entity_df, setup_entity_table

    tabledata, entities_df = setup_entity_table(
        os.path.join(ds_objects._path, objects_fullname),
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
        flipxy=False,
    )

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=False))
    entities = entities[entities[:, 3] == selected_class]
    entities[:, 3] = np.array([0] * len(entities))

    # default params TODO make generic, allow editing
    entity_meta = {
        "0": {
            "name": "class0",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
        "1": {
            "name": "class1",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
        "2": {
            "name": "class2",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
        "3": {
            "name": "class3",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
        "4": {
            "name": "class4",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
        "5": {
            "name": "class5",
            "size": np.array(size),
            "core_radius": np.array((7, 7, 7)),
        },
    }

    combined_clustered_pts, classwise_entities = organize_entities(
        features[0], entities, entity_meta, plot_all=False
    )

    wparams = {}
    wparams["entities_offset"] = (0, 0, 0)

    wf = PatchWorkflow(
        features,
        combined_clustered_pts,
        classwise_entities,
        features[0],
        wparams,
        combined_clustered_pts,
    )

    combined_clustered_pts, classwise_entities = organize_entities(
        wf.vols[0], wf.locs, entity_meta, plot_all=False
    )

    wf.params["entity_meta"] = entity_meta
    from survos2.entity.anno.pseudo import make_pseudomasks

    anno_masks, anno_acwe = make_pseudomasks(
        wf,
        classwise_entities,
        acwe=acwe,
        padding=(128, 128, 128),
        core_mask_radius=size,
        balloon=balloon,
        threshold=threshold,
        iterations=iterations,
        smoothing=smoothing,
    )

    if acwe:
        combined_anno = anno_acwe["0"]
    else:
        combined_anno = anno_masks["0"]["mask"]

    combined_anno = (combined_anno > 0.1) * 1.0

    with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
        DM.out[:] = combined_anno

