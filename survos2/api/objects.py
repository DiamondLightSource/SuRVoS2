import logging
import os.path as op

import dask.array as da
import hug
import numpy as np
from loguru import logger

from survos2.api import workspace as ws
from survos2.api.types import (
    DataURI,
    DataURIList,
    Float,
    FloatList,
    FloatOrVector,
    Int,
    IntList,
    SmartBoolean,
    String,
)
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy

__objects_fill__ = 0
__objects_dtype__ = "uint32"
__objects_group__ = "objects"
__objects_names__ = ["points", "boxes"]


@hug.get()
def points(
    dst: DataURI,
    fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector,
) -> "GEOMETRY":
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")
    # store in dst
    logger.info(f"Storing in dataset {dst}")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)


@hug.get()
def boxes(
    dst: DataURI,
    fullname: String,
    scale: float,
    offset: FloatOrVector,
    crop_start: FloatOrVector,
    crop_end: FloatOrVector,
) -> "GEOMETRY":
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")

    # store in dst
    logger.info(f"Storing in dataset {dst}")

    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = np.zeros_like(img_volume)
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("scale", scale)
        dst_dataset.set_attr("offset", offset)
        dst_dataset.set_attr("crop_start", crop_start)
        dst_dataset.set_attr("crop_end", crop_end)

        csv_saved_fullname = dst_dataset.save_file(fullname)
        logger.info(f"Saving {fullname} to {csv_saved_fullname}")
        dst_dataset.set_attr("fullname", csv_saved_fullname)



@hug.get()
def create(workspace: String, fullname: String, order: Int = 0):
    objects_type = __objects_names__[order]

    ds = ws.auto_create_dataset(
        workspace,
        objects_type,
        __objects_group__,
        __objects_dtype__,
        fill=__objects_fill__,
    )

    ds.set_attr("kind", objects_type)
    ds.set_attr("fullname", fullname)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(
    workspace: String,
    full: SmartBoolean = False,
    filter: SmartBoolean = True,
    order: Int = 0,
):
    datasets = ws.existing_datasets(workspace, group=__objects_group__)

    if full:
        datasets = {
            "{}/{}".format(__objects_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    else:
        datasets = {k: dataset_repr(v) for k, v in datasets.items()}

    if filter:
        datasets = {k: v for k, v in datasets.items() if v["kind"] != "unknown"}

    return datasets


@hug.get()
def remove(workspace: String, objects_id: String):
    ws.delete_dataset(workspace, objects_id, group=__objects_group__)


@hug.get()
def rename(workspace: String, objects_id: String, new_name: String):
    ws.rename_dataset(workspace, objects_id, __objects_group__, new_name)


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[""].items():

        if name[1:] in ["available", "create", "existing", "remove", "rename", "group"]:
            continue
        logger.debug(f"Object types available {name}")
        name = name[1:]
        func = method["GET"][None].interface.spec
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
