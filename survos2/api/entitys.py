import hug
import logging
import os.path as op
import numpy as np
import dask.array as da
from loguru import logger

from survos2.utils import encode_numpy
from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.api.utils import save_metadata, dataset_repr
from survos2.api.types import (
    DataURI,
    Float,
    SmartBoolean,
    FloatOrVector,
    IntList,
    String,
    Int,
    FloatList,
    DataURIList,
)
from survos2.api import workspace as ws
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager


__entitys_fill__ = 0
__entitys_dtype__ = "uint32"
__entitys_group__ = "entitys"
__entitys_names__ = ["points", "boxes"]


@hug.get()
def set_csv(dst: DataURI, fullname: String) -> "Points":

    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
        logger.info(f"Got __data__ volume of size {img_volume.shape}")

    # store in dst
    logger.info(f"Opening dataset {dst}")
    with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
        DM.out[:] = img_volume
        dst_dataset = DM.sources[0]
        dst_dataset.set_attr("fullname", fullname)


@hug.get()
def create(workspace: String, fullname: String, order: Int = 0):
    entitys_type = __entitys_names__[order]

    ds = ws.auto_create_dataset(
        workspace,
        entitys_type,
        __entitys_group__,
        __entitys_dtype__,
        fill=__entitys_fill__,
    )

    ds.set_attr("kind", entitys_type)
    ds.set_attr("fullname", fullname)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace: String, full: SmartBoolean = False, order: Int = 0):
    filter = __entitys_names__[order]
    datasets = ws.existing_datasets(workspace, group=__entitys_group__, filter=filter)
    if full:
        return {
            "{}/{}".format(__entitys_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    return {k: dataset_repr(v) for k, v in datasets.items()}


@hug.get()
def remove(workspace: String, entitys_id: String):
    ws.delete_dataset(workspace, entitys_id, group=__entitys_group__)


@hug.get()
def rename(workspace: String, entitys_id: String, new_name: String):
    ws.rename_dataset(workspace, entitys_id, __entitys_group__, new_name)
