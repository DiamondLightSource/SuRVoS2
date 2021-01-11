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


__analyzer_fill__ = 0
__analyzer_dtype__ = "uint32"
__analyzer_group__ = "analyzer"
__analyzer_names__ = ["histogram", "stats"]


@hug.get()
@save_metadata
def simple_stats(
    workspace: String,
    feature_ids: DataURIList,
    dst: DataURI,
):
    print(f"Calculating stats on {feature_id}")


@hug.get()
def create(workspace: String, order: Int = 0):
    analyzer_type = __analyzer_names__[order]

    ds = ws.auto_create_dataset(
        workspace,
        analyzer_type,
        __analyzer_group__,
        __analyzer_dtype__,
        fill=__analyzer_fill__,
    )

    ds.set_attr("kind", analyzer_type)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace: String, full: SmartBoolean = False, order: Int = 0):
    filter = __analyzer_names__[order]
    datasets = ws.existing_datasets(workspace, group=__analyzer_group__, filter=filter)
    if full:
        return {
            "{}/{}".format(__analyzer_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    return {k: dataset_repr(v) for k, v in datasets.items()}


@hug.get()
def remove(workspace: String, analyzer_id: String):
    ws.delete_dataset(workspace, analyzer_id, group=__analyzer_group__)


@hug.get()
def rename(workspace: String, analyzer_id: String, new_name: String):
    ws.rename_dataset(workspace, analyzer_id, __analyzer_group__, new_name)
