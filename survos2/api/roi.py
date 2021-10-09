import logging
import os.path as op
import dask.array as da
import hug
import numpy as np
from skimage.segmentation import slic
from ast import literal_eval
from loguru import logger


import survos2
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
from survos2.api.utils import dataset_repr, save_metadata
from survos2.improc import map_blocks
from survos2.io import dataset_from_uri
from survos2.utils import encode_numpy
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel

__export_fill__ = 0
__export_dtype__ = "float32"
__export_group__ = "roi"


@hug.get()
def create(workspace: String, roi_fname: String):
    original_ws = DataModel.g.current_workspace
    roi_dict = {}
    DataModel.g.current_workspace = workspace
    print(workspace)
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        ds_metadata = src_dataset.get_metadata()
        print(ds_metadata)
        if not "roi_fnames" in ds_metadata:
            src_dataset.set_metadata("roi_fnames", roi_dict)
        else:
            roi_dict = src_dataset.get_metadata("roi_fnames")
        num_entries = len(roi_dict.keys())
        roi_dict[num_entries+1] = roi_fname
        print(roi_dict)
        src_dataset.set_metadata("roi_fnames", roi_dict)
        metadata = dict()
        metadata['id'] = len(roi_dict.keys())
        metadata['name'] = roi_fname
        print(metadata)
        return metadata
    DataModel.g.current_workspace = original_ws


@hug.get()
def pull_anno(roi_fname : String):
    roi_ws = ws.get(roi_fname)
    print(roi_ws)
    #DataModel.g.current_workspace = roi_ws
    ds = ws.get_dataset(roi_fname, '001_level', group="annotations")
    #labels = ds.get_metadata("labels", {})
    print(ds[:].shape)
    print(ds[:])
    roi_parts = roi_fname.split("_")
    z_min = int(roi_parts[-6])
    z_max = int(roi_parts[-5])
    x_min = int(roi_parts[-4])
    x_max = int(roi_parts[-3])
    y_min = int(roi_parts[-2])
    y_max = int(roi_parts[-1])

    dst = DataModel.g.dataset_uri('001_level', group="annotations")
    main_anno = dataset_from_uri(dst, mode="rw")
    main_anno[z_min:z_max, x_min:x_max, y_min:y_max] = ds[:]
    # with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
    #     src_dataset = DM.sources[0]

@hug.get()
def existing():
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        ds_metadata = src_dataset.get_metadata()
        if not "roi_fnames" in ds_metadata:
            src_dataset.set_metadata("roi_fnames", {})
            return {}
        roi_fnames = ds_metadata["roi_fnames"]
        return roi_fnames


@hug.get()
def remove(workspace: String, roi_fname: String):
    src = DataModel.g.dataset_uri("__data__")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        roi_fnames = src_dataset.get_metadata("roi_fnames")
        for k,v in roi_fnames.items():
            if (v==roi_fname):
                selected = k    
        del roi_fnames[selected]    
        src_dataset.set_metadata("roi_fnames", roi_fnames)
