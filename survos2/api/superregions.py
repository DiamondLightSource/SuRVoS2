import logging
import os.path as op

import dask.array as da
import hug
import numpy as np
from skimage.segmentation import slic


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
from skimage.segmentation import slic
from survos2.model import DataModel


__region_fill__ = 0
__region_dtype__ = "uint32"
__region_group__ = "superregions"
__region_names__ = [None, "supervoxels"]


@hug.get()
def get_volume(src: DataURI):
    logger.debug("Getting region volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@hug.get()
def get_slice(src: DataURI, slice_idx: Int, order: tuple):
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order)
    data = ds[slice_idx]
    return encode_numpy(data)


@hug.get()
def get_crop(src: DataURI, roi: IntList):
    logger.debug("Getting regions crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@hug.get()
@save_metadata
def supervoxels(
    src: DataURI,
    dst: DataURI,
    n_segments: Int = 10,
    compactness: Float = 20,
    spacing: FloatList = [1, 1, 1],
    multichannel: SmartBoolean = False,
    enforce_connectivity: SmartBoolean = False,
    out_dtype="int",
):
    with DatasetManager(src, out=None, dtype=out_dtype, fillvalue=0) as DM:
        src_data_arr = DM.sources[0][:]

    supervoxel_image = slic(
        src_data_arr,
        n_segments=n_segments,
        spacing=spacing,
        compactness=compactness,
        multichannel=False,
    )
    print(supervoxel_image)

    def pass_through(x):
        return x

    map_blocks(pass_through, supervoxel_image, out=dst, normalize=False)


@hug.get()
@save_metadata
def supervoxels_chunked(
    src: DataURIList,
    dst: DataURI,
    n_segments: Int = 10,
    compactness: Float = 20,
    spacing: FloatList = [1, 1, 1],
    multichannel: SmartBoolean = False,
    enforce_connectivity: SmartBoolean = False,
    out_dtype="int",
):

    map_blocks(
        slic,
        *src,
        out=dst,
        n_segments=n_segments,
        spacing=spacing,
        compactness=compactness,
        multichannel=False,
        enforce_connectivity=True,
        stack=False,
        timeit=True,
        uses_gpu=False,
        out_dtype=out_dtype,
        relabel=True,
    )

    with DatasetManager(dst, out=None, dtype=out_dtype, fillvalue=0) as DM:
        dst_dataset = DM.sources[0]
        supervoxel_image = dst_dataset[:]
        print(supervoxel_image.dtype)

    num_sv = len(np.unique(supervoxel_image))
    print(f"Number of supervoxels created: {num_sv}")

    dst_dataset.set_attr("num_supervoxels", num_sv)


@hug.get()
@save_metadata
def connected_components(src: DataURI, dst: DataURI, remap: SmartBoolean):
    """
    API wrapper for `survos2.improc.regions.ccl3d`.
    """
    from ..improc.regions.ccl import ccl3d

    map_blocks(ccl3d, src, out=dst, remap=remap)


@hug.get()
@save_metadata
def merge_regions(src: DataURI, labels: DataURI, dst: DataURI, min_size: Float):
    """
    API wrapper for `survos2.improc.regions.merge_small`.
    """
    from ..improc.regions.ccl import merge_small

    map_blocks(merge_small, src, labels, out=dst, min_size=min_size)


@hug.get()
def create(workspace: String, order: Int = 1, big: bool = False):
    region_type = __region_names__[order]
    if big:
        logger.debug("Creating int64 regions")
        ds = ws.auto_create_dataset(
            workspace,
            region_type,
            __region_group__,
            __region_dtype__,
            dtype=np.uint64,
            fill=__region_fill__,
        )
    else:
        logger.debug("Creating int32 regions")
        ds = ws.auto_create_dataset(
            workspace,
            region_type,
            __region_group__,
            __region_dtype__,
            dtype=np.uint32,
            fill=__region_fill__,
        )

    ds.set_attr("kind", region_type)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace: String, full: SmartBoolean = False, order: Int = 1):
    filter = __region_names__[order]
    datasets = ws.existing_datasets(workspace, group=__region_group__, filter=filter)
    if full:
        return {
            "{}/{}".format(__region_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    return {k: dataset_repr(v) for k, v in datasets.items()}


@hug.get()
def remove(workspace: String, region_id: String):
    ws.delete_dataset(workspace, region_id, group=__region_group__)


@hug.get()
def rename(workspace: String, region_id: String, new_name: String):
    ws.rename_dataset(workspace, region_id, __region_group__, new_name)