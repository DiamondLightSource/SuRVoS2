import ntpath
import numpy as np
from loguru import logger
from skimage.segmentation import slic
from survos2.api import workspace as ws
from survos2.api.utils import dataset_repr, save_metadata
from survos2.data_io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy
from fastapi import APIRouter

__region_fill__ = 0
__region_dtype__ = "uint32"
__region_group__ = "superregions"
__region_names__ = [None, "supervoxels"]

superregions = APIRouter()


@superregions.get("/get_volume")
def get_volume(src: str):
    logger.debug("Getting region volume")
    ds = dataset_from_uri(src, mode="r")
    data = ds[:]
    return encode_numpy(data)


@superregions.get("/get_slice")
def get_slice(src: str, slice_idx: int, order: tuple):
    ds = dataset_from_uri(src, mode="r")[:]
    ds = np.transpose(ds, order)
    data = ds[slice_idx]
    return encode_numpy(data)


@superregions.get("/get_crop")
def get_crop(src: str, roi: list):
    logger.debug("Getting regions crop")
    ds = dataset_from_uri(src, mode="r")
    data = ds[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]]
    return encode_numpy(data)


@superregions.get("/supervoxels")
@save_metadata
def supervoxels(
    src: str,
    dst: str,
    mask_id: str,
    n_segments: int = 10,
    compactness: float = 20,
    spacing: list = [1, 1, 1],
    multichannel: bool = False,
    enforce_connectivity: bool = False,
    out_dtype: str = "int",
    zero_parameter: bool = False,
    max_num_iter: int = 10,
):
    with DatasetManager(src, out=None, dtype=out_dtype, fillvalue=0) as DM:
        src_data_arr = DM.sources[0][:]

    # get image feature for mask, if any

    if mask_id == "None":
        mask_feature = None
    else:
        src = DataModel.g.dataset_uri(ntpath.basename(mask_id), group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            mask_feature = DM.sources[0][:].astype(np.uint32)
            logger.debug(f"Feature to use as mask shape {mask_feature.shape}")

    supervoxel_image = slic(
        src_data_arr,
        n_segments=n_segments,
        spacing=spacing,
        compactness=compactness,
        multichannel=False,
        max_num_iter=max_num_iter,
        slic_zero=zero_parameter,
        mask=mask_feature,
    )

    def pass_through(x):
        return x

    map_blocks(pass_through, supervoxel_image, out=dst, normalize=False)


@superregions.get("/supervoxels_chunked")
@save_metadata
def supervoxels_chunked(
    src: list,
    dst: str,
    n_segments: int = 10,
    compactness: float = 20,
    spacing: list = [1, 1, 1],
    multichannel: bool = False,
    enforce_connectivity: bool = False,
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

    num_sv = len(np.unique(supervoxel_image))
    logger.debug(f"Number of supervoxels created: {num_sv}")

    dst_dataset.set_attr("num_supervoxels", num_sv)


@superregions.get("/create")
def create(workspace: str, order: int = 1, big: bool = False):
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
            # dtype=np.uint32,
            fill=__region_fill__,
        )

    ds.set_attr("kind", region_type)
    return dataset_repr(ds)


@superregions.get("/existing")
def existing(workspace: str, full: bool = False, order: int = 1):
    filter = __region_names__[order]
    datasets = ws.existing_datasets(workspace, group=__region_group__, filter=filter)
    if full:
        return {"{}/{}".format(__region_group__, k): dataset_repr(v) for k, v in datasets.items()}
    return {k: dataset_repr(v) for k, v in datasets.items()}


@superregions.get("/remove")
def remove(workspace: str, region_id: str):
    ws.delete_dataset(workspace, region_id, group=__region_group__)

    return {"done": True}


@superregions.get("/rename")
def rename(workspace: str, region_id: str, new_name: str):
    ws.rename_dataset(workspace, region_id, __region_group__, new_name)
