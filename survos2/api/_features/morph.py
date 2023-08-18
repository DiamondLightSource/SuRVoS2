
import numpy as np
from loguru import logger

from survos2.api import workspace as ws

from survos2.api.utils import save_metadata, get_function_api, dataset_repr, pass_through, simple_norm, rescale_denan 

from survos2.improc import map_blocks
from survos2.data_io import dataset_from_uri
from survos2.utils import encode_numpy, encode_numpy_slice
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.api._features.wavelet import features as wavelet

from typing import List
import pickle
from fastapi import APIRouter, Query, File, UploadFile


features = APIRouter()




@features.get("/dilation", response_model=None)
@save_metadata
def dilation(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from survos2.server.filtering import dilate

    map_blocks(
        dilate,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/erosion", response_model=None)
@save_metadata
def erosion(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from survos2.server.filtering import erode

    map_blocks(
        erode,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/opening", response_model=None)
@save_metadata
def opening(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from survos2.server.filtering import opening

    map_blocks(
        opening,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/closing", response_model=None)
@save_metadata
def closing(src: str, dst: str, num_iter: int = 1) -> "MORPHOLOGY":
    from survos2.server.filtering import closing

    map_blocks(
        closing,
        src,
        num_iter=num_iter,
        out=dst,
        normalize=True,
        pad=max(4, int(num_iter * 2)),
    )


@features.get("/distance_transform_edt", response_model=None)
@save_metadata
def distance_transform_edt(src: str, dst: str) -> "MORPHOLOGY":
    from survos2.server.filtering import distance_transform_edt

    logger.debug(f"Calculating distance transform")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = distance_transform_edt(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/skeletonize", response_model=None)
@save_metadata
def skeletonize(src: str, dst: str) -> "MORPHOLOGY":
    from survos2.server.filtering import skeletonize

    logger.debug(f"Calculating medial axis")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = skeletonize(src_dataset_arr)

    map_blocks(pass_through, filtered, out=dst, normalize=False)
