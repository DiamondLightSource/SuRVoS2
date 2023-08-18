import numpy as np
from loguru import logger

from survos2.api import workspace as ws

from survos2.api.utils import dataset_repr, get_function_api
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



@features.get("/raw", response_model=None)
@save_metadata
def raw(src: str, dst: str) -> "BASE":
    map_blocks(pass_through, src, out=dst, normalize=True)


@features.get("/simple_invert", response_model=None)
@save_metadata
def simple_invert(src: str, dst: str) -> "BASE":
    from survos2.server.filtering import simple_invert

    map_blocks(simple_invert, src, out=dst, normalize=True)


@features.get("/invert_threshold", response_model=None)
@save_metadata
def invert_threshold(src: str, dst: str, thresh: float = 0.5) -> "BASE":
    from survos2.server.filtering import invert_threshold

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = invert_threshold(src_dataset_arr, thresh=thresh)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/threshold", response_model=None)
@save_metadata
def threshold(src: str, dst: str, threshold: float = 0.5) -> "BASE":
    from survos2.server.filtering import threshold as threshold_fn

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        filtered = threshold_fn(src_dataset_arr, thresh=threshold)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/rescale", response_model=None)
@save_metadata
def rescale(src: str, dst: str) -> "BASE":
    logger.debug(f"Rescaling src {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0][:]

        filtered = rescale_denan(src_dataset)

    map_blocks(pass_through, filtered, out=dst, normalize=False)


@features.get("/gamma_correct", response_model=None)
@save_metadata
def gamma_correct(src: str, dst: str, gamma: float = 1.0) -> "BASE":
    from survos2.server.filtering import gamma_adjust

    map_blocks(gamma_adjust, src, gamma=gamma, out=dst, normalize=True)

