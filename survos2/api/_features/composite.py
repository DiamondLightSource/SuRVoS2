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




@features.get("/feature_composite", response_model=None)
@save_metadata
def feature_composite(
    src: str,
    dst: str,
    workspace: str,
    feature_A: str,
    feature_B: str,
    op: str,
):
    DataModel.g.current_workspace = workspace
    src_A = DataModel.g.dataset_uri(feature_A, group="features")
    with DatasetManager(src_A, out=None, dtype="uint16", fillvalue=0) as DM:
        src_A_dataset = DM.sources[0]
        src_A_arr = src_A_dataset[:]
        logger.info(f"Obtained src A with shape {src_A_arr.shape}")

    src_B = DataModel.g.dataset_uri(feature_B, group="features")
    with DatasetManager(src_B, out=None, dtype="uint16", fillvalue=0) as DM:
        src_B_dataset = DM.sources[0]
        src_B_arr = src_B_dataset[:]
        logger.info(f"Obtained src B with shape {src_B_arr.shape}")
    if op == "+":
        result = src_A_arr + src_B_arr
    else:
        result = src_A_arr * src_B_arr

    map_blocks(pass_through, result, out=dst, normalize=False)
