
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




@features.get("/structure_tensor_determinant", response_model=None)
@save_metadata
def structure_tensor_determinant(src: str, dst: str, sigma: List[int] = Query()) -> "BLOB":
    from survos2.server.filtering.blob import compute_structure_tensor_determinant

    map_blocks(
        compute_structure_tensor_determinant,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=True,
    )

    simple_norm(dst)


@features.get("/frangi", response_model=None)
@save_metadata
def frangi(
    src: str,
    dst: str,
    scale_min: float = 1.0,
    scale_max: float = 4.0,
    scale_step: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma=15,
) -> "BLOB":
    from survos2.server.filtering.blob import compute_frangi

    map_blocks(
        compute_frangi,
        src,
        out=dst,
        scale_range=(scale_min, scale_max),
        scale_step=1.0,
        alpha=0.5,
        beta=0.5,
        gamma=15,
        dark_response=True,
        normalize=True,
        pad=max(4, int((scale_max * 2))),
    )

    simple_norm(dst)


@features.get("/hessian_eigenvalues", response_model=None)
@save_metadata
def hessian_eigenvalues(src: str, dst: str, sigma: List[int] = Query()) -> "BLOB":
    from survos2.server.filtering.blob import hessian_eigvals_image

    map_blocks(
        hessian_eigvals_image,
        src,
        out=dst,
        pad=max(4, int((max(sigma) * 2))),
        sigma=sigma,
        normalize=True,
    )

    simple_norm(dst)
