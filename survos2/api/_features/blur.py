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




@features.get("/tvdenoise", response_model=None)
@save_metadata
def tvdenoise(
    src: str,
    dst: str,
    regularization_amount: float = 0.001,
    pad: int = 8,
    max_iter: int = 100,
) -> "DENOISING":
    from survos2.server.filtering.blur import tvdenoise_kornia

    map_blocks(
        tvdenoise_kornia,
        src,
        out=dst,
        regularization_amount=regularization_amount,
        max_iter=max_iter,
        pad=pad,
        normalize=True,
    )



@features.get("/gaussian_blur", response_model=None)
@save_metadata
def gaussian_blur(src: str, dst: str, sigma: List[int] = Query()) -> "DENOISING":
    from survos2.server.filtering import gaussian_blur_kornia

    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = np.array([sigma] * 3)

    if sigma[0] == 0:
        from skimage.filters import gaussian

        map_blocks(
            gaussian,
            src,
            out=dst,
            sigma=(1.0, sigma[1], sigma[2]),
            pad=0,
            normalize=False,
        )

    else:
        map_blocks(
            gaussian_blur_kornia,
            src,
            out=dst,
            sigma=sigma,
            pad=max(4, int((max(sigma) * 2))),
            normalize=False,
        )



@features.get("/gaussian_norm", response_model=None)
@save_metadata
def gaussian_norm(src: str, dst: str, sigma: List[int] = Query()) -> "NEIGHBORHOOD":
    from survos2.server.filtering.blur import gaussian_norm

    map_blocks(
        gaussian_norm,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=False,
    )

    simple_norm(dst)


@features.get("/gaussian_center", response_model=None)
@save_metadata
def gaussian_center(src: str, dst: str, sigma: List[int] = Query()) -> "NEIGHBORHOOD":
    from survos2.server.filtering.blur import gaussian_center

    map_blocks(
        gaussian_center,
        src,
        out=dst,
        sigma=sigma,
        pad=max(4, int((max(sigma) * 2))),
        normalize=False,
    )
    simple_norm(dst)


@features.get("/median", response_model=None)
@save_metadata
def median(src: str, dst: str, median_size: int = 1, num_iter: int = 1) -> "DENOISING":
    from survos2.server.filtering import median

    map_blocks(
        median,
        src,
        median_size=median_size,
        num_iter=num_iter,
        out=dst,
        pad=max(4, int((median_size * 2))),
        normalize=False,
    )

