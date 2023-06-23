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




@features.get("/spatial_gradient_3d", response_model=None)
@save_metadata
def spatial_gradient_3d(src: str, dst: str, dim: int = 0) -> "EDGES":
    from survos2.server.filtering import spatial_gradient_3d

    map_blocks(
        spatial_gradient_3d,
        src,
        out=dst,
        dim=dim,
        normalize=True,
    )

    simple_norm(dst)


@features.get("/difference_of_gaussians", response_model=None)
@save_metadata
def difference_of_gaussians(
    src: str, dst: str, sigma: List[int] = Query(), sigma_ratio: float = 2
) -> "EDGES":
    from survos2.server.filtering.edge import compute_difference_gaussians

    if isinstance(sigma, float) or isinstance(sigma, int):
        sigma = np.array([sigma] * 3)
    map_blocks(
        compute_difference_gaussians,
        src,
        out=dst,
        sigma=sigma,
        sigma_ratio=sigma_ratio,
        pad=max(4, int((np.max(sigma) * 2))),
        normalize=True,
    )
    simple_norm(dst)



@features.get("/laplacian", response_model=None)
@save_metadata
def laplacian(src: str, dst: str, sigma: List[int] = Query()) -> "EDGES":
    from survos2.server.filtering import ndimage_laplacian

    map_blocks(
        ndimage_laplacian,
        src,
        out=dst,
        kernel_size=sigma,
        # pad=max(4, int(max(np.array(kernel_size))) * 3),
        pad=max(4, int((max(sigma) * 2))),
        normalize=False,
    )

    simple_norm(dst)
