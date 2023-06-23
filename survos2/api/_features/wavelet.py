import numpy as np
from loguru import logger

from survos2.api import workspace as ws
from survos2.api.utils import save_metadata, get_function_api, dataset_repr, pass_through, simple_norm, rescale_denan 

from survos2.api.utils import dataset_repr, get_function_api
from survos2.api.utils import save_metadata, dataset_repr
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.api.utils import pass_through
from fastapi import APIRouter


features = APIRouter()


@features.get("/wavelet", response_model=None)
@save_metadata
def wavelet(
    src: str,
    dst: str,
    threshold: float = 64.0,
    level: int = 1,
    wavelet: str = "sym3",
    hard: bool = True,
) -> "WAVELET":
    from survos2.server.filtering import wavelet as wavelet_fn

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    result = wavelet_fn(
        src_dataset_arr, level=level, wavelet=str(wavelet), threshold=threshold, hard=hard
    )
    map_blocks(pass_through, result, out=dst, normalize=False)
