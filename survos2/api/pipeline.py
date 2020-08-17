
import hug
import logging
import os.path as op

import numpy as np
import dask.array as da

from survos2.utils import encode_numpy
from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.api.utils import save_metadata, dataset_repr
from survos2.api.types import DataURI, Float, SmartBoolean, \
    FloatOrVector, IntList, String, Int, FloatList, DataURIList
from survos2.api import workspace as ws

from loguru import logger


@hug.get() #@save_metadata
def prepare_payload(src:DataURI, dst:DataURI, lamda:Float=10,
                    max_iter:Int=100) -> 'Denoising':
    """
    API wrapper around `survos2.improc.features.tv.tvdenoising3d`.
    """
    from ..improc.features.tv import tvdenoising3d
    map_blocks(tvdenoising3d, src, out=dst, lamda=lamda, max_iter=max_iter,
               normalize=True)


@hug.get()
def run_pipeline(workspace:String, ):
    logger.debug("Run pipeline")
