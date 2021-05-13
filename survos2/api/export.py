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



__export_fill__ = 0
__export_dtype__ = "uint32"
__export_group__ = "export"
__region_names__ = [None, "supervoxels"]  # , 'megavoxels']



