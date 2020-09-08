""""
Supervoxel generation 


Things that go wrong with supervoxel generation:

The image used is not smooth enough. The algorithm generates far too many supervoxels and 
crashes if an overly detailed image is used as the source image.

TODO
+add padding before supervoxel generation


"""
import os
import sys
from typing import List
import numpy as np
import pandas as pd
import scipy

from collections import namedtuple
from skimage import img_as_ubyte, img_as_float
from skimage import io
from scipy import ndimage

#
# SuRVoS 2 imports
#

from functools import partial

from survos2.improc import map_blocks

# from survos2.improc.features import gaussian, tvdenoising3d
from survos2.improc.regions.rag import create_rag
from survos2.improc.regions.slic import slic3d
from survos2.improc.segmentation import _qpbo as qpbo
from survos2.improc.segmentation.appearance import train, predict, refine, invrmap
from survos2.improc.segmentation.mappings import rmeans
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import decode_numpy, encode_numpy
from survos2.api.utils import save_metadata, dataset_repr

import survos2.api.workspace as ws


from survos2.server.model import SRData

from loguru import logger


def generate_supervoxels(dataset_feats, filtered_stack, dataset_feats_idx, slic_params):
    """Generate a supervoxel volume image
    
    Arguments:
        dataset_feats {list of filtered volumes} -- list of filters of original input image volume
        filtered_stack {volume that is a stack of dataset_feats} -- reshaped version of dataset feats for rmeans
        dataset_feats_idx {int} -- index of the filter to use for supervoxel calculation
        slic_params {dict} -- Supervoxel generation parameters
    
    Returns: dataclass with all the information required for prediction
    """
    logger.info(f"Using feature idx {dataset_feats_idx} for supervoxels.")
    logger.info(
        f"SRFeatures for supervoxels have shape {dataset_feats[dataset_feats_idx].shape}"
    )
    logger.info(f"Generating supervoxels with params: {slic_params}")

    block_z, block_x, block_y = dataset_feats[0].shape

    # map_blocks through Dask
    supervoxel_vol = map_blocks(
        slic3d,
        dataset_feats[dataset_feats_idx].astype(np.float32),
        **slic_params,
        timeit=False,
    )
    supervoxel_vol = supervoxel_vol.astype(np.uint32, copy=True)  #
    logger.info(f"Finished slic with supervoxel vol of shape {supervoxel_vol.shape}")

    supervoxel_vol = supervoxel_vol[...]
    supervoxel_vol = np.asarray(supervoxel_vol)  # .astype(np.uint32, copy=True)
    supervoxel_vol = np.nan_to_num(supervoxel_vol)
    logger.info(
        f"Calling rmeans with filtered_stack { len(filtered_stack)} and supervoxel_vol {supervoxel_vol.shape}"
    )
    supervoxel_features = rmeans(filtered_stack, supervoxel_vol)

    logger.info(
        f"Finished rmeans with supervoxel_features of shape {supervoxel_features.shape}"
    )

    supervoxel_rag = create_rag(np.array(supervoxel_vol), connectivity=6)

    logger.info(
        "MaxMin SV Feat: {} {}".format(np.max(supervoxel_vol), np.min(supervoxel_vol))
    )

    superregions = SRData(supervoxel_vol, supervoxel_features, supervoxel_rag)

    return superregions


def superregion_factory(
    supervoxel_vol: np.ndarray, features_stack: np.ndarray
) -> SRData:
    supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32, copy=True)
    supervoxel_features = rmeans(features_stack, supervoxel_vol)
    logger.info(
        f"Finished rmeans with supervoxel_features of shape {supervoxel_features.shape}"
    )

    supervoxel_rag = create_rag(np.array(supervoxel_vol), connectivity=6)
    logger.info(
        "MaxMin SV Feat: {} {}".format(np.max(supervoxel_vol), np.min(supervoxel_vol))
    )

    superregions = SRData(supervoxel_vol, supervoxel_features, supervoxel_rag)

    return superregions


def prepare_supervoxels(
    supervoxels: List[str],
    filtered_stack: np.ndarray,
    roi_crop: np.ndarray,
    resample_amt: float,
) -> SRData:
    """Load supervoxels from file, then generate supervoxel features from a features stack and the supervoxel rag,
    then bundle as SRData and return.

    Args:
        supervoxels (List[str]): list of supervoxels
        filtered_stack (np.ndarray): stack of filters
        roi_crop (np.ndarray): roi to crop to
        resample_amt (float): zoom level

    Returns:
        [SRData]: superregions dataclass object 
    """

    logger.debug(f"Loading supervoxel file {supervoxels[0]}")
    logger.debug(f"Roi crop {roi_crop}")

    supervoxel_vol = dataset_from_uri(supervoxels[0], mode="r")
    supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32, copy=False)
    supervoxel_vol = np.nan_to_num(supervoxel_vol)

    supervoxel_proc = supervoxel_vol[
        roi_crop[0] : roi_crop[1], roi_crop[2] : roi_crop[3], roi_crop[4] : roi_crop[5]
    ].astype(np.uint32, copy=False)

    supervoxel_proc = scipy.ndimage.zoom(supervoxel_proc, resample_amt, order=1)

    logger.info(
        f"Loading Supervoxel {os.path.basename(supervoxels[0])} with shape {supervoxel_proc.shape}"
    )

    supervoxel_features = rmeans(filtered_stack, supervoxel_proc)
    supervoxel_rag = create_rag(
        np.array(supervoxel_proc).astype(np.uint32), connectivity=6
    )

    supervoxel_features = []
    supervoxel_rag = []

    superregions = SRData(supervoxel_proc, supervoxel_features, supervoxel_rag)

    return superregions
