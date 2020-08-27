import hug
import logging
import numpy as np
import dask.array as da
from loguru import logger

from survos2.utils import encode_numpy
from survos2.io import dataset_from_uri
from survos2.improc import map_blocks
from survos2.api.utils import save_metadata, dataset_repr
from survos2.api.types import DataURI, Float, SmartBoolean, \
    FloatOrVector, IntList, String, Int, FloatList, DataURIList
from survos2.api import workspace as ws
from survos2.server.supervoxels import superregion_factory
from survos2.server.features import features_factory
from survos2.server.superseg import sr_prediction

def prediction(feature_1: np.ndarray, feature_2: np.ndarray, 
            supervoxel_image: np.ndarray, anno_image: np.ndarray) -> np.ndarray:   
   
    feats = features_factory([feature_1, feature_2 ])

    logger.info(f"Number of features calculated: {len(feats.features_stack)}")
    
    sr = superregion_factory(supervoxel_image.astype(np.uint32), feats.features_stack)
    
    logger.info(f"Calculated superregions {sr}")

    srprediction = sr_prediction(feats.features_stack,
                                    anno_image.astype(np.uint16),
                                    sr,
                                    predict_params)
    
    logger.info(f"Made prediction {srprediction}")
    
    return srprediction.prob_map


@hug.get()
def run_pipeline(workspace: String, ):
    logger.debug("Run pipeline")

@hug.get()
def add_pipeline(workspace: String, ):
    logger.debug("Add pipeline")

@hug.get()
def list_pipelines(workspace: String, ):
    logger.debug("List pipeline")
