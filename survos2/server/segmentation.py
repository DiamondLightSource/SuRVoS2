"""
Core segmentation functions for SuRVoS Super-region segmentation

BUGS
slic can fail (PyCuda stack pop error) if the input feature is not sensible 
    all zeros is easily done (FIX)
    
    many crashes are occuring due to memory issues in pycuda as a result of too many superregions being created due to:
        overly complex/not smoothed input
        too large input
        too small supervoxels
 
    too large supervoxels also crashes

TODO
    better validate supervoxel generation params
    
"""
import os
import sys
from enum import Enum
from typing import List
import numpy as np
import pandas as pd
import uvicorn
import joblib
import pandas as pd
import glob

import h5py
import ntpath
import scipy
import yaml
from numba import jit
from collections import namedtuple
from skimage import img_as_ubyte, img_as_float
from skimage import io

from scipy import ndimage

#from immerframe import Proxy
#from starlette.requests import Request

#
# SuRVoS 2 imports
#

from functools import partial

from survos2.improc import map_blocks
from survos2.improc.features import gaussian, tvdenoising3d
from survos2.improc.regions.rag import create_rag
from survos2.improc.regions.slic import slic3d
from survos2.improc.segmentation import _qpbo as qpbo
#from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.appearance import train, predict, refine, invrmap
from survos2.improc.segmentation.mappings import rmeans, normalize
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import decode_numpy, encode_numpy
from survos2.api.utils import save_metadata, dataset_repr
from survos2.helpers import AttrDict

import survos2.server.workspace as ws
from survos2.server.filtering import crop_and_resample, prepare_features, generate_features
from survos2.server.supervoxels import prepare_supervoxels, generate_supervoxels
from survos2.utils import logger


# original rlabels always was hanging, replaced with quick numba fix
@jit
def simple_rlabels(y, R, ny, nr, min_ratio):
    
    N = R.shape[0]

    sizes = np.zeros(nr, dtype=np.uint32)
    counts = np.zeros((nr, ny), dtype=np.uint32)
    out = np.zeros(nr, dtype=np.uint32)

    for i in range(N):
        l = y[i]
        s = R[i]
        sizes[s] += 1

        if l > 0:
            counts[s, l] += 1

    for i in range(nr):
        cmax = 0
        smin = sizes[i] * min_ratio

        for j in range(1, ny):
            curr = counts[i, j]

            if curr > cmax and curr >= smin:
                cmax = curr
                out[i] = j

    return out


def rlabels(y : np.uint16, R : np.uint32, nr : int =None, ny : int=None, norm:bool=None, min_ratio:int=0):
    
    #WARNING: silently fails if types are not exactly correct
    y = np.array(y)
    R = np.array(R)
    
    nr = nr or R.max() + 1
    ny = ny or y.max() + 1

    logger.info("running rlabels")

    logger.debug(str(type(y)))
    logger.debug(str(type(R)))
    logger.debug((y.shape, R.shape))

    try:
        features = simple_rlabels(y.ravel(), R.ravel(), ny, nr, min_ratio)

    except Exception as err:
        logger.error(f"simple_rlabels exception: {err}")

    # features = mappings._rlabels(y.ravel(), R.ravel(), ny, nr, min_ratio)

    logger.debug(f"Features {features}")

    return features
    #return normalize(features).astype(np.uint16, norm=norm)

#
# Predict
#


    
def process_anno_and_predict(features_stack, annotation_volume, supervoxel_vol, predict_params):
    """Main superregion 
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        supervoxel_vol {image volume} -- Labeled image defining superregions.
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=6)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

    logger.debug(f"clf: {clf}")

    try:
        P = predict(supervoxel_features, clf, label=True, probs=True) # proj=predict_params['proj'])
        
        probs = P['probs']
        prob_map = invrmap(P['class'], supervoxel_vol)
        num_supervox = supervoxel_vol.max() + 1
        class_labels = P['class'] #- 1
        
        pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
        conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)

        conf_map = invrmap(P['probs'], supervoxel_vol)

        full_svmask = np.zeros(num_supervox, np.bool)
        full_svmask[supervoxel_vol.ravel()] = True 

    except Exception as err:
        logger.error(f"Prediction exception: {err}")

    return prob_map, probs, pred_map, conf_map, P



def process_anno_and_predict_old(features_stack, annotation_volume, supervoxel_vol, predict_params):
    """Main superregion 
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        supervoxel_vol {image volume} -- Labeled image defining superregions.
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=6)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape, Yr.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

    logger.debug(f"clf: {clf}")

    try:
        P = predict(supervoxel_features, clf, label=True, probs=True) # proj=predict_params['proj'])
        probs = P['probs']
        prob_map = invrmap(P['class'], supervoxel_vol)
        num_supervox = supervoxel_vol.max() + 1
        class_labels = P['class'] #- 1
        
        pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
        conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)

        conf_map = invrmap(P['probs'], supervoxel_vol)

        full_svmask = np.zeros(num_supervox, np.bool)
        full_svmask[supervoxel_vol.ravel()] = True 

    except Exception as err:
        logger.error(f"Prediction exception: {err}")

    return prob_map, probs, pred_map, conf_map


def mrf_refinement(P, supervoxel_vol, features_stack, lam=0.5, gamma=False):
        
    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=6)
    
        unary = (-np.ma.log(P['probs'])).filled()
        pred = P['class']
        labels = np.asarray(list(set(np.unique(P['class'][supervoxel_vol])) - set([-1])), np.int32)
        unary = unary.astype(np.float32)
        mapping = np.zeros(pred.max() + 1, np.int32)
        mapping[labels] = np.arange(labels.size)
        idx = np.where(pred > -1)[0]
        col = mapping[pred[idx]]
        unary[idx, col] = 0
        y_ref = refine(supervoxel_rag, unary, supervoxel_rag, lam, gamma=gamma)
        Rp_ref = invrmap(y_ref, supervoxel_vol)

        logger.debug(f"Calculated mrf refinement with shape: {Rp_ref.shape}")
        
    except Exception as err:
        logger.error(f"Supervoxel generation exception: {err}")

    return Rp_ref

