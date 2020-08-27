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

import joblib
import pandas as pd
import glob

import h5py
import ntpath
import scipy
import yaml
from numba import jit
from collections import namedtuple

from functools import partial
import numpy as np
import logging as log
import ast

import networkx as nx

import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import label_propagation
import sklearn
from sklearn import cluster, datasets, mixture
from sklearn.cluster import SpectralClustering

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import label_propagation
from sklearn import mixture

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import ExtraTreesClassifier, \
                             RandomForestClassifier, \
                             AdaBoostClassifier,\
                             GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans

from sklearn.linear_model import SGDClassifier

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import pairwise_distances

from skimage.segmentation import relabel_sequential
from skimage import img_as_ubyte, img_as_float
from skimage import io

from scipy import ndimage
from scipy.stats import entropy



from loguru import logger

#
# SuRVoS 2 imports
#

from functools import partial

from survos2.improc import map_blocks
from survos2.improc.features import gaussian, tvdenoising3d
from survos2.improc.regions.rag import create_rag
from survos2.improc.regions.slic import slic3d
from survos2.improc.segmentation import _qpbo as qpbo
from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.appearance import train, predict, refine, invrmap
from survos2.improc.segmentation.mappings import rmeans, normalize
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import decode_numpy, encode_numpy
from survos2.api.utils import save_metadata, dataset_repr
from survos2.helpers import AttrDict

from survos2.improc import map_blocks
from survos2.improc.features import gaussian, tvdenoising3d
from survos2.improc.regions.rag import create_rag
from survos2.improc.regions.slic import slic3d
from survos2.improc.segmentation import _qpbo as qpbo
from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.appearance import train, predict, refine, invrmap
from survos2.improc.segmentation.mappings import rmeans, normalize
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import decode_numpy, encode_numpy
from survos2.api.utils import save_metadata, dataset_repr
from survos2.helpers import AttrDict
import survos2.api.workspace as ws

from survos2.server.features import prepare_features, generate_features
from survos2.server.supervoxels import prepare_supervoxels, generate_supervoxels
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
from survos2.frontend.model import ClientData

from survos2.server.model import SRData, SRPrediction
from survos2.server.region_labeling import rlabels


def obtain_classifier(clf_p):
    
    if clf_p['clf'] == 'ensemble':
        mode = 'ensemble'

        if clf_p['type'] == 'rf':
            clf = RandomForestClassifier(n_estimators=clf_p['n_estimators'],
                                         max_depth=clf_p['max_depth'],
                                         n_jobs=clf_p['n_jobs'])

        elif clf_p['type'] == 'erf':
            clf = ExtraTreesClassifier(n_estimators=clf_p['n_estimators'],
                                       max_depth=clf_p['max_depth'],
                                       n_jobs=clf_p['n_jobs'])
        elif clf_p['type'] == 'ada':
            clf = AdaBoostClassifier(n_estimators=clf_p['n_estimators'],
                                     learning_rate=clf_p['learning_rate'])
        else:
            clf = GradientBoostingClassifier(n_estimators=clf_p['n_estimators'],
                                             max_depth=clf_p['max_depth'],
                                             learning_rate=clf_p['learning_rate'],
                                             subsample=clf_p['subsample'])
   
    elif clf_p['clf'] == 'svm':
        mode = 'svm'
        clf = SVC(C=clf_p['C'], gamma=clf_p['gamma'], kernel=clf_p['kernel'],
                  probability=True)
    
    elif clf_p['clf'] == 'sgd':
        mode = 'sgd'
        clf = SGDClassifier(loss=clf_p['loss'], penalty=clf_p['penalty'],
                            alpha=clf_p['alpha'], n_iter=clf_p['n_iter'])
    
    else:
        raise Exception('Classifier not supported')
    
    return clf, mode



def train(X_train, y_train, project=False, rnd=42, **kwargs):
    if project is not False:
        if project == 'rproj':
            proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=rnd)
        elif project == 'std':
            proj = StandardScaler()
        elif project == 'pca':
            proj = PCA(n_components='mle', whiten=True, random_state=rnd)
        elif project == 'rbf':
            proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
        else:
            raise Error('Projection {} not available'.format(project))

        X_train = proj.fit_transform(X_train)

    kwargs.setdefault('random_state', rnd)
    
    clf = RandomForestClassifier(**kwargs)
    
    clf.fit(X_train, y_train)

    if project is not False:
        return clf, proj

    return clf


def predict(X, clf, proj=None, label=True, probs=False, log=False):
    if proj is not None:
        X = proj.transform(X)
    result = {}
    if probs:
        result['probs'] = clf.predict_proba(X)
    if log:
        result['log_probs'] = clf.predict_log_proba(X)
    if label:
        result['class'] = clf.predict(X)
    return result



def sr_prediction(features_stack, annotation_volume, sr : SRData, predict_params):
    """Prepare superregions and predict
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        sr {image volume} -- Prepared superregions SRData
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    logger.debug(f"Using annotation volume of shape {annotation_volume.shape}")
    Yr = rlabels(annotation_volume.astype(np.uint16), sr.supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required
    logger.debug(f"Unique labels in anno: {np.unique(annotation_volume)}")
    
    i_train = Yr > -1

    logger.debug(f"i_train {i_train}")
    logger.debug(f"supervoxel_features: {sr.supervoxel_features}")
    X_train = sr.supervoxel_features[i_train]

    # Projection
    #proj = PCA(n_components='mle', whiten=True, random_state=42)
    #proj = StandardScaler()
    #proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=42)
    #rnd = 42
    #proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
    #X_train = proj.fit_transform(X_train)
    #print(X_train)
    
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

    logger.debug(f"Predicting with clf: {clf}")

    P = predict(sr.supervoxel_features, clf, label=True, probs=True )#, proj=predict_params['proj'])
    
    prob_map = invrmap(P['class'], sr.supervoxel_vol)
    num_supervox = sr.supervoxel_vol.max() + 1
    conf_map = invrmap(P['probs'], sr.supervoxel_vol)
        
    srprediction = SRPrediction(prob_map, conf_map)
    
    return srprediction



