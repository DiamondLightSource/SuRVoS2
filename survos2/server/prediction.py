
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



# Sklearn
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
from survos2.server.filtering import crop_and_resample, prepare_features, generate_features
from survos2.server.supervoxels import prepare_supervoxels, generate_supervoxels
from survos2.utils import logger


from survos2.server.filtering import generate_features, simple_laplacian
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm

from survos2.server.config import appState
scfg = appState.scfg



from survos2.server.model import Superregions

def make_prediction(features_stack, annotation_volume, sr : Superregions, predict_params):
    """Prepare superregions and predict
    
    Arguments:
        features_stack {stacked image volumes} -- Stack of filtered volumes to use for features
        annotation_volume {image volume} -- Annotation volume (label image)
        supervoxel_vol {image volume} -- Labeled image defining superregions.
    
    Returns:
        (volume, volume, volume) -- tuple of raw prediction, prediction mapped to labels and confidence map
    """

    logger.debug(f"Feature stack: {features_stack.shape}")
    
    #try:
    #    supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
    #    supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
    #    supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=18)

    logger.debug(f"Using annotation volume of shape {annotation_volume.shape}")
    Yr = rlabels(annotation_volume.astype(np.uint16), sr.supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    logger.debug(f"Unique labels in anno: {np.unique(annotation_volume)}")
    #except Exception as err:    
    #    logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {sr.supervoxel_vol.shape}")

    i_train = Yr > -1

    logger.debug(f"i_train {i_train}")
    logger.debug(f"supervoxel_features: {sr.supervoxel_features}")
    X_train = sr.supervoxel_features[i_train]


    #
    # Projection
    #

    #proj = PCA(n_components='mle', whiten=True, random_state=42)
    #proj = StandardScaler()
    
    #proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=42)
    #rnd = 42
    #proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
    #X_train = proj.fit_transform(X_train)

    #print(X_train)
    
    Y_train = Yr[i_train]
    
    
    #clf = train(X_train, Y_train, 
    #            n_estimators=15 ,
    #            project=predict_params['proj'])

    clf = train(X_train, Y_train, n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

    logger.debug(f"Predicting with clf: {clf}")
    
    P = predict(sr.supervoxel_features, clf, label=True, probs=True )#, proj=predict_params['proj'])

    
    probs = P['probs']
    
    prob_map = invrmap(P['class'], sr.supervoxel_vol)
    num_supervox = sr.supervoxel_vol.max() + 1
    #class_labels = P['class'] #- 1

    pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
    #conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)
    conf_map = invrmap(P['probs'], sr.supervoxel_vol)
    
    #full_svmask = np.zeros(num_supervox, np.bool)
    #full_svmask[supervoxel_vol.ravel()] = True 
    
    from survos2.server.model import SRPrediction

    srprediction = SRPrediction(prob_map, conf_map, probs)
    
    #except Exception as err:
    #    logger.error(f"Prediction exception: {err}")
    #    #return 0
        
    return srprediction





# original rlabels always was hanging, replaced with quick numba fix

# y: labels (annotation vol)
# R: superregions 
# nr: number of superregions
# ny: number of labels
@jit
def simple_rlabels(y, R, ny, nr, min_ratio):
    
    N = R.shape[0]

    sizes = np.zeros(nr, dtype=np.uint32)
    counts = np.zeros((nr, ny), dtype=np.uint32)
    out = np.zeros(nr, dtype=np.uint32)

    for i in range(N):
        label = y[i]
        region = R[i]
        sizes[region] += 1

        if label > 0:
            counts[region, label] += 1

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




# original rlabels always was hanging, replaced with quick numba fix

# y: labels (annotation vol)
# R: superregions 
# nr: number of superregions
# ny: number of labels
@jit
def simple_rlabels(y, R, ny, nr, min_ratio):
    
    N = R.shape[0]

    sizes = np.zeros(nr, dtype=np.uint32)
    counts = np.zeros((nr, ny), dtype=np.uint32)
    out = np.zeros(nr, dtype=np.uint32)

    for i in range(N):
        label = y[i]
        region = R[i]
        sizes[region] += 1

        if label > 0:
            counts[region, label] += 1

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




def calc_feats(cropped_vol):
    scfg = appState.scfg
    img_vol =cropped_vol
    roi_crop = scfg.roi_crop

    roi_crop = [0,cropped_vol.shape[0],0,cropped_vol.shape[1], 0,cropped_vol.shape[2]]
    
    feature_params = [ [gaussian, scfg.filter1['gauss_params']],
    [gaussian, scfg.filter2['gauss_params'] ],
    [simple_laplacian, scfg.filter4['laplacian_params'] ],
    [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]]


    #feature_params = [ 
    #    [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]
    #]

    feats = generate_features(img_vol, feature_params, roi_crop, 1.0)
    return feats
    
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
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=18)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, 
                n_estimators=predict_params['n_estimators']) #, project=predict_params['proj'])

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

        #full_svmask = np.zeros(num_supervox, np.bool)
        #full_svmask[supervoxel_vol.ravel()] = True 

    except Exception as err:
        logger.error(f"Prediction exception: {err}")

    return prob_map, probs, pred_map, conf_map, P


def _train_classifier(clf, X_train, y_train, rnd=42, project=None):

    if ast.literal_eval(project) is not None:
        
        log.info('+ Projecting features')
        
        if project == 'rproj':
            proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=rnd)
        elif project == 'std':
            proj = StandardScaler()
        elif project == 'pca':
            proj = PCA(n_components='mle', whiten=True, random_state=rnd)
        else:
            log.error('Projection {} not available'.format(project))
            return

        X_train = proj.fit_transform(X_train)

    log.info('+ Training classifier')
    clf.fit(X_train, y_train)

def _classifier_predict(X, clf):
    result = {}
    log.info('+ Predicting labels')
    result['class'] = clf.predict(X)
    result['probs'] = clf.predict_proba(X)
    return result


clf_p = {'clf': 'ensemble',
         'type': 'rf',
         'n_estimators': 15,
        'max_depth' : 20,
        'n_jobs': 1}

def _obtain_classifier(clf_p):
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
        #elif project == 'rpca':
        #    proj = RandomizedPCA(whiten=True, random_state=rnd)
        elif project == 'rbf':
            proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
        else:
            raise Error('Projection {} not available'.format(project))

        X_train = proj.fit_transform(X_train)

    kwargs.setdefault('random_state', rnd)
    clf = RandomForestClassifier(**kwargs)
    
    
    #clf = ExtraTreesClassifier(**kwargs)
    #clf = GradientBoostingClassifier(**kwargs)
    #clf = SVC(
    #              probability=True)
    #clf = AdaBoostClassifier(**kwargs)
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


def make_prediction2(features_stack, annotation_volume, supervoxel_vol, predict_params):
    """Prepare superregions and predict
    
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
        supervoxel_rag = create_rag(np.array(supervoxel_vol).astype(np.uint32), connectivity=18)

        Yr = rlabels(annotation_volume.astype(np.uint16), supervoxel_vol.astype(np.uint32))  # unsigned char and unsigned int required

    except Exception as err:
        
        logger.info(f"Supervoxel rag and feature generation exception {err}")

    logger.debug(f"Supervoxels: {supervoxel_vol.shape}")

    i_train = Yr > -1
    X_train = supervoxel_features[i_train]
    
    #proj = PCA(n_components='mle', whiten=True, random_state=42)
    #proj = StandardScaler()
    proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=42)
    rnd = 42
    #proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)

    X_train = proj.fit_transform(X_train)
    
    print(X_train)
    Y_train = Yr[i_train]

    clf = train(X_train, Y_train, 
                n_estimators=55 ,
                project=predict_params['proj'])

    logger.debug(f"clf: {clf}")

    
    try:
        P = predict(X_train, clf, label=True, probs=True )#, proj=predict_params['proj'])
        probs = P['probs']
        prob_map = invrmap(P['class'], supervoxel_vol)
        num_supervox = supervoxel_vol.max() + 1
        class_labels = P['class'] #- 1
        
        pred_map = np.empty(num_supervox, dtype=P['class'].dtype)
        conf_map = np.empty(num_supervox, dtype=P['probs'].dtype)

        conf_map = invrmap(P['probs'], supervoxel_vol)

        #full_svmask = np.zeros(num_supervox, np.bool)
        #full_svmask[supervoxel_vol.ravel()] = True 
        logger.debug("Finished prediction.")

        from survos2.server.model import SRPrediction

        srprediction = SRPrediction(prob_map, conf_map, probs)

    except Exception as err:
        logger.error(f"Prediction exception: {err}")
        return 0
    return srpredicton

