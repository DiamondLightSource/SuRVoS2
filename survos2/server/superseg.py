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
from functools import partial
from typing import List
import numpy as np
import pandas as pd
import sklearn
from loguru import logger
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import SparseRandomProjection
from survos2.server.features import features_factory
from survos2.server.model import SRData, SRPrediction
from survos2.server.region_labeling import rlabels
from survos2.server.supervoxels import invrmap, superregion_factory
from survos2.server.state import cfg
from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.mappings import rmeans
from survos2.improc.regions.rag import create_rag

#
# SuRVoS 2 imports
#

PRED_MIN = 0


def obtain_classifier(clf_p):

    if clf_p["clf"] == "ensemble":
        mode = "ensemble"

        if clf_p["type"] == "rf":
            clf = RandomForestClassifier(
                n_estimators=clf_p["n_estimators"],
                max_depth=clf_p["max_depth"],
                n_jobs=clf_p["n_jobs"],
            )

        elif clf_p["type"] == "erf":
            clf = ExtraTreesClassifier(
                n_estimators=clf_p["n_estimators"],
                max_depth=clf_p["max_depth"],
                n_jobs=clf_p["n_jobs"],
            )
        elif clf_p["type"] == "ada":
            clf = AdaBoostClassifier(
                n_estimators=clf_p["n_estimators"], learning_rate=clf_p["learning_rate"]
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=clf_p["n_estimators"],
                max_depth=clf_p["max_depth"],
                learning_rate=clf_p["learning_rate"],
                subsample=clf_p["subsample"],
            )

    elif clf_p["clf"] == "svm":
        mode = "svm"
        clf = SVC(
            C=clf_p["C"], gamma=clf_p["gamma"], kernel=clf_p["kernel"], probability=True
        )

    elif clf_p["clf"] == "sgd":
        mode = "sgd"
        clf = SGDClassifier(
            loss=clf_p["loss"],
            penalty=clf_p["penalty"],
            alpha=clf_p["alpha"],
            n_iter=clf_p["n_iter"],
        )

    else:
        raise Exception("Classifier not supported")

    return clf, mode


def train(X_train, y_train, project=False, rnd=42, **kwargs):
    if project is not False:
        if project == "rproj":
            proj = SparseRandomProjection(
                n_components=X_train.shape[1], random_state=rnd
            )
        elif project == "std":
            proj = StandardScaler()
        elif project == "pca":
            proj = PCA(n_components="mle", whiten=True, random_state=rnd)
        elif project == "rbf":
            proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
        else:
            raise Error("Projection {} not available".format(project))

        X_train = proj.fit_transform(X_train)

    kwargs.setdefault("random_state", rnd)

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
        result["probs"] = clf.predict_proba(X)
    if log:
        result["log_probs"] = clf.predict_log_proba(X)
    if label:
        result["class"] = clf.predict(X)
    return result


def _sr_prediction(
    features_stack,
    annotation_volume,
    sr: SRData,
    predict_params,
    do_pca=False,
    num_components=0,
):
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
    Yr = rlabels(
        annotation_volume.astype(np.uint16), sr.supervoxel_vol.astype(np.uint32)
    )  # unsigned char and unsigned int required
    logger.debug(f"Unique labels in anno: {np.unique(annotation_volume)}")

    i_train = Yr > PRED_MIN

    X = sr.supervoxel_features
    if do_pca:
        proj = PCA(n_components="mle", whiten=True, random_state=20)
        # proj = StandardScaler()
        # proj = SparseRandomProjection(
        #    n_components=X.shape[1], random_state=random_state
        # )
        # proj = RBFSampler(
        #    n_components=max(X.shape[1], 50), random_state=random_state
        # )
        X = proj.fit_transform(X)

    X_train = X[i_train]

    # Projection

    Y_train = Yr[i_train]

    clf = train(
        X_train, Y_train, n_estimators=predict_params["n_estimators"]
    )  # , project=predict_params['proj'])

    logger.debug(f"Predicting with clf: {clf}")

    P = predict(X, clf, label=True, probs=True)  # , proj=predict_params['proj'])

    prob_map = invrmap(P["class"], sr.supervoxel_vol)
    num_supervox = sr.supervoxel_vol.max() + 1
    conf_map = invrmap(P["probs"], sr.supervoxel_vol)

    srprediction = SRPrediction(prob_map, conf_map, P)

    return srprediction


def sr_predict(
    supervoxel_image: np.ndarray,
    anno_image: np.ndarray,
    feature_images: List[np.ndarray],
    lam: float,
    do_pca: bool,
    num_components: int,
) -> np.ndarray:

    feats = features_factory(feature_images)
    logger.info(f"Number of features calculated: {len(feats.features_stack)}")

    sr = superregion_factory(supervoxel_image.astype(np.uint32), feats.features_stack)
    logger.info(f"Calculated superregions: {sr}")

    srprediction = _sr_prediction(
        feats.features_stack,
        anno_image.astype(np.uint16),
        sr,
        cfg.pipeline["predict_params"],
        do_pca=do_pca,
        num_components=num_components,
    )

    prob_map = srprediction.prob_map
    logger.info(f"Made sr prediction: {srprediction}")

    prob_map = mrf_refinement(
        srprediction.P, supervoxel_image, feats.features_stack, lam=lam
    )
    logger.info(f"Calculated MRF Refinement")

    return prob_map


def mrf_refinement(P, supervoxel_vol, features_stack, lam=0.5, gamma=False):

    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(
            np.array(supervoxel_vol).astype(np.uint32), connectivity=6
        )

        unary = (-np.ma.log(P["probs"])).filled()
        pred = P["class"]
        labels = np.asarray(
            list(set(np.unique(P["class"][supervoxel_vol])) - set([-1])), np.int32
        )
        unary = unary.astype(np.float32)

        mapping = np.zeros(pred.max() + 1, np.int32)
        mapping[labels] = np.arange(labels.size)

        idx = np.where(pred > PRED_MIN)[0]
        col = mapping[pred[idx]]
        unary[idx, col] = 1

        y_ref = refine(supervoxel_rag, unary, supervoxel_rag, lam, gamma=gamma)
        logger.debug(f"Pred max {pred.max()}")
        Rp_ref = (pred.max() + 1) - (invrmap(y_ref, supervoxel_vol) + 1)

        logger.debug(
            f"Calculated mrf refinement with lamda {lam} of shape: {Rp_ref.shape}"
        )

    except Exception as err:
        logger.error(f"MRF refinement exception: {err}")

    return Rp_ref
