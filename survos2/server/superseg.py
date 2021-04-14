"""
Core segmentation functions for SuRVoS Super-region segmentation

"""

import os
import sys
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn
from loguru import logger
from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC

from survos2.improc.segmentation._qpbo import solve_aexpansion, solve_binary
from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.mappings import rmeans
from survos2.server.features import features_factory
from survos2.server.model import SRData, SRPrediction
from survos2.server.region_labeling import rlabels
from survos2.server.state import cfg
from survos2.server.supervoxels import invrmap, superregion_factory

PRED_MIN = 0  # label value to use as minimum prediction label


def obtain_classifier(clf_p):
    if clf_p["clf"] == "Ensemble":
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


def train(X_train, y_train, predict_params, project=None, rnd=42, **kwargs):
    logger.debug(f"Using projection {project} and predict params {predict_params}")

    if project == "rproj":
        proj = SparseRandomProjection(n_components=X_train.shape[1], random_state=rnd)
        logger.debug(f"Projection is {proj}")
        X_train = proj.fit_transform(X_train)

    elif project == "std":
        proj = StandardScaler()
        logger.debug(f"Projection is {proj}")
        X_train = proj.fit_transform(X_train)

    elif project == "pca":
        proj = PCA(n_components="mle", whiten=True, random_state=rnd)
        logger.debug(f"Projection is {proj}")
        X_train = proj.fit_transform(X_train)

    elif project == "rbf":
        proj = RBFSampler(n_components=max(X_train.shape[1], 50), random_state=rnd)
        logger.debug(f"Projection is {proj}")
        X_train = proj.fit_transform(X_train)
    else:
        proj = None

    kwargs.setdefault("random_state", rnd)

    clf, mode = obtain_classifier(predict_params)

    logger.debug(
        f"Obtained classifier {clf}, fitting on {X_train.shape}, {y_train.shape}"
    )
    clf.fit(X_train, y_train)

    return clf, proj


def predict(X, clf, proj=None, label=True, probs=False, log=False):
    if proj is not None:
        logger.debug(f"Predicting with projection {proj}")
        X = proj.transform(X)
    result = {}
    if probs:
        result["probs"] = clf.predict_proba(X)
    if log:
        result["log_probs"] = clf.predict_log_proba(X)
    if label:
        result["class"] = clf.predict(X)
    return result


def train_and_classify_regions(
    features_stack: np.ndarray,
    annotation_volume: np.ndarray,
    sr: SRData,
    mask: Optional[np.ndarray],
    superseg_cfg: dict,
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

    X = sr.supervoxel_features
    if mask is not None:
        logger.debug(
            f"Masking with mask of shape {mask.shape} and unique labels  {np.unique(mask)}"
        )
        mask.shape = -1
        mask = mask.astype(np.int16)
        _mask = np.bincount(sr.supervoxel_vol.ravel(), weights=mask.ravel() * 2 - 1) > 0
        logger.debug(f"Flattened mask of shape {_mask.shape}")

        idx_train = (Yr > PRED_MIN) & _mask
    else:
        idx_train = Yr > PRED_MIN

    X_train = X[idx_train]
    Y_train = Yr[idx_train]

    logger.info(
        f"+ Predicting with params {superseg_cfg['predict_params']} on {X_train.shape}, {Y_train.shape}"
    )

    # train and predict
    clf, proj = train(
        X_train,
        Y_train,
        predict_params=superseg_cfg["predict_params"],
        n_estimators=superseg_cfg["predict_params"]["n_estimators"],
        project=superseg_cfg["predict_params"]["proj"],
    )
    logger.debug(f"Predicted  with clf: {clf} and projection {proj}")
    P = predict(X, clf, label=True, probs=True, proj=proj)

    # map superregion prediction back to voxel image
    prob_map = invrmap(P["class"], sr.supervoxel_vol)
    num_supervox = sr.supervoxel_vol.max() + 1
    conf_map = invrmap(P["probs"], sr.supervoxel_vol)
    srprediction = SRPrediction(prob_map, conf_map, P)

    return srprediction


def sr_predict(
    supervoxel_image: np.ndarray,  # Supervoxel label image
    anno_image: np.ndarray,  # Annotation label image
    feature_images: List[np.ndarray],  # List of feature volumes
    mask: Optional[np.ndarray],
    superseg_cfg: dict,
    refine: bool,
    lam: float,  # lambda parameter to MRF Refinement
) -> np.ndarray:  # Volume of predicted region labels
    """Region classification combined with MRF Refinement
    """

    feats = features_factory(feature_images)
    logger.info(f"Number of features calculated: {len(feats.features_stack)}")

    sr = superregion_factory(supervoxel_image.astype(np.uint32), feats.features_stack)
    #logger.info(f"Calculated superregions: {sr}")

    srprediction = train_and_classify_regions(
        feats.features_stack,
        anno_image.astype(np.uint16),
        sr,
        mask,
        superseg_cfg
    )

    prob_map = srprediction.prob_map
    logger.info(f"Made sr prediction: {srprediction}")

    if refine:
        prob_map = mrf_refinement(
            srprediction.P, supervoxel_image, feats.features_stack, lam=lam
        )
        logger.info(f"Calculated MRF Refinement")

    return prob_map


def mrf_refinement(P, supervoxel_vol, features_stack, lam=0.5, gamma=False):

    from survos2.improc.regions.rag import create_rag

    try:
        supervoxel_vol = np.array(supervoxel_vol).astype(np.uint32)
        supervoxel_features = rmeans(features_stack.astype(np.float32), supervoxel_vol)
        supervoxel_rag = create_rag(
            np.array(supervoxel_vol).astype(np.uint32), connectivity=26
        )

        unary = (-np.ma.log(P["probs"])).filled()
        pred = P["class"]
        labels = np.asarray(
            list(set(np.unique(P["class"][supervoxel_vol])) - set([-1])), np.int32
        )

        unary = unary.astype(np.float32)
        mapping = np.zeros(pred.max() + 1, np.int32)
        mapping[labels] = np.arange(labels.size)
        idx = np.where(pred > -1)[0]
        col = mapping[pred[idx]]
        unary[idx, col] = 0

    except Exception as err:
        logger.error(f"MRF refinement setup exception: {err}")

    try:
        y_ref = refine(
            supervoxel_features, np.nan_to_num(unary), supervoxel_rag, lam, gamma=gamma
        )
        logger.debug(f"Pred max {pred.max()}")

    except Exception as err:
        logger.error(f"MRF refinement exception: {err}")

    try:
        labels_out = (pred.max() + 1) - (invrmap(y_ref, supervoxel_vol) + 1)
        # labels_out = invrmap(y_ref, supervoxel_vol) + 1

        logger.debug(
            f"Calculated mrf refinement with lamda {lam} of shape: {labels_out.shape}"
        )
        labels_out = np.nan_to_num(labels_out)

        return labels_out

    except Exception as err:
        logger.error(f"MRF post refinement exception: {err}")
