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
from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import SVC
from survos2.improc.features._spencoding import _sp_labels
from survos2.improc.features.spencoding import sphist, spmeans, spstats
from survos2.improc.features.spgraph import aggregate_neighbors
from survos2.improc.regions.rag import create_rag
from survos2.improc.segmentation._qpbo import solve_aexpansion, solve_binary
from survos2.improc.segmentation.appearance import refine
from survos2.improc.segmentation.mappings import rmeans
from survos2.server.features import features_factory
from survos2.server.model import SRData, SRPrediction
from survos2.server.region_labeling import rlabels
from survos2.server.state import cfg
from survos2.server.supervoxels import invrmap, superregion_factory

PRED_MIN = 0  # label valus to use as minimum prediction label



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



def compute_supervoxel_descriptor(supervoxels, descriptors, desc_type, desc_bins):
    'Mean' 'Quantized' 'Textons' 'Covar' 'Sigma Set'

    if desc_type == 'Mean':
        return spmeans(descriptors, supervoxels)
    elif desc_type == 'Covar':
        return spstats(descriptors, supervoxels, mode='add', norm=None)
    elif desc_type == 'Sigma Set':
        return spstats(descriptors, supervoxels, mode='add', sigmaset=True, norm=None)

    if desc_type == 'Textons':
        logger.info('+ Applying PCA')
        descriptors = IncrementalPCA(batch_size=100).fit_transform(descriptors)

    logger.info('+ Quantizing descriptors')
    cluster = MiniBatchKMeans(n_clusters=desc_bins, batch_size=100).fit_predict(descriptors)
    return sphist(cluster.astype(np.int32), supervoxels, nbins=desc_bins)


def extract_descriptors(supervoxels=None, features=None,
                        projection=None, desc_type=None, desc_bins=None,
                        nh_order=None, sp_edges=None):
    total = len(features)

    logger.info('+ Reserving memory for {} features'.format(total))

    #descriptors = np.zeros(DM.region_shape() + (total,), np.float32)

    #for i in range(total):
    #    logger.info('    * Loading feature {}'.format(features[i]))
    #    descriptors[..., i] = DM.load_slices(features[i])

    sp = None
    mask = None

    if supervoxels is not None:
        logger.info('+ Loading supervoxels')
        
        #sp = DM.load_slices(supervoxels)
        
        if sp.min() < 0:
            raise Exception('Supervoxels need to be recomputed for this ROI')
        descriptors.shape = (-1, total)


        src = DataModel.g.dataset_uri(region_id, group="regions")
        with DatasetManager(src, out=None, dtype="float32") as DM:
            dst_dataset = DM.sources[0]
            num_sv = dst_dataset.get_attr("num_supervoxels")
            #num_sv = DM.attr(supervoxels, 'num_supervoxels')
                
        mask = np.zeros(num_sv, np.bool)
        mask[sp.ravel()] = True

        logger.info('+ Computing descriptors: {} ({})'.format(desc_type, desc_bins))
        descriptors = compute_supervoxel_descriptor(sp, descriptors,
                                                    desc_type, desc_bins)
        nh_order = int(nh_order)
        if nh_order > 0:

            logger.info('+ Loading edges into memory')
            #edges = DM.load_ds(sp_edges)

            logger.info('+ Filtering edges for selected ROI')
            idx = mask[edges[:, 0]] & mask[edges[:, 1]]
            edges = edges[idx]

            logger.info('+ Aggregating neighbour features')
            G = nx.Graph()
            G.add_edges_from(edges)
            descriptors = aggregate_neighbors(descriptors, G, mode='append',
                                              norm='mean', order=nh_order)
        descriptors = descriptors[mask]

    return descriptors, sp, mask


def train(X_train, y_train, project=None, rnd=42, **kwargs):
    logger.debug(f"Using projection {project}")
    if project == "rproj":
        proj = SparseRandomProjection(
            n_components=X_train.shape[1], random_state=rnd
        )
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

    clf, mode = obtain_classifier(cfg.pipeline['predict_params'])

    logger.debug(f"Obtained classifier {clf}, fitting on {X_train.shape}, {y_train.shape}")
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


def classify_regions(
    features_stack,
    annotation_volume,
    sr: SRData,
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
    X_train = X[i_train]
    Y_train = Yr[i_train]

    logger.debug(f"Predicting with params {cfg.pipeline['predict_params']} on {X_train.shape}, {Y_train.shape}")
    clf, proj = train(X_train, 
        Y_train, 
        n_estimators=cfg.pipeline['predict_params']["n_estimators"], 
        project=cfg.pipeline["predict_params"]['proj'])
    logger.debug(f"Predicted  with clf: {clf} and projection {proj}")


    P = predict(X, clf, label=True, probs=True, proj=proj)
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
    num_components: int,
) -> np.ndarray:
    """Region classification combined with MRF Refinement

    Parameters
    ----------
    supervoxel_image : np.ndarray
        Supervoxel label image 
    anno_image : np.ndarray
        Annotation label image
    feature_images : List[np.ndarray]
        List of feature volumes
    lam : float
        Lambda parameter to MRF Refinement
    
    Returns
    -------
    np.ndarray
        Volume of predicted region labels
    """

    feats = features_factory(feature_images)
    logger.info(f"Number of features calculated: {len(feats.features_stack)}")
    sr = superregion_factory(supervoxel_image.astype(np.uint32), feats.features_stack)
    logger.info(f"Calculated superregions: {sr}")
    logger.debug(f"Making prediction with {cfg.pipeline['predict_params']}")
    
    srprediction = classify_regions(
        feats.features_stack,
        anno_image.astype(np.uint16),
        sr,
        num_components=num_components
    )

    prob_map = srprediction.prob_map
    logger.info(f"Made sr prediction: {srprediction}")

    prob_map = mrf_refinement(
        srprediction.P, 
        supervoxel_image, 
        feats.features_stack, lam=lam
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
        # Rp_ref = (pred.max() + 1) - (invrmap(y_ref, supervoxel_vol) + 1)
        labels_out = invrmap(y_ref, supervoxel_vol) + 1
        logger.debug(
            f"Calculated mrf refinement with lamda {lam} of shape: {labels_out.shape}"
        )

    except Exception as err:
        logger.error(f"MRF refinement exception: {err}")

    return labels_out
