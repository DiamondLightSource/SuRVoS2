import numpy as np
from typing import List
from dataclasses import dataclass


"""
Classes that wrap essential inputs to different parts of the survos pipeline.

Allow caching of intermediate pipeline state.

"""


@dataclass
class Features:
    """
    TODO: unnec. heavy, duplication of the same info (flattening the features or reshaping should be light enough to do when needed)
    
    """
    filtered_layers : List[np.ndarray]
    dataset_feats : np.ndarray
    features_stack : np.ndarray


@dataclass 
class Superregions:
    """Dataclass bundling all the supervoxel information required for prediction.

    (supervoxel vol, supervoxel features, region adjacency graph) 
    
    All of this information is usually calculated once, and then used repeatedly.
    """
    supervoxel_vol : np.ndarray
    supervoxel_features: List[np.ndarray]
    supervoxel_rag: np.ndarray



@dataclass
class SRPrediction:
    prob_map : np.ndarray
    conf_map : np.ndarray
    probs : np.ndarray


@dataclass
class SegData:
    img_vol_proc : List[np.ndarray] 
    feats : Features
    dataset_anno_proc : np.ndarray 
    superregions : np.ndarray

