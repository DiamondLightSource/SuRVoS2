import numpy as np
from typing import List, Optional
from dataclasses import dataclass


"""
Classes that wrap essential inputs to different parts of the survos pipeline.

Allow caching of intermediate pipeline state.

"""


@dataclass
class SRFeatures:
    # All the filter information needed for superregion and prediction
    filtered_layers: List[np.ndarray]
    dataset_feats: np.ndarray
    features_stack: np.ndarray


@dataclass
class SRData:
    """Dataclass bundling the supervoxel information required for prediction.
    (supervoxel vol, supervoxel features, region adjacency graph)
    All of this information is usually calculated once, and then used repeatedly.
    """

    supervoxel_vol: np.ndarray
    supervoxel_features: List[np.ndarray]
    supervoxel_rag: np.ndarray


@dataclass
class SRPrediction:
    """Survos segmentation prediction result"""

    prob_map: np.ndarray
    conf_map: np.ndarray
    P: dict


@dataclass
class SRSegmentation:
    """Required for a survos segmentation prediction"""

    features: Optional[SRFeatures]
    superregions: Optional[SRData]
    srprediction: Optional[SRPrediction]


# old
# @dataclass
# class SegData:
#    img_vol_proc : List[np.ndarray]
#    feats : SRFeatures
#    dataset_anno_proc : np.ndarray
#    superregions : np.ndarray
