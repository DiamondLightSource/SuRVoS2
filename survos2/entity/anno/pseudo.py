from survos2 import survos
from napari import layers

survos.init_api()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from skimage import data, measure

import torch
from torch.optim import SGD, Adam, LBFGS, AdamW
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch import optim
import torchvision.utils
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator
import os
import h5py
import ast
import json
import time
import pandas as pd
import sys
import math
import os
from matplotlib import patches, patheffects
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from pprint import pprint


from survos2.frontend.nb_utils import (
    view_vols_points,
    view_vols_labels,
    view_volumes,
    view_volume,
)

from survos2.server.supervoxels import generate_supervoxels
from survos2.server.features import generate_features, prepare_prediction_features
from survos2.server.model import SRData, SRFeatures
from survos2.entity.sampler import crop_vol_and_pts
from survos2.entity.entities import make_entity_df
from survos2.entity.sampler import sample_marked_patches

# from survos2.server.pipeline_old  import predict_and_agg
from survos2.entity.detect.trainer import (
    loss_dice,
    loss_calc,
    log_metrics,
    train_model_cbs,
    TrainerCallback,
)


from survos2.entity.detect.dataset import BoundingVolumeDataset, SmallVolDataset

from survos2.server.pipeline_ops import (
    make_masks,
    make_noop,
    make_bb,
    acwe,
    make_sr,
    clean_segmentation,
    predict_and_agg,
    saliency_pipeline,
)
from survos2.entity.detect.proposal_agg import predict_and_agg
from survos2.server.pipeline import Pipeline, Patch
from survos2.server.config import cfg

from survos2.server.filtering import (
    gaussian_blur,
    spatial_gradient_3d,
    ndimage_laplacian,
)
from survos2.improc.features.tv import tvdenoising3d
from survos2.server.filtering.morph import erode, dilate, median
from survos2.entity.sampler import crop_vol_and_pts_bb
from survos2.entity.entities import make_entity_df
from survos2.frontend.nb_utils import plot_slice_and_pts
from survos2.entity.entities import (
    make_entity_bvol,
    make_bounding_vols,
    init_entity_workflow,
)
from survos2.entity.anno.masks import generate_anno

import h5py


def make_pseudomasks(wf, acwe=False):

    padding = (64, 64, 64)
    anno_masks, padded_vol = generate_anno(
        wf.vols[0],
        wf.entities,
        cfg,
        padding=padding,
        remove_padding=True,
        core_mask_radius=(8, 8, 8),
    )

    anno_gen = np.sum([anno_masks[i]["mask"] for i in anno_masks.keys()], axis=0)
    anno_shell_gen = np.sum(
        [anno_masks[i]["shell_mask"] for i in anno_masks.keys()], axis=0
    )

    anno_all = (
        anno_shell_gen
        + anno_masks["5"]["mask"]
        + anno_masks["2"]["mask"]
        + anno_masks["1"]["mask"]
    )

    anno_all = (anno_all > 0) * 1.0

    plot_slice_and_pts(anno_all, None, wf.vols[0], (89, 200, 200))

    if acwe:
        p = Patch(
            {"Main": wf.vols[1]}, {}, {"Points": wf.entities["0"]["entities"]}, {}
        )
        p.image_layers["total_mask"] = (anno_shell_gen > 0) * 1.0
        p = acwe(p, cfg["pipeline"])

    return anno_masks, anno_all
