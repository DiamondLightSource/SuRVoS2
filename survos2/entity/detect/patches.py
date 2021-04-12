import os
from matplotlib import patches, patheffects
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from pprint import pprint
import h5py
import ast
import json
import time
import pandas as pd
import sys
import math

from survos2 import survos
from napari import layers

survos.init_api()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
from datetime import datetime
from skimage import data, measure


import torch

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from tqdm import tqdm


from survos2.entity.sampler import sample_bvol
from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.frontend.nb_utils import plot_slice_and_pts


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


# from survos2.entity.detect.detector import plot_bb_2d, roi_pool_2d
# from survos2.entity.detect.detector import measure_regions, filter_small_components
from survos2.entity.detect.dataset import BoundingVolumeDataset, SmallVolDataset

from survos2.server.pipeline_ops import (
    make_features,
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
from survos2.server.state import cfg

from survos2.server.filtering import (
    gaussian_blur_kornia,
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
from survos2.entity.sampler import sample_bvol, sample_marked_patches
from survos2.entity.sampler import sample_bvol
from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.entity.detect.utils import get_surface, get_largest_cc
from survos2.frontend.nb_utils import plot_slice_and_pts
from survos2.entity.sampler import sample_bvol, sample_marked_patches, offset_points
from survos2.entity.detect.utils import pad_vol
from survos2.server.pipeline_ops import make_features, make_sr, predict_sr
from survos2.entity.detect.dataset import SmallVolDataset


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_patches(
    wf,
    outdir,
    vol_num=0,
    proposal_vol=None,
    use_gold_anno=False,
    use_proposal_file=False,
    filter_proposal=False,
    get_biggest_cc=False,
):
    # make bg mask
    from torchio import IMAGE, LOCATION
    from torchio.data.inference import GridSampler, GridAggregator
    import torchvision.utils
    from torchvision import transforms

    target_cents = np.array(wf.locs)[:, 0:4]
    target_cents = target_cents[:, [0, 2, 1, 3]]
    targs_all = centroid_to_bvol(
        target_cents, bvol_dim=wf.params["entity_meta"]["0"]["size"], flipxy=True
    )
    mask_all = viz_bvols(wf.vols[0], targs_all)

    plot_slice_and_pts(
        mask_all,
        wf.locs,
        wf.vols[0],
        (90, wf.vols[0].shape[1] // 2, wf.vols[0].shape[2] // 2),
    )

    # load anno

    if use_gold_anno:
        print("Using gold anno")
        map_fullpath = (
            "/dls/science/users/xsy37748/data/vf_detect/combined_anno_gen_ed2_1910.h5"
        )

        with h5py.File(map_fullpath, "r") as hf:
            print(hf.keys())
            gold_anno = hf["map"][:]

        proposal = gold_anno  # [:,0:1344,0:1344]

    elif use_proposal_file:
        map_fullpath = (
            "/dls/science/users/xsy37748/data/out/fpn_proposal_mask_1_3010.h5"
        )
        map_fullpath = (
            "/dls/science/users/xsy37748/data/out/ru_proposal_mask_G2_2110.h5"
        )

        with h5py.File(map_fullpath, "r") as hf:
            print(hf.keys())
            proposal = hf["map"][:]

    else:
        proposal = proposal_vol

    print(
        f"Loaded annotation of shape {proposal.shape} for vol of shape {wf.vols[0].shape}"
    )

    if filter_proposal:
        cfg["feature_params"] = {}
        cfg["feature_params"]["out1"] = [[gaussian_blur, {"sigma": 1}]]
        # cfg["feature_params"]["out2"] = [[erode, {'thresh' : 0.1, 'num_iter': 2} ]]
        cfg["feature_params"]["out2"] = [[dilate, {"thresh": 0.1, "num_iter": 1}]]
        # cfg["feature_params"]["out2"] = [[median, {'thresh' : 0.1, 'num_iter': 1, 'median_size':4} ]]

        p = Patch({"Main": proposal}, {}, {}, {})
        p = make_features(p, cfg)
        proposal_filt = p.image_layers["out2"]

    else:
        proposal_filt = proposal

    plot_slice_and_pts(proposal_filt, None, None, (40, 400, 400))

    # Prepare patch dataset
    selected_locs = wf.locs[wf.locs[:, 3] == 0]
    # selected_locs = wf.locs
    mask_vol_size = wf.params["entity_meta"]["0"]["size"]
    mask_vol_size = (26, 26, 26)
    target_cents = np.array(selected_locs)[:, 0:4]
    target_cents = target_cents[:, [0, 2, 1, 3]]
    targs_all_1 = centroid_to_bvol(target_cents, bvol_dim=mask_vol_size, flipxy=True)
    mask_gt = viz_bvols(wf.vols[0], targs_all_1)

    plot_slice_and_pts(
        mask_gt,
        wf.locs,
        wf.vols[0],
        (90, wf.vols[0].shape[1] // 2, wf.vols[0].shape[2] // 2),
    )

    padding = (64, 64, 64)
    padded_vol = pad_vol(wf.vols[vol_num], padding)
    padded_anno = pad_vol((proposal_filt > 0.4) * 1.0, padding)
    some_pts = offset_points(selected_locs, padding, scale=32, random_offset=True)
    plot_slice_and_pts(padded_vol, None, padded_anno, (130, 200, 200))

    patch_size = (64, 64, 64)
    print(some_pts.shape)
    marked_patches_anno = sample_marked_patches(
        padded_anno, some_pts, some_pts, patch_size=patch_size
    )
    marked_patches = sample_marked_patches(
        padded_vol, some_pts, some_pts, patch_size=patch_size
    )

    img_vols = marked_patches.vols
    bvols = marked_patches.vols_bbs
    labels = marked_patches.vols_locs[
        :, 3
    ]  # np.array([0] * marked_patches.vols.shape[0])
    label_vols = marked_patches_anno.vols
    label_bvols = marked_patches_anno.vols_bbs
    label_labels = marked_patches_anno.vols_locs[
        :, 3
    ]  # np.array([0] * marked_patches.vols.shape[0])
    marked_patches.vols_locs.shape

    print(np.unique(label_vols), np.mean(img_vols[0]))

    if get_biggest_cc:
        label_vols_f = []
        for i, lvol in enumerate(label_vols):
            label_vols_f.append(get_largest_cc(lvol))
        label_vols_f = np.array(label_vols_f)

    raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
        img_vols, label_vols, test_size=0.2, random_state=42
    )

    print(raw_X_train.shape, raw_X_test.shape, raw_y_train.shape, raw_y_test.shape)

    smallvol_mask_trans = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset3d = SmallVolDataset(
        raw_X_train, raw_y_train, slice_num=None, dim=3, transform=smallvol_mask_trans
    )
    train_dataset3d.class_names = np.unique(raw_y_train).astype(np.uint16)

    test_dataset3d = SmallVolDataset(
        raw_X_test, raw_y_test, slice_num=None, dim=3, transform=smallvol_mask_trans
    )
    test_dataset3d.class_names = np.unique(raw_y_test).astype(np.uint16)

    train_loader3d = torch.utils.data.DataLoader(
        train_dataset3d, batch_size=1, shuffle=True, num_workers=0, drop_last=False
    )

    test_loader3d = torch.utils.data.DataLoader(
        test_dataset3d, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )

    # view_volumes([precropped_wf1])

    for i in range(5):
        img, lbl = next(iter(train_loader3d))
        img = img.squeeze(0).numpy()
        lbl = lbl.squeeze(0).numpy()
        # sample[0].shape
        print(img.shape, lbl.shape)
        from survos2.frontend.nb_utils import show_images

        show_images([img[32, :], lbl[32, :]])
        # plt.title(lbl)
        print(f"Unique mask values: {np.unique(lbl)}")

    print(img_vols.shape, label_vols.shape)

    # wf.params["selected_locs"] = selected_locs

    wf.params["outdir"] = outdir

    # save vols

    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    output_dataset = True

    workflow_name = wf.params["workflow_name"]
    workflow_name = "quartergt"

    if output_dataset:
        map_fullpath = os.path.join(
            wf.params["outdir"],
            str(wf.params["proj"])
            + "_"
            + str(workflow_name)
            + str(len(img_vols))
            + "_img_vols_"
            + str(dt_string)
            + ".h5",
        )
        wf.params["img_vols_fullpath"] = map_fullpath
        with h5py.File(map_fullpath, "w") as hf:
            hf.create_dataset("data", data=img_vols)

        map_fullpath = os.path.join(
            wf.params["outdir"],
            str(wf.params["proj"])
            + "_"
            + str(workflow_name)
            + str(len(label_vols))
            + "_img_labels_"
            + str(dt_string)
            + ".h5",
        )
        wf.params["label_vols_fullpath"] = map_fullpath
        with h5py.File(map_fullpath, "w") as hf:
            hf.create_dataset("data", data=label_vols)

        map_fullpath = os.path.join(
            wf.params["outdir"],
            str(wf.params["proj"])
            + "_"
            + str(workflow_name)
            + "_"
            + str(len(label_vols))
            + "_mask_gt_"
            + str(dt_string)
            + ".h5",
        )
        wf.params["mask_gt"] = map_fullpath
        with h5py.File(map_fullpath, "w") as hf:
            hf.create_dataset("data", data=mask_gt)
    fullfname = wf.params["proj"] + dt_string + ".json"
    # with open(fullfname, "w") as outfile:
    #    json.dump(wf.params, outfile, indent=4, sort_keys=True, cls=NumpyEncoder)

    return wf.params["img_vols_fullpath"], wf.params["label_vols_fullpath"]
