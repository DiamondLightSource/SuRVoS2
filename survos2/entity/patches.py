import ast
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from pprint import pprint
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridAggregator, GridSampler
from torchvision import transforms
from tqdm import tqdm


from scipy import ndimage
import torchvision.utils
from skimage import data, measure
from sklearn.model_selection import train_test_split
from survos2 import survos
from survos2.entity.entities import (
    make_bounding_vols,
    make_entity_bvol,
    make_entity_df,
    calc_bounding_vols,
)


from survos2.entity.utils import get_largest_cc, get_surface, pad_vol
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts,
    crop_vol_and_pts_bb,
    offset_points,
    sample_bvol,
    sample_marked_patches,
    viz_bvols,
)
from survos2.frontend.nb_utils import (
    slice_plot,
    show_images,
    view_vols_labels,
    view_vols_points,
    view_volume,
    view_volumes,
)



# LabeledDataset
class SmallVolDataset(Dataset):
    def __init__(
        self, images, labels, class_names=None, slice_num=None, dim=3, transform=None
    ):
        self.input_images, self.target_labels = images, labels
        self.transform = transform
        self.class_names = class_names
        self.slice_num = slice_num
        self.dim = dim

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        image = self.input_images[idx]
        label = self.target_labels[idx]

        if self.dim == 2 and self.slice_num is not None:
            image = image[self.slice_num, :]
            image = np.stack((image, image, image)).T

        if self.transform:
            image = self.transform(image.T)
            label = self.transform(label.T)
        return image, label


def get_largest_cc(I):
    """Return the largest connected component

    Args:
        I (np.ndarray): Image volume

    Returns:
        np.ndarray: Int array of the largest connected component
    """
    img = I > 0
    label_im, nb_labels = ndimage.label(img)
    sizes = ndimage.sum(I, label_im, range(nb_labels + 1))
    max_sz = np.max(sizes)
    lab_sz = sizes[label_im]
    cc = lab_sz == max_sz
    cc = cc.astype(int)

    return cc


def pad_vol(vol, padding):
    """Pad a volume with zeros.

    Args:
        vol (np.ndsarray): Image array
        padding (List): 3-element list of padding amount in each direction.

    Returns:
        np.ndarray: resultant padded array
    """
    padded_vol = np.zeros(
        (
            vol.shape[0] + padding[0] * 2,
            vol.shape[1] + padding[1] * 2,
            vol.shape[2] + padding[2] * 2,
        )
    )

    padded_vol[
        padding[0] : vol.shape[0] + padding[0],
        padding[1] : vol.shape[1] + padding[1],
        padding[2] : vol.shape[2] + padding[2],
    ] = vol

    return padded_vol

@dataclass
class PatchWorkflow:
    """Dataclass for PatchWorkflows used by the patch-based 3d fcn 
    """
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    bg_mask: np.ndarray
    params: dict
    gold: np.ndarray

def organize_entities(
    img_vol, clustered_pts, entity_meta, flipxy=False, plot_all=False
):
    class_idxs = entity_meta.keys()
    classwise_entities = []

    for c in class_idxs:
        pt_idxs = clustered_pts[:, 3] == int(c)
        classwise_pts = clustered_pts[pt_idxs]
        clustered_df = make_entity_df(classwise_pts, flipxy=flipxy)
        classwise_pts = np.array(clustered_df)
        classwise_entities.append(classwise_pts)
        entity_meta[c]["entities"] = classwise_pts
        
    combined_clustered_pts = np.concatenate(classwise_entities)

    return combined_clustered_pts, entity_meta


def load_patch_vols(train_vols):
    with h5py.File(train_vols[0], "r") as hf:
        print(hf.keys())
        img_vols = hf["data"][:]

    with h5py.File(train_vols[1], "r") as hf:
        print(hf.keys())
        label_vols = hf["data"][:]
    print(img_vols.shape, label_vols.shape)

    return img_vols, label_vols


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_patches(
    wf,
    selected_locs,
    outdir,
    vol_num=0,
    proposal_vol=None,
    padding=(64, 64, 64),
    num_augs=2,
    max_vols=-1,
):
    """Make a patch dataset by sampling an image volume at selected locations.

    Args:
        wf (PatchWorkflow): A PatchWorkflow object.
        selected_locs (np.ndarray): A numpy array of selected locations.
        outdir (string): Output directory to store dataset in.
        vol_num (int, optional): Volume number to take from PatchWorkflow. Defaults to 0.
        proposal_vol ([type], optional): [description]. Defaults to None.
        padding (tuple, optional): Bounding volume size. Defaults to (64, 64, 64).
        num_augs (int, optional): Number of data augmentations. Defaults to 2.
        max_vols (int, optional): Hard limit on the number of patches. Defaults to -1 (none).

    Returns:
        tuple: Tuple of hdf5 file name for images and labels.
    """

    padded_vol = pad_vol(wf.vols[vol_num], padding)
    padded_anno = pad_vol(proposal_vol, padding)
    if num_augs > 0:
        some_pts = np.vstack(
            [
                offset_points(selected_locs, padding, scale=32, random_offset=True)
                for i in range(num_augs)
            ]
        )
        print(f"Augmented point locations {some_pts.shape}")
    else:
        some_pts = offset_points(
            selected_locs, np.array(padding), scale=32, random_offset=False
        )

    patch_size = padding
    marked_patches_anno = sample_marked_patches(
        padded_anno, some_pts, some_pts, patch_size=patch_size
    )
    marked_patches = sample_marked_patches(
        padded_vol, some_pts, some_pts, patch_size=patch_size
    )

    img_vols = marked_patches.vols
    label_vols = marked_patches_anno.vols
    marked_patches.vols_locs.shape

    print(
        f"Marked patches, unique label vols {np.unique(label_vols)}, img mean: {np.mean(img_vols[0])}"
    )

    if num_augs > 0:

        img_vols_flipped = []
        label_vols_flipped = []

        for i, vol in enumerate(img_vols):
            img_vols_flipped.append(np.fliplr(vol))
            img_vols_flipped.append(np.flipud(vol))
        for i, vol in enumerate(label_vols):
            label_vols_flipped.append(np.fliplr(vol))
            label_vols_flipped.append(np.flipud(vol))

        img_vols = np.vstack((img_vols, np.array(img_vols_flipped)))
        label_vols = np.vstack((label_vols, np.array(label_vols_flipped)))


    if max_vols > 0:
        img_vols = img_vols[0:max_vols]
        label_vols = label_vols[0:max_vols]

    wf.params["outdir"] = outdir

    # save vols
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")

    workflow_name = wf.params["workflow_name"]

    workflow_name = "patch_vols"

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

    print(f"Saving image vols {map_fullpath}")

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
    print(f"Saving image vols {map_fullpath}")

    # # save annotation mask (input image with the annotation volume regions masked)
    # map_fullpath = os.path.join(
    #     wf.params["outdir"],
    #     str(wf.params["proj"])
    #     + "_"
    #     + str(workflow_name)
    #     + "_"
    #     + str(len(label_vols))
    #     + "_mask_gt_"
    #     + str(dt_string)
    #     + ".h5",
    # )
    # wf.params["mask_gt"] = map_fullpath
    # with h5py.File(map_fullpath, "w") as hf:
    #     hf.create_dataset("data", data=mask_gt)
    # print(f"Saving image vols {map_fullpath}")

    return wf.params["img_vols_fullpath"], wf.params["label_vols_fullpath"]



def prepare_dataloaders(
    img_vols, label_vols, model_type, batch_size=1
):
    from sklearn.model_selection import train_test_split

    raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
        img_vols, (label_vols > 0) * 1.0, test_size=0.1, random_state=42
    )
    print(
        f"Prepared train X : {raw_X_train.shape} and train y: {raw_y_train.shape}  and test X: {raw_X_test.shape} and test y {raw_y_test.shape}"
    )

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

    if model_type != "unet" or model_type != "fpn3d":
        dataloaders = {
            "train": DataLoader(
                train_dataset3d, batch_size=batch_size, shuffle=True, num_workers=0
            ),
            "val": DataLoader(
                test_dataset3d, batch_size=batch_size, shuffle=False, num_workers=0
            ),
        }

    return dataloaders
