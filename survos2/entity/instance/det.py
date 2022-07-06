import json
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
import torch

from sklearn.model_selection import train_test_split
from survos2.entity.entities import (
    calc_bounding_vols,
    make_entity_df
)
from survos2.entity.patches import SmallVolDataset
from survos2.entity.utils import get_largest_cc, pad_vol
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts_bb,
    offset_points,
    sample_bvol,
    sample_marked_patches,
    viz_bvols,
)
from survos2.entity.anno.pseudo import organize_entities
from survos2.frontend.nb_utils import (
    slice_plot,
    show_images,
)

from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
import torch
from sklearn.model_selection import train_test_split
from survos2.entity.entities import (
    calc_bounding_vols,
    make_entity_df
)

from survos2.entity.patches import SmallVolDataset
from survos2.entity.utils import get_largest_cc, pad_vol
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts_bb,
    offset_points,
    sample_bvol,
    sample_marked_patches,
    viz_bvols,
)
from survos2.entity.anno.pseudo import organize_entities
from survos2.frontend.nb_utils import (
    slice_plot,
    show_images,
)

from torch.utils.data import DataLoader

from torchvision import transforms


@dataclass
class PatchWorkflow:
    vols: List[np.ndarray]
    locs: np.ndarray
    entities: dict
    bg_mask: np.ndarray
    params: dict
    gold: np.ndarray


def init_entity_workflow(project_file, roi_name, plot_all=False, load_bg_mask=False):
    with open(project_file) as project_file:
        wparams = json.load(project_file)
    proj = wparams["proj"]

    if proj == "vf":
        original_data = h5py.File(
            os.path.join(wparams["input_dir"], wparams["vol_fname"]), "r"
        )
        ds = original_data[wparams["dataset_name"]]
        # wf1 = ds["workflow_1"]
        wf2 = ds[wparams["workflow_name"]]

        ds_export = original_data.get("data_export")
        # wf1_wrangled = ds_export["workflow1_wrangled_export"]
        vol_shape_x = wf2[0].shape[0]
        vol_shape_y = wf2[0].shape[1]
        vol_shape_z = len(wf2)
        img_volume = wf2
        print(f"Loaded image volume of shape {img_volume.shape}")

    if proj == "hunt":
        # fname = wparams['vol_fname']
        fname = wparams["vol_fname"]
        original_data = h5py.File(os.path.join(wparams["datasets_dir"], fname), "r")
        img_volume = original_data["data"][:]
        wf1 = img_volume

    print(f"Loaded image volume of shape {img_volume.shape}")

    workflow_name = wparams["workflow_name"]
    input_dir = wparams["input_dir"]
    out_dir = wparams["outdir"]
    torch_models_fullpath = wparams["torch_models_fullpath"]
    project_file = wparams["project_file"]
    entity_fpath = wparams["entity_fpath"]
    # entity_fnames = wparams["entity_fnames"]
    entity_fname = wparams["entity_fname"]
    datasets_dir = wparams["datasets_dir"]
    entities_offset = wparams["entities_offset"]
    offset = wparams["entities_offset"]
    entity_meta = wparams["entity_meta"]
    main_bv = wparams["main_bv"]
    
    gold_fname = wparams["gold_fname"]
    gold_fpath = wparams["gold_fpath"]

    #
    # load object data
    #
    entities_df = pd.read_csv(os.path.join(entity_fpath, entity_fname))
    entities_df.drop(
        entities_df.columns[entities_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    entity_pts = np.array(entities_df)

    scale_z, scale_x, scale_y = 1.0, 1.0, 1.0
    entity_pts[:, 0] = (entity_pts[:, 0] * scale_z) + offset[0]
    entity_pts[:, 1] = (entity_pts[:, 1] * scale_x) + offset[1]
    entity_pts[:, 2] = (entity_pts[:, 2] * scale_y) + offset[2]

    #
    # Crop main volume
    #
    main_bv = calc_bounding_vols(main_bv)
    bb = main_bv[roi_name]["bb"]
    print(f"Main bounding box: {bb}")
    roi_name = "_".join(map(str, bb))

    logger.debug(roi_name)

    #
    # load gold data
    #
    gold_df = pd.read_csv(os.path.join(gold_fpath, gold_fname))
    gold_df.drop(
        gold_df.columns[gold_df.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    gold_pts = np.array(gold_df)
    # gold_df = make_entity_df(gold_pts, flipxy=True)
    # gold_pts = np.array(e_df)
    scale_z, scale_x, scale_y = 1.0, 1.0, 1.0
    gold_pts[:, 0] = (gold_pts[:, 0] * scale_z) + offset[0]
    gold_pts[:, 1] = (gold_pts[:, 1] * scale_x) + offset[1]
    gold_pts[:, 2] = (gold_pts[:, 2] * scale_y) + offset[2]

    # precropped_wf2, gold_pts = crop_vol_and_pts_bb(
    #     img_volume, gold_pts, bounding_box=bb, debug_verbose=True, offset=True
    # )

    print(f"Loaded entities of shape {entities_df.shape}")

    #
    # Load bg mask
    #
    # with h5py.File(os.path.join(wparams["datasets_dir"],bg_mask_fname), "r") as hf:
    #    logger.debug(f"Loaded bg mask file with keys {hf.keys()}")

    if load_bg_mask:
        bg_mask_fname = wparams["bg_mask_fname"]
        bg_mask_fullname = os.path.join(wparams["datasets_dir"], bg_mask_fname)
        bg_mask_file = h5py.File(bg_mask_fullname, "r")
        print(bg_mask_fullname)
        bg_mask = bg_mask_file["mask"][:]
    else:
        bg_mask=np.ones_like(img_volume)

    print(f"Number of entities loaded: {entity_pts.shape} ")
    precropped_wf2, precropped_pts = crop_vol_and_pts_bb(
        img_volume, entity_pts, bounding_box=bb, debug_verbose=True, offset=True
    )
    combined_clustered_pts, classwise_entities = organize_entities(
        precropped_wf2, precropped_pts, entity_meta, plot_all=plot_all
    )
    bg_mask_crop = sample_bvol(bg_mask, bb)
    print(
        f"Cropping background mask of shape {bg_mask.shape} with bounding box: {bb} to shape of {bg_mask_crop.shape}"
    )
    # bg_mask_crop = bg_mask
    wf = PatchWorkflow(
        [precropped_wf2, precropped_wf2],
        combined_clustered_pts,
        classwise_entities,
        bg_mask_crop,
        wparams,
        gold_pts,
    )

    if plot_all:
        plt.figure(figsize=(15, 15))
        plt.imshow(wf.vols[0][0, :], cmap="gray")
        plt.title("Input volume")
        slice_plot(wf.vols[1], wf.locs, None, (40, 200, 200))

    return wf


def make_augmented_entities(aug_pts, random_bg_pts=True):
    if random_bg_pts:
        print(f"Number of randomly sampled background locations: {aug_pts.shape}")
        #class1 and class2
        # class1
        fg_entities = np.array(make_entity_df(np.array(aug_pts), flipxy=False))
        fg_entities[:,3][fg_entities[:,3]==1] = 0
        fg_entities[:,3][fg_entities[:,3]==2] = 0
        fg_entities[:,3][fg_entities[:,3]==5] = 0

        bg_entities = fg_entities[fg_entities[:,3]==6]
        bg_entities[:,3][bg_entities[:,3]==6] = 1
        print(bg_entities.shape)

        augmented_entities = np.concatenate((fg_entities[fg_entities[:,3]==0],bg_entities))
        print(augmented_entities.shape)
    else:                # class2
        fg_entities = np.array(make_entity_df(np.array(gt_entities), flipxy=False))
        #fg_entities[:,3][fg_entities[:,3]==1] = 0
        #fg_entities[:,3][fg_entities[:,3]==2] = 0
        bg_entities = gt_entities.copy()
        bg_entities[:,3][bg_entities[:,3]==6] = 1
        print(bg_entities.shape, np.unique(bg_entities[:,3]))

        bg_entities = np.concatenate((bg_entities, fg_entities[fg_entities[:,3]==0], fg_entities[fg_entities[:,3]==2], fg_entities[fg_entities[:,3]==1]))
        fg_entities[:,3][fg_entities[:,3]==5] = 0

        print(bg_entities.shape)

        #bg_entities[:,3][bg_entities[:,3]==0] = 1
        bg_entities[:,3][bg_entities[:,3]==2] = 1
        bg_entities[:,3][bg_entities[:,3]==1] = 1
        bg_entities[:,3][bg_entities[:,3]==5] = 1

        augmented_entities = np.concatenate((fg_entities[fg_entities[:,3]==0],bg_entities))
        print(augmented_entities.shape)
        
    return augmented_entities
