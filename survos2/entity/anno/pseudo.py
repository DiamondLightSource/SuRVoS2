import ast
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, List
from loguru import logger
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import patches, patheffects
from napari import layers
from skimage import data, measure
from skimage import exposure
import morphsnakes as ms

from torch import optim
from torch.autograd import Variable
from torch.nn import init
from torch.optim import LBFGS, SGD, Adam, AdamW, lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm


from survos2.entity.sampler import (
    generate_random_points_in_volume,
)
from survos2.frontend.nb_utils import (
    slice_plot,
)


from survos2.server.model import SRData, SRFeatures
from survos2.entity.pipeline import Patch
from survos2.server.state import cfg

from survos2.entity.entities import make_entity_df


def organize_entities(img_vol, clustered_pts, entity_meta, flipxy=False, plot_all=False):
    """Sorts clustered points by class and stores in entity_meta for further processing by generate_annotation

    Parameters
    ----------
    img_vol : np.ndarray
        Image volume (just for visualization in notebook)
    clustered_pts : np.ndarray
        Entity array (z,x,y,class)
    entity_meta : dict
        Dictionary by class of each entity, with additional information about the size of the object to render at the centroid
    flipxy : bool, optional
        Flip x y coordinates, by default False
    plot_all : bool, optional
        Plotting in notebooks, by default False

    Returns
    -------
    Tuple
        Tuple of (points, dict)
    """
    class_idxs = entity_meta.keys()
    classwise_entities = []

    for c in class_idxs:
        pt_idxs = clustered_pts[:, 3] == int(c)
        classwise_pts = clustered_pts[pt_idxs]
        clustered_df = make_entity_df(classwise_pts, flipxy=flipxy)
        classwise_pts = np.array(clustered_df)
        classwise_entities.append(classwise_pts)
        entity_meta[c]["entities"] = classwise_pts
        if plot_all:
            plt.figure(figsize=(9, 9))
            plt.imshow(img_vol[img_vol.shape[0] // 4, :], cmap="gray")
            plt.scatter(classwise_pts[:, 1], classwise_pts[:, 2], c="cyan")
            plt.title(
                str(entity_meta[c]["name"]) + " Clustered Locations: " + str(len(classwise_pts))
            )

    combined_clustered_pts = np.concatenate(classwise_entities)

    return combined_clustered_pts, entity_meta


def make_acwe(patch: Patch, params: dict):
    """
    Active Contour
    Setup and run ACWE

    """
    edge_map = 1.0 - patch.image_layers["Main"]
    edge_map -= np.min(edge_map)
    edge_map = edge_map / np.max(edge_map)
    logger.debug("Calculating ACWE")

    seg1 = ms.morphological_geodesic_active_contour(
        edge_map,
        iterations=params["iterations"],
        init_level_set=patch.image_layers["total_mask"],
        smoothing=params["smoothing"],
        threshold=params["threshold"],
        balloon=params["balloon"],
    )
    outer_mask = ((seg1 * 1.0) > 0) * 2.0
    patch.image_layers["acwe"] = outer_mask

    return patch


def generate_augmented_entities(
    wf,
    generate_random_bg_entities=False,
    num_before_masking=60,
    stratified_selection=False,
    class_proportion={0: 1, 1: 1.0, 2: 1.0, 5: 1},
):
    entities = wf.locs

    if stratified_selection:
        stratified_entities = []
        for c in np.unique(entities[:, 3]):
            single_class = entities[entities[:, 3] == c]
            entities_sel = np.random.choice(
                range(len(single_class)), int(class_proportion[c] * len(single_class))
            )
            stratified_entities.append(single_class[entities_sel])
        gt_entities = np.concatenate(stratified_entities)
        print(f"Produced {len(gt_entities)} entities.")
    else:
        gt_entities = entities

    if generate_random_bg_entities:
        random_entities = generate_random_points_in_volume(wf.vols[0], num_before_masking).astype(
            np.uint32
        )
        from survos2.entity.utils import remove_masked_entities

        print(f"Before masking random entities generated of shape {random_entities.shape}")
        random_entities = remove_masked_entities(wf.bg_mask, random_entities)

        print(f"After masking: {random_entities.shape}")
        random_entities[:, 3] = np.array([6] * len(random_entities))
        # augmented_entities = np.vstack((gt_entities, masked_entities))
        # print(f"Produced augmented entities array of shape {augmented_entities.shape}")
    else:
        random_entities = []

    return gt_entities, random_entities


def generate_annotation_volume(
    wf,
    entity_meta,
    gt_proportion=1.0,
    padding=(64, 64, 64),
    generate_random_bg_entities=False,
    num_before_masking=60,
    acwe=False,
    stratified_selection=False,
    class_proportion={0: 1, 1: 1.0, 2: 1.0, 5: 1},
):
    """Generate a volume of annotation from an entity_meta dict

    Parameters
    ----------
    wf : PatchWorkflow
        PatchWorkflow objet
    entity_meta : dict
        Dictionary of entities by class, storing additional info about size of generated objects.
    gt_proportion : float, optional
        Proportion of given entities to select, by default 1.0
    padding : tuple, optional
        Max size of generated blobs, by default (64, 64, 64)
    generate_random_bg_entities : bool, optional
        Add additional random points, by default False
    num_before_masking : int, optional
        Additional random points can be masked. This is the total number of points to add before masking, by default 60
    acwe : bool, optional
        Active contour with edges, by default False
    stratified_selection : bool, optional
        Use the class proportion dict to select different proportions by class, by default False
    class_proportion : dict, optional
        Dictionary giving proportions to select by class, by default {0: 1, 1: 1.0, 2: 1.0, 5: 1}

    Returns
    -------
    [type]
        [description]
    """
    entities = wf.locs
    # entities_sel = np.random.choice(
    #    range(len(entities)), int(gt_proportion * len(entities))
    # )
    # gt_entities = entities[entities_sel]

    if stratified_selection:
        stratified_entities = []
        for c in np.unique(entities[:, 3]):
            single_class = entities[entities[:, 3] == c]
            entities_sel = np.random.choice(
                range(len(single_class)), int(class_proportion[c] * len(single_class))
            )
            stratified_entities.append(single_class[entities_sel])
        gt_entities = np.concatenate(stratified_entities)
        print(f"Produced {len(gt_entities)} entities.")
    else:
        gt_entities = entities

    if generate_random_bg_entities:
        random_entities = generate_random_points_in_volume(wf.vols[0], num_before_masking).astype(
            np.uint32
        )
        from survos2.entity.utils import remove_masked_entities

        print(f"Before masking random entities generated of shape {random_entities.shape}")
        # random_entities = remove_masked_entities(wf.bg_mask, random_entities)

        # print(f"After masking: {random_entities.shape}")
        random_entities[:, 3] = np.array([6] * len(random_entities))
        # augmented_entities = np.vstack((gt_entities, masked_entities))
        # print(f"Produced augmented entities array of shape {augmented_entities.shape}")
    else:
        random_entities = []

    anno_masks, anno_all, gt_entities = make_anno(
        wf, gt_entities, entity_meta, gt_proportion, padding, acwe=acwe
    )

    return anno_masks, anno_all, gt_entities, random_entities


def make_anno(wf, entities, entity_meta, gt_proportion, padding, acwe=False, plot_all=True):
    """Helper function to produce annotation with a single command

    Parameters
    ----------
    wf : PatchWorkflow
        A PatchWorkflow object gathers all the info required for a segmentation workflow.
    entities : np.ndarray
        Array of entitites (z,x,y,class)
    entity_meta : dict
        Dictionary storing information about each entity, by class. Information such as the size of the object to render
        at the centroid location is stored.
    gt_proportion : float
        Proportion of provided entities to select randomly from the entitties array.
    padding : tuple
        Maximum size of the objects being generated (to cover literal edge cases)
    acwe : bool, optional
        Active Contour With Edges, using the generated mask to initialize the location of the contour, by default False
    plot_all : bool, optional
        Descriptive plotting, by default True

    Returns
    -------
    Tuple
        Generated masks a dictionary by class of each mask, All generated masks including acwe, array of entities selected
    """
    combined_clustered_pts, classwise_entities = organize_entities(
        wf.vols[0], entities, entity_meta, plot_all=plot_all
    )
    wf.params["entity_meta"] = entity_meta
    anno_masks, anno_all = make_pseudomasks(
        wf,
        classwise_entities,
        acwe=acwe,
        padding=padding,
        core_mask_radius=(12, 12, 12),
    )

    
    return anno_masks, anno_all, entities


def make_pseudomasks(
    wf,
    classwise_entities,
    padding=(64, 64, 64),
    core_mask_radius=(8, 8, 8),
    acwe=False,
    balloon=1.3,
    threshold=0.1,
    iterations=1,
    smoothing=1,
    plot_all=False,
):
    """Function to generate the mask annotation

    Parameters
    ----------
    wf : np.ndarray
        PatchWorkflow object
    classwise_entities : np.ndarray
        Sorted array of entities
    padding : tuple, optional
        Max size of generated blob, by default (64, 64, 64)
    core_mask_radius : tuple, optional
        Default mask radius of core for shell generation, by default (8, 8, 8)
    acwe : bool, optional
        Active Contour Without Edges, by default False
    balloon : float, optional
        Balloon parameter for ACWE, by default 1.3
    threshold : float, optional
        Threshold value for calculating ACWE contour, by default 0.1
    iterations : int, optional
        Number of iterations of ACWE, by default 1
    smoothing : int, optional
        Smoothing applied to ACWE contour, by default 1
    plot_all : bool, optional
        Descriptive plots, by default False

    Returns
    -------
    Tuple of (Dict, List)
        (Dict storing all the generated annotation, List of same)
    """

    from survos2.entity.anno.masks import generate_anno

    anno_masks, padded_vol = generate_anno(
        wf.vols[0],
        classwise_entities,
        cfg,
        padding=padding,
        remove_padding=True,
        core_mask_radius=core_mask_radius,
    )

    anno_gen = np.sum([anno_masks[i]["mask"] for i in anno_masks.keys()], axis=0)
    anno_shell_gen = np.sum([anno_masks[i]["shell_mask"] for i in anno_masks.keys()], axis=0)

    
    anno_all = [anno_masks[i]["mask"] for i in classwise_entities.keys()]
    anno_all.extend(anno_shell_gen)

    # if plot_all:
    #    slice_plot(anno_all, None, wf.vols[0], (89, 200, 200))

    anno_acwe = {}
    if acwe:
        for i in classwise_entities.keys():
            p = Patch(
                {"Main": wf.vols[0]},
                {},
                {"Points": classwise_entities["0"]["entities"]},
                {},
            )
            p.image_layers["total_mask"] = (anno_masks[i]["mask"] > 0) * 1.0
            params = cfg["pipeline"]
            params["smoothing"] = smoothing
            params["threshold"] = threshold
            params["iterations"] = iterations
            params["balloon"] = balloon

            p = make_acwe(p, params)

            anno_acwe[i] = p.image_layers["acwe"]

            anno_all = anno_acwe


    return anno_masks, anno_all
