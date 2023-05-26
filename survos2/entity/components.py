import os
from typing import Collection, List

import numpy as np
import pandas as pd
import skimage
import torch
import torch.utils.data as data
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle

from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label
from skimage import data, measure
from survos2.frontend.nb_utils import show_images
from survos2.entity.entities import make_entity_bvol, centroid_to_bvol, make_entity_df


def component_bounding_boxes(images):
    bbs_tables = []
    bbs_arrs = []

    for image in images:
        bbs_arr = measure_components(image)
        bbs_arrs.append(bbs_arr)
        bbs_table = make_entity_bvol(bbs_arr)
        bbs_tables.append(bbs_table)

    return bbs_tables, bbs_arrs


def measure_components(image):
    labeled_array, num_features = label(image.astype(np.uint))
    print(f"Measured {num_features} features")
    objs = ndimage.measurements.find_objects(labeled_array)

    bbs = []

    for i, obj in enumerate(objs):
        z_dim = obj[0].stop - obj[0].start
        x_dim = obj[1].stop - obj[1].start
        y_dim = obj[2].stop - obj[2].start
        z = obj[0].start + (z_dim / 2.0)
        x = obj[1].start + (x_dim / 2.0)
        y = obj[2].start + (y_dim / 2.0)

        area = z_dim * x_dim * y_dim
        bbs.append(
            (
                i,
                area,
                z,
                y,
                x,
                obj[0].start,
                obj[1].start,
                obj[2].start,
                obj[0].stop,
                obj[1].stop,
                obj[2].stop,
            )
        )

    bbs_arr = np.array(bbs).astype(np.uint)
    return bbs_arr


def filter_proposal_mask(mask, thresh=0.5, num_erosions=3, num_dilations=3, num_medians=1):
    """
    Apply morphology and medians to input mask image, which is thresholded

    Parameters
    ----------

    mask : np.ndarray
    Floating point image

    thresh : float
    Threshold value to use to create binary mask
    """
    holdout = (mask >= thresh) * 1.0
    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_erosions):
        holdout = ndimage.binary_erosion(holdout, structure=struct2).astype(holdout.dtype)

    for i in range(num_dilations):
        holdout = ndimage.binary_dilation(holdout, structure=struct2).astype(holdout.dtype)

    for i in range(num_medians):
        holdout = ndimage.median_filter(holdout, 4).astype(holdout.dtype)

    return holdout


def measure_regions(labeled_images, properties=["label", "area", "centroid", "bbox"]):
    tables = [
        skimage.measure.regionprops_table(image, properties=properties) for image in labeled_images
    ]

    tables = [pd.DataFrame(table) for table in tables]

    tables = [
        table.rename(
            columns={
                "label": "class_code",
                "centroid-0": "z",
                "centroid-1": "x",
                "centroid-2": "y",
                "bbox-0": "bb_s_z",
                "bbox-1": "bb_s_x",
                "bbox-2": "bb_s_y",
                "bbox-3": "bb_f_z",
                "bbox-4": "bb_f_x",
                "bbox-5": "bb_f_y",
            }
        )
        for table in tables
    ]
    return tables


import numba


@numba.jit(nopython=True)
def copy_and_composite_components(images, labeled_images, tables_arr, selected_idxs):
    for img_idx in range(len(images)):
        table_idx = selected_idxs[img_idx]
        total_mask = np.zeros_like(images[img_idx])

        for iloc in table_idx:
            bb = [
                tables_arr[img_idx][4][iloc],
                tables_arr[img_idx][5][iloc],
                tables_arr[img_idx][6][iloc],
                tables_arr[img_idx][7][iloc],
                tables_arr[img_idx][8][iloc],
                tables_arr[img_idx][9][iloc],
            ]
            mask = (labeled_images[img_idx] == tables_arr[img_idx][0][iloc]) * 1.0
            total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] = (
                total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
                + mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
            )

    return total_mask


def filter_small_components_numba(images, min_component_size=0):
    """
    Filter components smaller than min_component_size

    Parameters
    ----------
    images : List[np.ndarray]

    min_component_size : int


    """
    labeled_images = [measure.label(image) for image in images]
    tables = measure_regions(labeled_images)

    selected = [tables[i][tables[i]["area"] > min_component_size] for i in range(len(tables))]

    filtered_images = []
    tables_arr = np.array(tables)

    selected_idxs = []
    for img_idx in range(len(images)):
        table_idxs = list(selected[img_idx].index.values)
        selected_idxs.append(table_idxs)

    selected_idxs = np.array(selected_idxs)
    total_mask = copy_and_composite_components(images, labeled_images, tables_arr, selected_idxs)

    return total_mask, tables, labeled_images


def filter_small_components(images, min_component_size=0):
    """
    Filter components smaller than min_component_size

    Parameters
    ----------
    images : List[np.ndarray]

    min_component_size : int


    """
    labeled_images = [measure.label(image) for image in images]
    tables = measure_regions(labeled_images)

    selected = [tables[i][tables[i]["area"] > min_component_size] for i in range(len(tables))]

    filtered_images = []

    for img_idx in range(len(images)):
        table_idx = list(selected[img_idx].index.values)
        print(
            f"For image {img_idx}, out of {len(tables[img_idx])}, keeping {len(table_idx)} components"
        )

        total_mask = np.zeros_like(images[img_idx])

        for iloc in table_idx:
            bb = [
                tables[img_idx]["bb_s_z"][iloc],
                tables[img_idx]["bb_s_x"][iloc],
                tables[img_idx]["bb_s_y"][iloc],
                tables[img_idx]["bb_f_z"][iloc],
                tables[img_idx]["bb_f_x"][iloc],
                tables[img_idx]["bb_f_y"][iloc],
            ]

            mask = (labeled_images[img_idx] == tables[img_idx]["class_code"][iloc]) * 1.0
            total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] = (
                total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
                + mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
            )

        # filtered_images.append((total_mask * images[img_idx]) * 1.0)
        filtered_images.append(total_mask)
    return filtered_images[0], tables, labeled_images


def filter_components2(images, min_component_size=0, max_component_size=1e9):
    """
    Filter components smaller than min_component_size

    Parameters
    ----------
    images : List[np.ndarray]

    min_component_size : int


    """
    labeled_images = [measure.label(image) for image in images]
    tables = measure_regions(labeled_images)

    selected = [
        tables[i][
            np.logical_and(
                tables[i]["area"] > min_component_size, tables[i]["area"] < max_component_size
            )
        ]
        for i in range(len(tables))
    ]

    filtered_images = []

    for img_idx in range(len(images)):
        table_idx = list(selected[img_idx].index.values)
        print(
            f"For image {img_idx}, out of {len(tables[img_idx])}, keeping {len(table_idx)} components"
        )

        total_mask = np.zeros_like(images[img_idx])

        for iloc in table_idx:
            bb = [
                tables[img_idx]["bb_s_z"][iloc],
                tables[img_idx]["bb_s_x"][iloc],
                tables[img_idx]["bb_s_y"][iloc],
                tables[img_idx]["bb_f_z"][iloc],
                tables[img_idx]["bb_f_x"][iloc],
                tables[img_idx]["bb_f_y"][iloc],
            ]

            mask = (labeled_images[img_idx] == tables[img_idx]["class_code"][iloc]) * 1.0
            total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] = (
                total_mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
                + mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]]
            )

        # filtered_images.append((total_mask * images[img_idx]) * 1.0)
        filtered_images.append(total_mask)
    return filtered_images[0], tables, labeled_images


def measure_big_blobs(images: List[np.ndarray]):
    filtered_images = filter_small_components(images)
    labeled_images = [measure.label(image) for image in filtered_images]
    filtered_tables = measure_regions(labeled_images)
    return filtered_tables


def get_entity_at_loc(entities_df, selected_idx):
    return entities_df[
        np.logical_and(
            np.logical_and(entities_df.z == selected_idx[0], entities_df.x == selected_idx[1]),
            entities_df.y == selected_idx[2],
        )
    ]
