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

# from survos2.entity.Various import draw_rect
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, label
from skimage import data, measure
from survos2.frontend.nb_utils import show_images
from survos2.entity.entities import make_entity_bvol

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


def filter_proposal_mask(
    proposal_mask, thresh=0.5, num_erosions=3, num_dilations=3, num_medians=1
):
    holdout = (proposal_mask >= thresh) * 1.0
    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_erosions):
        holdout = ndimage.binary_erosion(holdout, structure=struct2).astype(
            holdout.dtype
        )

    for i in range(num_dilations):
        holdout = ndimage.binary_dilation(holdout, structure=struct2).astype(
            holdout.dtype
        )

    for i in range(num_medians):
        holdout = ndimage.median_filter(holdout, 4).astype(holdout.dtype)

    return holdout


def measure_regions(labeled_images, properties=["label", "area", "centroid", "bbox"]):
    tables = [
        skimage.measure.regionprops_table(image, properties=properties)
        for image in labeled_images
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


def filter_small_components(images, component_size=1000):
    labeled_images = [measure.label(image) for image in images]
    tables = measure_regions(labeled_images)

    too_big = [
        tables[i][tables[i]["area"] > component_size] for i in range(len(tables))
    ]
    # too_small = [
    #    tables[i][tables[i]["area"] < component_size] for i in range(len(tables))
    # ]

    filtered_images = []

    for img_idx in range(len(images)):
        sel_classes = list(too_big[img_idx]["class_code"])
        print(f"Out of {len(tables[0])}, keeping {len(sel_classes)} components")
        total_mask = np.zeros_like(images[img_idx])

        for idx in sel_classes:
            mask = (labeled_images[img_idx] == idx) * 1.0
            total_mask = total_mask + mask

        filtered_images.append((total_mask * images[img_idx]) * 1.0)

    return total_mask


def measure_big_blobs(images: List[np.ndarray]):
    filtered_images = filter_small_components(images)
    labeled_images = [measure.label(image) for image in filtered_images]
    filtered_tables = measure_regions(labeled_images)
    return filtered_tables


def get_entity_at_loc(entities_df, selected_idx):
    return entities_df[
        np.logical_and(
            np.logical_and(
                entities_df.z == selected_idx[0], entities_df.x == selected_idx[1]
            ),
            entities_df.y == selected_idx[2],
        )
    ]

