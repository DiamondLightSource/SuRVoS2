import ast
import pandas as pd
import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from statistics import mode, StatisticsError
from loguru import logger

# from survos2.frontend.utils import get_img_in_bbox
from survos2.entity.sampler import get_img_in_bbox

import skimage
from skimage import img_as_ubyte
import imageio
from survos2.entity.entities import make_entity_df
from sklearn.cluster import DBSCAN
import hdbscan


def aggregate(
    entity_df,
    img_shape,
    remove_outliers=False,
    outlier_score_thresh=0.9,
    max_num_points=34,
    params={"algorithm": "HDBSCAN", "min_cluster_size": 2, "min_samples": 1},
):
    """Aggregate classified points. Takes a dataframe of entities and an image size, clusters points
    (filtering out large clusters) and performs majority voting on the cluster class.

    Parameters
    ----------
    entity_df : Dataframe
        Dataframe of entities
    img_shape : tuple
        Image size as a tuple
    outlier_score_thresh : float, optional
        Score above which to remove, by default 0.9
    max_num_points : int, optional
        Maximum number of points to keep a cluster, by default 34
    params : dict, optional
        Parameter dictionary, by default {"algorithm": "HDBSCAN", "min_cluster_size": 2, "min_samples": 1}

    Returns
    -------
    Dataframe
        Dataframe of aggregated entities.
    """
    entity_df = make_entity_df(np.array(entity_df))
    X_rescaled, scaling = normalized_coords2(entity_df, img_shape)

    # use either HDBSCAN or DBSCAN to cluster the points
    if params["algorithm"] == "HDBSCAN":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"]
        ).fit(X_rescaled)

        label_code = clusterer.labels_
        num_clusters_found = len(np.unique(label_code))
        print(f"Number of clustered found {num_clusters_found}")
        core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
        core_samples_mask = clusterer.outlier_scores_ < outlier_score_thresh
        # core_samples_mask[db.core_sample_indices_] = True

        labels = clusterer.labels_
        if remove_outliers:
            labels = labels[core_samples_mask]
    else:
        clusterer = DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit(X_rescaled)
        label_code = clusterer.labels_
        num_clusters_found = len(np.unique(label_code))
        print(f"Number of clustered found {num_clusters_found}")
        labels = clusterer.labels_

    # Separate clusters into two groups based on the size of the cluster
    cluster_coords = []
    cluster_sizes = []
    other_coords = []

    for l in np.unique(labels)[0:]:
        if np.sum(labels == l) < max_num_points:
            cluster_coords.append(X_rescaled[labels == l])
            cluster_sizes.append(np.sum(labels == l))
        else:
            other_coords.append(X_rescaled[labels == l])

    cluster_coords = np.array(cluster_coords)
    cluster_sizes = np.array(cluster_sizes)
    print(f"Mean cluster size: {np.mean(cluster_sizes)}")
    refined_ent = np.concatenate(cluster_coords)
    print(f"Refined entity array shape {refined_ent.shape}")

    # Majority voting on class within a cluster
    agg = aggregate_cluster_votes(cluster_coords)
    refined_ent = np.array([centroid_3d_with_class(c) for c in agg])

    refined_ent[:, 0] = refined_ent[:, 0] * 1 / scaling[0]
    refined_ent[:, 1] = refined_ent[:, 1] * 1 / scaling[2]
    refined_ent[:, 2] = refined_ent[:, 2] * 1 / scaling[1]
    refined_entity_df = make_entity_df(refined_ent, flipxy=False)
    print(f"Aggregated entity length {len(agg)}")
    return refined_entity_df


def normalized_coords2(entities_df, img_shape, scale_minmax=False):
    """Scale points so max dimensions of point volume is between 0 and 1"""
    oe = np.array(entities_df)
    orig_pts = oe.astype(np.float32)
    X = orig_pts.copy()

    X_rescaled = orig_pts.copy()

    scale_x = 1.0 / np.max(X[:, 1])
    scale_y = 1.0 / np.max(X[:, 2])
    scale_z = (1.0 / np.max(X[:, 0])) * 0.02

    if scale_minmax:
        xlim = (
            np.min(X_rescaled[:, 1]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 1]).astype(np.uint16) + 0.1,
        )
        ylim = (
            np.min(X_rescaled[:, 2]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 2]).astype(np.uint16) + 0.1,
        )
        zlim = (
            np.min(X_rescaled[:, 0]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 0]).astype(np.uint16) + 0.1,
        )
    else:
        xlim = (0, img_shape[1])
        ylim = (0, img_shape[2])
        zlim = (0, img_shape[0])

    X_rescaled[:, 1] *= scale_x
    X_rescaled[:, 2] *= scale_y
    X_rescaled[:, 0] *= scale_z

    print(f"Rescaled to {scale_x}, {scale_y}, {scale_z}")

    return X_rescaled, (scale_z, scale_x, scale_y)


def normalized_coords(entities_df, img_volume, scale_minmax=False):
    """Scale points so max dimensions of point volume is between 0 and 1"""
    chip = img_volume
    oe = np.array(entities_df)
    orig_pts = oe.astype(np.float32)
    X = orig_pts.copy()

    X_rescaled = orig_pts.copy()

    scale_x = 1.0 / np.max(X[:, 1])
    scale_y = 1.0 / np.max(X[:, 2])
    scale_z = (1.0 / np.max(X[:, 0])) * 0.02

    if scale_minmax:
        xlim = (
            np.min(X_rescaled[:, 1]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 1]).astype(np.uint16) + 0.1,
        )
        ylim = (
            np.min(X_rescaled[:, 2]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 2]).astype(np.uint16) + 0.1,
        )
        zlim = (
            np.min(X_rescaled[:, 0]).astype(np.uint16) - 0.1,
            np.max(X_rescaled[:, 0]).astype(np.uint16) + 0.1,
        )
    else:
        xlim = (0, chip.shape[1])
        ylim = (0, chip.shape[2])
        zlim = (0, chip.shape[0])

    X_rescaled[:, 1] *= scale_x
    X_rescaled[:, 2] *= scale_y
    X_rescaled[:, 0] *= scale_z

    print(f"Rescaled to {scale_x}, {scale_y}, {scale_z}")

    return X_rescaled, (scale_z, scale_x, scale_y)


def get_window(image_volume, sliceno, xstart, ystart, xend, yend):
    return image_volume[sliceno, xstart:xend, ystart:yend]


def aggregate_cluster_votes(cluster_coords):
    agg = []

    for c in cluster_coords:
        cluster_df = pd.DataFrame(
            np.array(c), columns=["slice", "click_x", "click_y", "class_code"]
        )
        cluster_df = cluster_df.astype(
            {
                "slice": "float32",
                "click_x": "float32",
                "click_y": "float32",
                "class_code": "int32",
            }
        )
        agg.append(majority_voting(cluster_df))
    return agg


def cluster_centroid_coords(X_rescaled, labels, MAX_NUM_PTS_PER_CLUSTER=20):
    unique_labels = set(labels)
    # print(unique_labels)
    cluster_coords = []
    cluster_sizes = []
    other_coords = []

    for l in np.unique(labels)[0:]:
        if np.sum(labels == l) < MAX_NUM_PTS_PER_CLUSTER:
            cluster_coords.append(X_rescaled[labels == l])
            cluster_sizes.append(np.sum(labels == l))
        else:
            other_coords.append(X_rescaled[labels == l])

    cluster_coords = np.array(cluster_coords)
    cluster_sizes = np.array(cluster_sizes)
    print(f"Mean cluster size: {np.mean(cluster_sizes)}")

    return cluster_coords, cluster_sizes


def centroid_3d_with_class(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x / length, sum_y / length, sum_z / length, arr[0, 3]


def majority_voting(cluster_df):
    """Given a dataframe of points with classes, take the unique_centroids and gather
    all points with the same centroid and assign the class to that centroid that is
    the class of the majority of points.

    Parameters
    ----------
    cluster_df : Dataframe
        Dataframe of points with classes

    Returns
    -------
    Dataframe
        Dataframe of points with classes
    """
    unique_centroids = cluster_df.drop_duplicates(["slice", "click_x", "click_y"])
    # unique_centroids = cluster_df
    key_names = ["slice", "click_x", "click_y"]
    cc = []

    # majority vote
    for i in range(len(unique_centroids)):
        key = (
            unique_centroids.iloc[i]["slice"],
            unique_centroids.iloc[i]["click_x"],
            unique_centroids.iloc[i]["click_y"],
        )
        # get the class code for each of the points with a common centroid
        class_codes = list(cluster_df[(cluster_df[key_names] == key).all(1)]["class_code"])
        # choose the most common class to be the class of the centroid

        c = Counter(class_codes)
        classes_mostcommon = c.most_common(1)
        # print(classes_mostcommon)
        cc.append((key[0], key[2], key[1], classes_mostcommon[0][0]))

    output_df = pd.DataFrame(np.array(cc), columns=["slice", "click_x", "click_y", "class_code"])

    output_df = output_df.astype(
        {
            "slice": "float32",
            "click_x": "float32",
            "click_y": "float32",
            "class_code": "int32",
        }
    )

    return np.array(output_df)


def aggregate_common_centroids(entities):
    unique_centroids = entities.drop_duplicates(["z", "x", "y"])
    key_names = ["z", "x", "y"]
    # keys = centroids_key[i]
    cc = []
    print(f"{len(unique_centroids)} unique centroids")

    # majority vote
    for i in range(len(unique_centroids)):
        key = (
            unique_centroids.iloc[i]["z"],
            unique_centroids.iloc[i]["x"],
            unique_centroids.iloc[i]["y"],
        )
        # print(key)
        class_codes = list(entities[(entities[key_names] == key).all(1)]["class_code"])
        # print(class_codes)
        # print(f"Mode: {mode(class_codes)}")
        c = Counter(class_codes)
        classes_mostcommon = c.most_common(1)
        # print(classes_mode)
        cc.append((key[0], key[2], key[1], classes_mostcommon[0][0]))

    entities_df = pd.DataFrame(np.array(cc), columns=["z", "x", "y", "class_code"])

    entities_df = entities_df.astype(
        {"z": "int32", "x": "int32", "y": "int32", "class_code": "int32"}
    )

    return entities_df


def parse_tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return


##########################################################
def extract_session_roi(
    roi_data, range_start=0, range_end=50, plot_slices=False, debug_verbose=False
):
    session_extract = []
    click_data = []  # list of per-session data
    global_idx = 0

    for idx in range(range_start, range_end):
        if idx % 500 == 0:
            print("Extracting roi for parent_data_roi index: {}".format(idx))
        roi_str = roi_data["parent_data_roi"][idx]
        roi = parse_tuple(roi_str.decode())

        sliceno, xstart, ystart, xend, yend = roi
        session_anno = parse_tuple(roi_data["roi_coord_tuples"][idx].decode())
        classification_id, roi_str2, clicks_list = session_anno
        session_extract.append((classification_id, roi_str2))
        clicks_arr = np.array(clicks_list)

        if debug_verbose:
            print("Extracting roi for parent_data_roi index: {}".format(idx))
            # print(roi_str)
            print("Session data: {}".format(session_anno))
            print("Roi strings: {}\n {}", roi_str, roi_str2)
            print("Clicks array: {}".format(clicks_arr))

        # extract per-session data and create list of session data
        session = []

        xs = []  # just for visualisation below
        ys = []

        discard_idx = []

        for click_idx in range(clicks_arr.shape[0]):
            x = xstart + clicks_arr[click_idx, 0]
            y = ystart + clicks_arr[click_idx, 1]

            session.append((sliceno, x, y))
            xs.append(x)
            ys.append(y)

        click_data.extend(session)

        # For viz and debug
        if plot_slices:
            imstack = wf1

            droi = imstack[sliceno, :, :].copy()
            droi = np.fliplr(droi)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(droi[xstart:xend, ystart:yend])

            plt.scatter(xs, ys, c="red")

    return click_data


def extract_session_roi2(
    imstack,
    zoon_list,
    range_start=0,
    range_end=50,
    plot_slices=False,
    debug_verbose=False,
):
    click_data = []  # list of per-session data
    global_idx = 0

    for idx in range(range_start, range_end):
        if idx % 5000 == 0:
            print("Extracting roi for parent_data_roi index: {}".format(idx))

        zoon_click = zoon_list[idx]
        sliceno, xstart, ystart, xend, yend, x, y = zoon_click

        if debug_verbose:
            print("Extracting roi for parent_data_roi index: {}".format(idx))
            print(roi_str)
            print("Roi strings: {}\n {}", roi_str)
            print("Clicks array: {}".format(clicks_arr))

        session = []
        xs = []
        ys = []

        discard_idx = []

        x = xstart + x
        y = ystart + y

        session.append((sliceno, x, y))
        xs.append(x)
        ys.append(y)

        click_data.extend(session)

        if plot_slices:
            droi = imstack[sliceno, :, :].copy()
            droi = np.fliplr(droi)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(droi[xstart:xend, ystart:yend])

            plt.scatter(xs, ys, c="red")

    return click_data


def generate_clicklist(click_data_arr, crop_roi, slicestart, sliceend):
    accum = 0
    lengths = []
    click_coords = []

    _, xstart, ystart, xend, yend = crop_roi

    max_x = np.max(click_data_arr[:, 1])
    min_x = np.min(click_data_arr[:, 1])
    x_coords_range = max_x - min_x
    y_coords_range = np.max(click_data_arr[:, 2]) - np.min(click_data_arr[:, 2])

    scale_factor_x = 1  # vol_shape_x / y_coords_range
    scale_factor_y = 1  # vol_shape_y / x_coords_range

    sel_clicks = click_data_arr[
        np.where(
            np.logical_and(click_data_arr[:, 0] >= slicestart, click_data_arr[:, 0] <= sliceend)
        )
    ]

    return sel_clicks, x_coords_range, y_coords_range


def generate_click_plot_data(img_data, click_coords, patch_size=(40, 40)):
    img_shortlist = []
    img_titles = []

    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size
        sel_cropped_clicks.append((sliceno, x, y, w, h))
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        img_titles.append(str(int(x)) + "_" + str(int(y)) + "_" + str(sliceno))

    return img_shortlist, img_titles


def generate_full_click_plot_data(img_data, click_coords, patch_size=(40, 40)):
    img_shortlist = []
    img_titles = []
    img_coords = []

    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size
        img_coords.append((sliceno, x, y, w, h))
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        # y_str = "{:.3f}".format(y)
        # x_str = "{:.3f}".format(x)
        img_titles.append(str(int(x)) + "_" + str(int(y)) + "_" + str(int(sliceno)))

    return img_shortlist, img_titles, img_coords


##########################################################################################################################
def generate_click_plot_data1(img_data, click_coords, patch_size=(40, 40)):
    sel_cropped_clicks = []
    img_shortlist = []
    img_titles = []
    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size
        sel_cropped_clicks.append((sliceno, x, y, w, h))
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        y_str = "{:10.4f}".format(y)
        x_str = "{:10.4f}".format(x)
        img_titles.append(x_str + " " + y_str + "\n " + "Slice no: " + str(sliceno))

    return img_shortlist, img_titles


def get_img_in_bbox1(image_volume, sliceno, x, y, w, h):
    return image_volume[int(sliceno), x - w : x + w, y - h : y + h]


def generate_clicklist(click_data_arr, crop_roi, slicestart, sliceend):
    accum = 0
    lengths = []
    click_coords = []

    _, xstart, ystart, xend, yend = crop_roi
    max_x = np.max(click_data_arr[:, 1])
    min_x = np.min(click_data_arr[:, 1])
    x_coords_range = max_x - min_x
    y_coords_range = np.max(click_data_arr[:, 2]) - np.min(click_data_arr[:, 2])

    scale_factor_x = 1  # vol_shape_x / y_coords_range
    scale_factor_y = 1  # vol_shape_y / x_coords_range

    sel_clicks = click_data_arr[
        np.where(
            np.logical_and(click_data_arr[:, 0] >= slicestart, click_data_arr[:, 0] <= sliceend)
        )
    ]

    return sel_clicks, x_coords_range, y_coords_range


def generate_click_plot_data1(img_data, click_coords):
    img_shortlist = []
    img_titles = []

    for j in range(len(click_coords)):
        if j % 5000 == 0:
            print("Generating click plot data: {}".format(j))

        sliceno, y, x = click_coords[j]
        w, h = (100, 100)
        print(x, y, w, h, sliceno)

        img = get_img_in_bbox(img_data, 75, int(np.ceil(x)), int(np.ceil(y)), w, h)
        img_shortlist.append(img)

        y_str = "{:10.4f}".format(y)
        x_str = "{:10.4f}".format(x)
        img_titles.append(x_str + " " + y_str + " " + "Slice no: " + str(sliceno))

    return img_shortlist, img_titles


def generate_click_plot_data(img_data, click_coords):
    img_shortlist = []
    img_titles = []
    for j in range(len(click_coords)):
        if j % 5000 == 0:
            print("Generating click plot data: {}".format(j))
        sliceno, y, x = click_coords[j]
        w, h = (100, 100)
        print(x, y, w, h, sliceno)
        img = get_img_in_bbox(img_data, 75, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        y_str = "{:10.4f}".format(y)
        x_str = "{:10.4f}".format(x)
        img_titles.append(x_str + " " + y_str + " " + "Slice no: " + str(sliceno))
    return img_shortlist, img_titles


def generate_click_plot_data_cropped(img_data, click_coords, bv, patch_size=(40, 40)):
    sel_cropped_clicks = []
    img_shortlist = []
    img_titles = []
    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size

        if sliceno < bv[1] and sliceno > bv[0]:
            sel_cropped_clicks.append((sliceno, x, y, w, h))
            img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

            img_shortlist.append(img)
            y_str = "{:10.4f}".format(y)
            x_str = "{:10.4f}".format(x)
            img_titles.append(x_str + " " + y_str + "\n " + "Slice no: " + str(sliceno))

    return img_shortlist, img_titles


def get_img_in_bbox2(image_volume, sliceno, x, y, w, h):
    return image_volume[int(sliceno), x - w : x + w, y - h : y + h]


def generate_click_plot_data2(img_data, click_coords, patch_size=(40, 40)):
    sel_cropped_clicks = []

    img_shortlist = []
    img_titles = []
    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size
        # print(x,y,w,h,sliceno)
        sel_cropped_clicks.append((sliceno, x, y, w, h))
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        # y_str = "{:.3f}".format(y)
        # x_str = "{:.3f}".format(x)
        img_titles.append(str(int(x)) + "_" + str(int(y)) + "_" + str(sliceno))

    return img_shortlist, img_titles


def generate_full_click_plot_data(img_data, click_coords, patch_size=(40, 40)):
    img_shortlist = []
    img_titles = []
    img_coords = []
    for j in range(len(click_coords)):
        sliceno, y, x, c = click_coords[j]
        w, h = patch_size
        # print(x,y,w,h,sliceno)
        img_coords.append((sliceno, x, y, w, h))

        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

        img_shortlist.append(img)
        # y_str = "{:.3f}".format(y)
        # x_str = "{:.3f}".format(x)
        img_titles.append(str(int(x)) + "_" + str(int(y)) + "_" + str(int(sliceno)) + str(c))

    return img_shortlist, img_titles, img_coords


def generate_stacked_click_plot_data(img_data_layers, click_coords, patch_size=(32, 32)):
    img_shortlistA = []
    img_shortlistB = []
    img_titles = []

    img_data1, img_data2 = img_data_layers

    for j in range(len(click_coords)):
        sliceno, y, x = click_coords[j]
        w, h = patch_size
        w = w // 2
        h = h // 2
        # print(x,y,w,h,sliceno)
        sel_cropped_clicks.append((sliceno, x, y, w, h))
        # img1 = get_img_in_bbox(img_data1, sliceno-1, int(np.ceil(x)),int(np.ceil(y)),w,h)
        img2 = get_img_in_bbox(img_data1, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)
        # img3 = get_img_in_bbox(img_data1, sliceno-1, int(np.ceil(x)),int(np.ceil(y)),w,h)

        # stacked_img = np.stack((img1, img2, img3), axis=-1)

        img_shortlistA.append(img2)

        # img1 = get_img_in_bbox(img_data1, sliceno-1, int(np.ceil(x)),int(np.ceil(y)),w,h)
        # img2 = get_img_in_bbox(img_data1, sliceno, int(np.ceil(x)),int(np.ceil(y)),w,h)
        # img3 = get_img_in_bbox(img_data1, sliceno+1, int(np.ceil(x)),int(np.ceil(y)),w,h)

        img1 = get_img_in_bbox(img_data2, sliceno - 1, int(np.ceil(x)), int(np.ceil(y)), w, h)
        img2 = get_img_in_bbox(img_data1, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)
        img3 = get_img_in_bbox(img_data2, sliceno - 1, int(np.ceil(x)), int(np.ceil(y)), w, h)

        stacked_img = np.stack((img1, img2, img3), axis=-1)

        img_shortlistB.append(stacked_img)

        # Generate a list of titles
        y_str = "{:10.4f}".format(y)
        x_str = "{:10.4f}".format(x)

        img_titles.append(x_str + " " + y_str + "\n " + "Slice no: " + str(sliceno))
        img_shortlists = [img_shortlistA, img_shortlistB]

    return img_shortlists, img_titles
