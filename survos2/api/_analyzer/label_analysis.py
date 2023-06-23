import ntpath
import os
import numpy as np
import pandas as pd
from loguru import logger

from matplotlib import pyplot as plt

from matplotlib.figure import Figure
from scipy import ndimage
from scipy.ndimage import measurements as measure
from skimage.morphology import ball, octahedron
from sklearn.decomposition import PCA

from survos2.api import workspace as ws
from survos2.entity.components import measure_components
from survos2.entity.entities import make_entity_bvol, make_entity_df

from survos2.entity.utils import get_surface

from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.api.utils import pass_through

from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()





FEATURE_OPTIONS = [
    "Average Intensity",
    "Sum Intensity",
    "Standard Deviation",
    "Variance",
    "Volume",
    "Log10(Volume)",
    "Volume (Bounding Box)",
    "Depth (Bounding Box)",
    "Height (Bounding Box)",
    "Width (Bounding Box)",
    "Log10(Volume) (Bounding Box)",
    "Volume (Oriented Bounding Box)",
    "1st Axis (Oriented Bounding Box)",
    "2nd Axis (Oriented Bounding Box)",
    "3rd Axis (Oriented Bounding Box)",
    "Log10(Volume) (Oriented Bounding Box)",
    "Position (X)",
    "Position (Y)",
    "Position (Z)",
    "Segmented Surface Area",
    "Segmented Volume",
    "Segmented Sphericty",
]

FEATURE_TYPES = [
    "intensity",
    "sum",
    "std",
    "var",
    "volume",
    "log_volume",
    "volume_bbox",
    "depth_bbox",
    "height_bbox",
    "width_bbox",
    "log_volume_bbox",
    "volume_ori_bbox",
    "depth_ori_bbox",
    "height_ori_bbox",
    "width_ori_bbox",
    "log_volume_ori_bbox",
    "x_pos",
    "y_pos",
    "z_pos",
    "seg_surface_area",
    "seg_volume",
    "seg_sphericity",
]



def window_to_bb(w):
    return (
        0,
        0,
        (w[0].start + w[0].stop) // 2,
        (w[1].start + w[1].stop) // 2,
        (w[2].start + w[2].stop) // 2,
        w[0].start,
        w[0].stop,
        w[1].start,
        w[1].stop,
        w[2].start,
        w[2].stop,
    )



def windows_to_bvols(windows):
    bbs = []
    for w in windows:
        bbs.append(window_to_bb(w))
    return bbs


def sample_windows(img_volume, win):
    z_st, z_end, y_st, y_end, x_st, x_end = win
    z_st = int(z_st)
    z_end = int(z_end)
    y_st = int(y_st)
    y_end = int(y_end)
    x_st = int(x_st)
    x_end = int(x_end)
    img = img_volume[z_st:z_end, y_st:y_end, x_st:x_end]
    return img





@analyzer.get("/label_analyzer", response_model=None)
def label_analyzer(
    src: str = Body(),
    dst: str = Body(),
    workspace: str = Body(),
    mode: str = Body(),
    pipelines_id: str = Body(),
    analyzers_id: str = Body(),
    annotations_id: str = Body(),
    feature_id: str = Body(),
    split_ops: dict = Body(),
    background_label: int = Body(),
) -> "SEGMENTATION":
    """Analyzes an integer image, finding it's connected components and
    allowing the user to calculate a table of statistics and to plot
    particular features as a histogram.

    Args:
        src (str): Source image URI
        dst (str): Destination image URI
        workspace (str): Workspace to use
        mode (str): Whether to use pipelines, annotations or analyzer images.
        pipelines_id (str): Pipeline URI if mode is pipelines.
        analyzers_id (str): Analyzer URI if mode is analyzers.
        annotations_id (str): Annotations URI if mode is annotations.
        feature_id (str): Feature image to calculate image stats from per-component.
        split_ops (dict): (Not used for label analyzer)
        background_label (int): Label to use as background for connected component calculation.

    Returns:
        list of dicts: List of dicts containing statistics for each component, as well as plot information.
    """

    return label_splitter(
        src,
        dst,
        workspace,
        mode,
        pipelines_id,
        analyzers_id,
        annotations_id,
        feature_id,
        split_ops,
        background_label,
    )


@analyzer.get("/label_splitter", response_model=None)
def label_splitter(
    src: str = Body(),
    dst: str = Body(),
    workspace: str = Body(),
    mode: str = Body(),
    pipelines_id: str = Body(),
    analyzers_id: str = Body(),
    annotations_id: str = Body(),
    feature_id: str = Body(),
    split_ops: dict = Body(),
    background_label: int = Body(),
) -> "SEGMENTATION":
    """Label splitter is Label Analyzer with the added ability to
    use rules to split the components of the image by feature properties.

    Same arguments as Label Analyzer but using:
        split_ops (dict): Dictionary of rules to use to split the components.

    """

    DataModel.g.current_workspace = workspace

    if mode == "1":
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif mode == "2":
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif mode == "3":
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")

    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        seg = DM.sources[0][:]
        logger.debug(f"src_dataset shape {seg[:].shape}")

    seg = seg.astype(np.uint32) & 15

    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature_dataset_arr = DM.sources[0][:]

    # if labels==None:
    labels = set(np.unique(seg)) - set([background_label])
    logger.info(f"Labels in segmentation: {labels}")

    new_labels = np.zeros(seg.shape, np.int32)
    total_labels = 0
    obj_labels = []

    for label in labels:
        mask = seg == label
        tmp_data = seg.copy()
        tmp_data[~mask] = 0
        tmp_labels, num = ndimage.measurements.label(tmp_data, structure=octahedron(1))
        mask = tmp_labels > 0
        new_labels[mask] = tmp_labels[mask] + total_labels
        total_labels += num
        obj_labels += [label] * num

    logger.debug(f"Number of unique labels: {np.unique(new_labels)}")

    objs = new_labels
    # objects = new_labels
    num_objects = total_labels
    logger.debug(f"Number of objects {num_objects}")
    objlabels = np.arange(1, num_objects + 1)

    obj_windows = measure.find_objects(objs)
    feature = []
    depth = []
    height = []
    width = []

    # seg_crops = []
    feature_surface_area = []
    feature_volume = []
    feature_sphericity = []

    for w in obj_windows:
        feature.append(
            (w[0].stop - w[0].start) * (w[1].stop - w[1].start) * (w[2].stop - w[2].start)
        )
        depth.append(w[0].stop - w[0].start)
        height.append(w[1].stop - w[1].start)
        width.append(w[2].stop - w[2].start)

        if (depth[-1] > 2) & (height[-1] > 2) & (width[-1] > 2):
            try:
                _, surface_area, volume, sphericity = get_surface(
                    seg[w[0].start : w[0].stop, w[1].start : w[1].stop, w[2].start : w[2].stop]
                )
                feature_surface_area.append(surface_area)
                feature_volume.append(volume)
                feature_sphericity.append(sphericity)
            except:
                d, h, w = depth[-1], height[-1], width[-1]
                feature_surface_area.append((d * h) + 4 * (h * w) + (d * w))
                feature_volume.append(d * h * w)
                sphericity_of_cube = 0.523599  # rounded to 6 fig
                feature_sphericity.append(sphericity_of_cube)
        else:
            d, h, w = depth[-1], height[-1], width[-1]
            feature_surface_area.append((d * h) + 4 * (h * w) + (d * w))
            feature_volume.append(d * h * w)
            sphericity_of_cube = 0.523599  # rounded to 6 fig
            feature_sphericity.append(sphericity_of_cube)

    ori_feature = []
    ori_depth = []
    ori_height = []
    ori_width = []

    for i, w in enumerate(obj_windows):
        z, y, x = np.where(objs[w] == i + 1)
        coords = np.c_[z, y, x]
        if coords.shape[0] >= 3:
            coords = PCA(n_components=3).fit_transform(coords)
        cmin, cmax = coords.min(0), coords.max(0)
        zz, yy, xx = (
            cmax[0] - cmin[0] + 1,
            cmax[1] - cmin[1] + 1,
            cmax[2] - cmin[2] + 1,
        )
        ori_feature.append(zz * yy * xx)
        ori_depth.append(zz)
        ori_height.append(yy)
        ori_width.append(xx)

    feature_ori = np.asarray(ori_feature, np.float32)
    feature_ori_depth = np.asarray(ori_depth, np.float32)
    feature_ori_height = np.asarray(ori_height, np.float32)
    feature_ori_width = np.asarray(ori_width, np.float32)
    feature_ori_log10 = np.log10(feature_ori)

    feature_bb_vol = np.asarray(feature, np.float32)
    feature_bb_vol_log10 = np.log10(feature_bb_vol)
    feature_bb_depth = np.asarray(depth, np.float32)
    feature_bb_height = np.asarray(height, np.float32)
    feature_bb_width = np.asarray(width, np.float32)

    feature_sum = ndimage.measurements.sum(feature_dataset_arr, objs, index=objlabels)
    feature_mean = ndimage.measurements.mean(feature_dataset_arr, objs, index=objlabels)
    feature_std = ndimage.measurements.standard_deviation(
        feature_dataset_arr, objs, index=objlabels
    )
    feature_var = ndimage.measurements.variance(feature_dataset_arr, objs, index=objlabels)

    feature_pos = measure.center_of_mass(objs, labels=objs, index=objlabels)
    feature_pos = np.asarray(feature_pos, dtype=np.float32)

    features_df = pd.DataFrame(
        {
            "Sum": feature_sum,
            "Mean": feature_mean,
            "Std": feature_std,
            "Var": feature_var,
            "z": feature_pos[:, 0],
            "x": feature_pos[:, 1],
            "y": feature_pos[:, 2],
            "bb_volume": feature_bb_vol,
            "bb_vol_log10": feature_bb_vol_log10,
            "bb_depth": feature_bb_depth,
            "bb_height": feature_bb_height,
            "bb_width": feature_bb_width,
            "ori_volume": feature_ori,
            "ori_vol_log10": feature_ori_log10,
            "ori_depth": feature_ori_depth,
            "ori_height": feature_ori_height,
            "ori_width": feature_ori_width,
            "seg_surface_area": feature_surface_area,
            "seg_volume": feature_volume,
            "seg_sphericity": feature_sphericity,
        }
    )

    sel_start, sel_end = 0, len(features_df)
    features_array = np.array(
        [
            [
                np.int32(np.float32(features_df.iloc[i]["z"])),
                np.int32(np.float32(features_df.iloc[i]["x"])),
                np.int32(np.float32(features_df.iloc[i]["y"])),
                np.float32(np.float32(features_df.iloc[i]["Sum"])),
                np.float32(np.float32(features_df.iloc[i]["Mean"])),
                np.float32(np.float32(features_df.iloc[i]["Std"])),
                np.float32(np.float32(features_df.iloc[i]["Var"])),
                np.float32(np.float32(features_df.iloc[i]["bb_volume"])),
                np.float32(np.float32(features_df.iloc[i]["bb_vol_log10"])),
                np.float32(np.float32(features_df.iloc[i]["bb_depth"])),
                np.float32(np.float32(features_df.iloc[i]["bb_height"])),
                np.float32(np.float32(features_df.iloc[i]["bb_width"])),
                np.float32(np.float32(features_df.iloc[i]["ori_volume"])),
                np.float32(np.float32(features_df.iloc[i]["ori_vol_log10"])),
                np.float32(np.float32(features_df.iloc[i]["ori_depth"])),
                np.float32(np.float32(features_df.iloc[i]["ori_height"])),
                np.float32(np.float32(features_df.iloc[i]["ori_width"])),
                np.float32(np.float32(features_df.iloc[i]["seg_surface_area"])),
                np.float32(np.float32(features_df.iloc[i]["seg_volume"])),
                np.float32(np.float32(features_df.iloc[i]["seg_sphericity"])),
            ]
            for i in range(sel_start, sel_end)
        ]
    )
    result_features = features_array

    rules = []
    calculate = False
    for k in split_ops.keys():
        if k != "context":
            split_op_card = split_ops[k]
            split_feature_index = int(split_op_card["split_feature_index"])
            split_op = int(split_op_card["split_op"])
            split_threshold = float(split_op_card["split_threshold"])

            if int(split_op) > 0:
                calculate = True
                s = int(split_op) - 1  # split_op starts at 1
                feature_names = [
                    "z",
                    "x",
                    "y",
                    "Sum",
                    "Mean",
                    "Std",
                    "Var",
                    "bb_vol",
                    "bb_vol_log10",
                    "bb_vol_depth",
                    "bb_vol_depth",
                    "bb_vol_height",
                    "bb_vol_width",
                    "ori_vol",
                    "ori_vol_log10",
                    "ori_vol_depth",
                    "ori_vol_depth",
                    "ori_vol_height",
                    "ori_vol_width",
                    "seg_surface_area",
                    "seg_volume",
                    "seg_sphericity",
                ]
                feature_index = int(split_feature_index)  # feature_names.index(split_feature_index)
                rules.append((int(feature_index), s, split_threshold))
                logger.debug(
                    f"Adding split rule: {split_feature_index} {split_op} {split_threshold}"
                )

        if calculate:
            masked_out, result_features = apply_rules(
                features_array, -1, rules, np.array(objlabels), num_objects
            )
            logger.debug(f"Masking out: {masked_out}")
            for i, l in enumerate(masked_out):
                if l != -1:
                    new_labels[new_labels == l] = 0
            new_labels = (new_labels > 0) * 1.0
        else:
            result_features = features_array

    map_blocks(pass_through, new_labels, out=dst, normalize=False)

    result_features = np.array(result_features)
    result_features = result_features.tolist()
    features_array = np.array(features_array)
    features_array = features_array.tolist()

    return result_features, features_array  # , bvols


def apply_rules(features: np.ndarray, label: int, rules: tuple, out: np.ndarray, num_objects: int):
    logger.debug("Applying rules")
    mask = np.ones(num_objects, dtype=bool)

    for f, s, t in rules:
        if s == 0:
            np.logical_and(mask, features[:, f] > t, out=mask)
            result_features = features[features[:, f] > t]
        else:
            np.logical_and(mask, features[:, f] < t, out=mask)
            result_features = features[features[:, f] < t]

    out[mask] = label
    return out, result_features



