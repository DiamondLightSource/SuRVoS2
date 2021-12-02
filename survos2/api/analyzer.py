import logging
import ntpath
import os.path as op
import pandas as pd
import dask.array as da
from numba.core.types.scalars import Integer
import hug
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
from loguru import logger
import tempfile

from matplotlib import offsetbox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from survos2.api import workspace as ws
from survos2.api.types import (
    DataURI,
    DataURIList,
    Float,
    FloatList,
    FloatOrVector,
    IntOrVector,
    Int,
    IntList,
    SmartBoolean,
    String,
)

from scipy import ndimage
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.frontend.components.entity import setup_entity_table
from survos2.frontend.nb_utils import summary_stats
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy
from survos2.entity.entities import make_entity_df, make_entity_bvol

from scipy.ndimage.measurements import label as splabel
from skimage.morphology import octahedron, ball
from scipy.ndimage import measurements as measure
from scipy.stats import binned_statistic
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
from survos2.improc import map_blocks
from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.entity.components import measure_components


def pass_through(x):
    return x

__analyzer_fill__ = 0
__analyzer_dtype__ = "uint32"
__analyzer_group__ = "analyzer"
__analyzer_names__ = [
    "find_connected_components",
    "object_stats",
    "object_detection_stats",
    "segmentation_stats",
    "label_splitter",
    "binary_image_stats",
    # "detector_predict",
    "spatial_clustering",
    "remove_masked_objects",
]


def component_bounding_boxes(images):
    bbs_tables = []
    bbs_arrs = []

    for image in images:
        bbs_arr = measure_components(image)
        bbs_arrs.append(bbs_arr)
        bbs_table = make_entity_bvol(bbs_arr)
        bbs_tables.append(bbs_table)

    return bbs_tables, bbs_arrs


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
]



@hug.get()
def label_splitter(
    src: DataURI, 
    dst: DataURI,
    mode: String,
    pipelines_id: DataURI,
    analyzers_id: DataURI,
    annotations_id: DataURI,
    feature_id: DataURI,
    split_ops : dict,
    background_label : Int
) -> "SEGMENTATION":

    
    if mode == '1':
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id), group="pipelines")
        print(f"Analyzer calc on pipeline src {src}")
    elif mode == '2':
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id), group="analyzer")
        print(f"Analyzer calc on analyzer src {src}")
    elif mode == '3':
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id), group="annotations")
        print(f"Analyzer calc on annotation src {src}")
    
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

    print(f"Number of unique labels: {np.unique(new_labels)}")

    objs = new_labels
    #objects = new_labels
    num_objects = total_labels
    logger.debug(f"Number of objects {num_objects}")
    objlabels = np.arange(1, num_objects + 1)

    obj_windows = measure.find_objects(objs)
    feature = []
    depth = []
    height = []
    width = []
    for w in obj_windows:
        feature.append(
            (w[0].stop - w[0].start)
            * (w[1].stop - w[1].start)
            * (w[2].stop - w[2].start)
        )
        depth.append(w[0].stop - w[0].start)
        height.append(w[1].stop - w[1].start)
        width.append(w[2].stop - w[2].start)

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
    feature_var = ndimage.measurements.variance(
        feature_dataset_arr, objs, index=objlabels
    )

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
                np.float32(np.float32(features_df.iloc[i]["ori_width"]))
            ]
            for i in range(sel_start, sel_end)
        ]
    )

    print(split_ops)
    rules = []
    for k in split_ops.keys():
        if k != 'context':
            split_op_card = split_ops[k]
            print(split_op_card)
            split_feature_index = int(split_op_card["split_feature_index"])
            split_op = int(split_op_card["split_op"])
            split_threshold = float(split_op_card["split_threshold"])
            
            if int(split_op) > 0:
                calculate =True
                s = int(split_op) - 1  # split_op starts at 1
                feature_names = ["z", "x", "y", "Sum", "Mean", "Std", "Var", "bb_vol", "bb_vol_log10", "bb_vol_depth", "bb_vol_depth","bb_vol_height", "bb_vol_width", "ori_vol", "ori_vol_log10", "ori_vol_depth", "ori_vol_depth","ori_vol_height", "ori_vol_width"]
                feature_index = int(
                    split_feature_index
                )  # feature_names.index(split_feature_index)
                rules.append((int(feature_index), s, split_threshold))
                print(f"Adding split rule: {split_feature_index} {split_op} {split_threshold}")
            else:
                calculate = False

    if calculate:
        masked_out, result_features = apply_rules(
            features_array, -1, rules, np.array(objlabels), num_objects
        )
        print(f"Masking out: {masked_out}")
        bg_label = max(np.unique(new_labels))
        print(f"Masking out bg label {bg_label}")
        for i, l in enumerate(masked_out):
            if l != -1:
                new_labels[new_labels == l] = 0

        new_labels[new_labels == bg_label] = 0
        new_labels = (new_labels > 0) * 1.0
    else:
        result_features = features_array
    
    
    map_blocks(pass_through, new_labels, out=dst, normalize=False)

    return result_features, features_array 

def apply_rules(
    features: np.ndarray, label: int, rules: tuple, out: np.ndarray, num_objects: int
):
    logger.debug("Applying rules")
    mask = np.ones(num_objects, dtype=np.bool)

    print(f"Label selected {label}")
    print(mask.shape)
    print(features.shape)
    print(out.shape)
    print(f"rules {rules}")

    for f, s, t in rules:
        if s == 0:
            np.logical_and(mask, features[:, f] > t, out=mask)
            result_features = features[features[:, f] > t]
        else:
            np.logical_and(mask, features[:, f] < t, out=mask)
            result_features = features[features[:, f] < t]

    out[mask] = label
    return out, result_features


def detect_blobs(
    padded_proposal,
    area_min=50,
    area_max=50000,
    plot_all=False,
):
    images = [padded_proposal]
    bbs_tables, bbs_arrs = component_bounding_boxes(images)
    print(f"Detecting blobs on image of shape {padded_proposal.shape}")
    zidx = padded_proposal.shape[0] // 2

    from survos2.frontend.nb_utils import summary_stats

    print("Component stats: ")
    print(f"{summary_stats(bbs_tables[0]['area'])}")

    if plot_all:
        for idx in range(len(bbs_tables)):
            print(idx)
            plt.figure(figsize=(5, 5))
            plt.imshow(images[idx][zidx, :], cmap="gray")
            plt.scatter(bbs_arrs[idx][:, 4], bbs_arrs[idx][:, 3])

    selected_entities = bbs_tables[0][
        (bbs_tables[0]["area"] > area_min) & (bbs_tables[0]["area"] < area_max)
    ]
    print(f"Number of selected entities {len(selected_entities)}")

    return bbs_tables, selected_entities


def plot_clustered_img(
    proj,
    colors,
    images=None,
    ax=None,
    thumb_frac=0.02,
    cmap="gray",
    title="Clustered Images",
):

    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    # ax = ax or plt.gca()
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(proj.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2 / 5:
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap, zorder=1), proj[i]
            )
            ax.add_artist(imagebox)

    ax.scatter(proj[:, 0], proj[:, 1], lw=0, s=40, zorder=200)


# @hug.get()
# def level_image_stats(
#     src: DataURI, dst: DataURI, anno_id: DataURI, label_index: Int
# ) -> "SEGMENTATION":
#     logger.debug(f"Finding connected components on annotation: {anno_id}")

#     with DatasetManager(anno_id, out=None, dtype="uint16", fillvalue=0) as DM:
#         src1_dataset = DM.sources[0]
#         anno_level = src1_dataset[:] & 15
#         logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

#     bbs_tables, selected_entities = detect_blobs((anno_level == label_index) * 1.0)
#     print(bbs_tables)
#     print(selected_entities)

#     result_list = []
#     for i in range(len(bbs_tables[0])):
#         result_list.append(
#             [
#                 bbs_tables[0].iloc[i]["area"],
#                 bbs_tables[0].iloc[i]["z"],
#                 bbs_tables[0].iloc[i]["y"],
#                 bbs_tables[0].iloc[i]["x"],
#             ]
#         )

#     return result_list


@hug.get()
def find_connected_components(
    src: DataURI, dst: DataURI, pipelines_id: DataURI,  label_index: Int, workspace: String
) -> "SEGMENTATION":
    logger.debug(f"Finding connected components on segmentation: {pipelines_id}")
    print(f"{DataModel.g.current_workspace}")
    src = DataModel.g.dataset_uri(pipelines_id, group="pipelines")
    print(src)
    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]
        logger.debug(f"src_dataset shape {src_dataset_arr[:].shape}")

    single_label_level = (src_dataset_arr == label_index) * 1.0

    bbs_tables, selected_entities = detect_blobs(single_label_level)
    print(bbs_tables)
    print(selected_entities)

    result_list = []
    for i in range(len(bbs_tables[0])):
        result_list.append(
            [
                bbs_tables[0].iloc[i]["area"],
                bbs_tables[0].iloc[i]["z"],
                bbs_tables[0].iloc[i]["y"],
                bbs_tables[0].iloc[i]["x"],
            ]
        )

    map_blocks(pass_through, single_label_level, out=dst, normalize=False)
    
    print(result_list)
    return result_list


@hug.get()
def binary_image_stats(
    src: DataURI, dst: DataURI, feature_id: DataURI, threshold: Float = 0.5
) -> "IMAGE":
    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    src_thresh = (src_dataset_arr > threshold) * 1.0
    bbs_tables, selected_entities = detect_blobs(src_thresh)
    print(bbs_tables)
    print(selected_entities)

    result_list = []
    for i in range(len(bbs_tables[0])):
        result_list.append(
            [
                bbs_tables[0].iloc[i]["area"],
                bbs_tables[0].iloc[i]["z"],
                bbs_tables[0].iloc[i]["x"],
                bbs_tables[0].iloc[i]["y"],
            ]
        )

    return result_list


def plot_to_image(arr, title="Histogram", vert_line_at=None):
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    # ax.axis("off")
    y, x, _ = ax.hist(arr, bins=16)
    fig.suptitle(title)

    if vert_line_at:
        print(f"Plotting vertical line at: {vert_line_at} {y.max()}")
        ax.axvline(x=vert_line_at, ymin=0, ymax=y.max(), color="r")

    canvas.draw()
    plot_image = np.asarray(canvas.buffer_rgba())
    return plot_image


# @hug.get()
# @save_metadata
# def image_stats(
#     src: DataURI, dst: DataURI,feature_ids: DataURIList, mask_ids: DataURIList
# ) -> "IMAGE":
#     logger.debug(f"Calculating stats on features: {feature_ids}")
#     src = DataModel.g.dataset_uri(ntpath.basename(feature_ids[0]), group="features")

#     with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
#         src_dataset_arr = DM.sources[0][:]
#         logger.debug(f"summary_stats {summary_stats(src_dataset_arr)}")

#     mask_src = DataModel.g.dataset_uri(ntpath.basename(mask_ids[0]), group="features")
#     with DatasetManager(mask_src, out=None, dtype="float32", fillvalue=0) as DM:
#         mask = DM.sources[0][:]

#     plot_image = plot_to_image((src_dataset_arr * mask).flatten(), "Masked histogram")
#     return encode_numpy(plot_image)


@hug.get()
def spatial_clustering(
    src: DataURI, 
    feature_id: DataURI,
    object_id: DataURI,
    workspace : String,
    params : dict,
) -> "OBJECTS":
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    scale = ds_objects.get_metadata("scale")
    print(f"Scaling objects by: {scale}")

    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname)

    logger.debug(f"Spatial clustering using feature as reference image: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    from survos2.entity.anno.crowd import aggregate

    refined_entity_df = aggregate(entities_df, src_dataset_arr.shape, params=params)
    print(refined_entity_df)

    result_list = []
    for i in range(len(refined_entity_df)):
        result_list.append(
            [
                refined_entity_df.iloc[i]["class_code"],
                refined_entity_df.iloc[i]["z"],
                refined_entity_df.iloc[i]["y"],
                refined_entity_df.iloc[i]["x"],
            ]
        )

    return result_list


def remove_masked_entities(bg_mask, entities):
    pts_vol = np.zeros_like(bg_mask)
    logger.debug(f"Masking on mask of shape {pts_vol.shape}")
    entities = entities.astype(np.uint32)
    for pt in entities:
        if (
            (pt[0] > 0)
            & (pt[0] < pts_vol.shape[0])
            & (pt[1] > 0)
            & (pt[1] < pts_vol.shape[1])
            & (pt[2] > 0)
            & (pt[2] < pts_vol.shape[2])
        ):
            pts_vol[pt[0], pt[1], pt[2]] = 1
    pts_vol = pts_vol * (1.0 - bg_mask)
    zs, xs, ys = np.where(pts_vol == 1)
    masked_entities = []
    for i in range(len(zs)):
        pt = [zs[i], ys[i], xs[i], 6]
        masked_entities.append(pt)
    return np.array(masked_entities)


@hug.get()
@save_metadata
def remove_masked_objects(
    src: DataURI, dst: DataURI,
    feature_id: DataURI,
    object_id: DataURI,
) -> "OBJECTS":
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    scale = ds_objects.get_metadata("scale")
    print(f"Scaling objects by: {scale}")

    objects_fullname = ds_objects.get_metadata("fullname")
    objects_scale = ds_objects.get_metadata("scale")
    objects_offset = ds_objects.get_metadata("offset")
    objects_crop_start = ds_objects.get_metadata("crop_start")
    objects_crop_end = ds_objects.get_metadata("crop_end")

    logger.debug(f"Getting objects from {src} and file {objects_fullname}")
    from survos2.frontend.components.entity import make_entity_df, setup_entity_table

    tabledata, entities_df = setup_entity_table(
        objects_fullname,
        scale=objects_scale,
        offset=objects_offset,
        crop_start=objects_crop_start,
        crop_end=objects_crop_end,
    )

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=False))

    logger.debug(f"Removing entities using feature as mask: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        mask = DM.sources[0][:]

    logger.debug(f"Initial number of objects: {len(entities_df)}")
    refined_entity_df = make_entity_df(
        remove_masked_entities((mask == 0) * 1.0, np.array(entities_df))
    )

    logger.debug(f"Removing entities using mask with shape {mask.shape}")
    result_list = []
    for i in range(len(refined_entity_df)):
        result_list.append(
            [
                refined_entity_df.iloc[i]["class_code"],
                refined_entity_df.iloc[i]["z"],
                refined_entity_df.iloc[i]["y"],
                refined_entity_df.iloc[i]["x"],
            ]
        )

    return result_list


@hug.get()
@save_metadata
def object_stats(
    src: DataURI, dst: DataURI,
    object_id: DataURI,
    feature_ids: DataURIList,
    stat_name: String,
) -> "OBJECTS":
    logger.debug(f"Calculating stats on objects: {object_id}")
    logger.debug(f"With features: {feature_ids}")

    src = DataModel.g.dataset_uri(ntpath.basename(feature_ids[0]), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_feature = DM.sources[0][:]
        # logger.debug(f"summary_stats {src_dataset[:]}")

    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    scale = ds_objects.get_metadata("scale")
    print(f"Scaling objects by: {scale}")

    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname)
    sel_start, sel_end = 0, len(entities_df)
    logger.info(f"Viewing entities {entities_fullname} from {sel_start} to {sel_end}")

    centers = np.array(
        [
            [
                np.int32(np.float32(entities_df.iloc[i]["z"]) * scale),
                np.int32(np.float32(entities_df.iloc[i]["x"]) * scale),
                np.int32(np.float32(entities_df.iloc[i]["y"]) * scale),
            ]
            for i in range(sel_start, sel_end)
        ]
    )
    box_size = 4

    print(f"Calculating statistic {stat_name} with box size of {box_size}")
    if stat_name == "0":
        stat_op = np.mean
        title = "Mean"
    elif stat_name == "1":
        stat_op = np.std
        title = "Standard Deviation"
    elif stat_name == "2":
        stat_op = np.var
        title = "Variance"
    point_features = [
        stat_op(
            ds_feature[
                c[0] - box_size : c[0] + box_size,
                c[1] - box_size : c[1] + box_size,
                c[2] - box_size : c[2] + box_size,
            ]
        )
        for c in centers
    ]

    plot_image = plot_to_image(point_features, title=title)

    return (point_features, encode_numpy(plot_image))


def analyze_detector_predictions(
    gt_entities,
    detected_entities,
    bvol_dim=(24, 24, 24),
    debug_verbose=True,
):
    print(f"Evaluating detections of shape {detected_entities.shape}")

    preds = centroid_to_bvol(detected_entities, bvol_dim=bvol_dim)
    targs = centroid_to_bvol(gt_entities, bvol_dim=bvol_dim)
    eval_result = eval_matches(preds, targs, debug_verbose=debug_verbose)
    return detected_entities





@hug.get()
def create(workspace: String, order: Int = 0):
    analyzer_type = __analyzer_names__[order]

    ds = ws.auto_create_dataset(
        workspace,
        analyzer_type,
        __analyzer_group__,
        __analyzer_dtype__,
        fill=__analyzer_fill__,
    )

    ds.set_attr("kind", analyzer_type)
    return dataset_repr(ds)


@hug.get()
@hug.local()
def existing(workspace: String, full: SmartBoolean = False, order: Int = 0):
    filter = __analyzer_names__[order]
    datasets = ws.existing_datasets(
        workspace, group=__analyzer_group__
    )  # , filter=filter)
    if full:
        return {
            "{}/{}".format(__analyzer_group__, k): dataset_repr(v)
            for k, v in datasets.items()
        }
    return {k: dataset_repr(v) for k, v in datasets.items()}


@hug.get()
def remove(workspace: String, analyzer_id: String):
    ws.delete_dataset(workspace, analyzer_id, group=__analyzer_group__)


@hug.get()
def rename(workspace: String, analyzer_id: String, new_name: String):
    ws.rename_dataset(workspace, analyzer_id, __analyzer_group__, new_name)


@hug.get()
def available():
    h = hug.API(__name__)
    all_features = []
    for name, method in h.http.routes[""].items():
        if name[1:] in ["available", "create", "existing", "remove", "rename", "group"]:
            continue
        name = name[1:]
        func = method["GET"][None].interface.spec
        desc = get_function_api(func)
        category = desc["returns"] or "Others"
        desc = dict(name=name, params=desc["params"], category=category)
        all_features.append(desc)
    return all_features
