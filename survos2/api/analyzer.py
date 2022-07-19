from cProfile import label
import logging
import ntpath
import os.path as op
from posixpath import split
from xmlrpc.client import Boolean
import pandas as pd
import dask.array as da
from numba.core.types.scalars import Integer
import hug
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
from loguru import logger
import tempfile
from functools import lru_cache
from matplotlib import offsetbox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import torch
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
from survos2.entity.sampler import centroid_to_bvol, offset_points, viz_bvols
from survos2.entity.utils import get_surface
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
from survos2.entity.patches import PatchWorkflow, organize_entities, make_patches
from survos2.entity.anno.pseudo import generate_augmented_entities, make_pseudomasks, make_anno
from survos2.entity.patches import pad_vol


from survos2.entity.cluster.cluster_plotting import cluster_scatter, plot_clustered_img

from survos2.entity.sampler import (
    generate_random_points_in_volume,
)

def pass_through(x):
    return x

__analyzer_fill__ = 0
__analyzer_dtype__ = "uint32"
__analyzer_group__ = "analyzer"
__analyzer_names__ = [
    "find_connected_components",
    "patch_stats",
    "object_detection_stats",
    "segmentation_stats",
    "label_analyzer",
    "label_splitter",
    "binary_image_stats",
    "spatial_clustering",
    "remove_masked_objects",
    "object_analyzer",
    "binary_classifier",
    "point_generator",
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
    "Segmented Surface Area",
    "Segmented Volume",
    "Segmented Sphericty"
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
    "seg_sphericity"
]


def window_to_bb(w):

    return 0, 0, (w[0].start + w[0].stop)//2, (w[1].start+w[1].stop)//2, (w[2].start+w[2].stop) // 2, w[0].start, w[0].stop, w[1].start, w[1].stop, w[2].start, w[2].stop

def windows_to_bvols(windows):
    bbs = []
    for w in windows:
        bbs.append(window_to_bb(w))
    return bbs

def sample_windows(img_volume, win):
    z_st, z_end,  y_st, y_end, x_st, x_end = win
    z_st = int(z_st)
    z_end = int(z_end)
    y_st = int(y_st)
    y_end = int(y_end)
    x_st = int(x_st)
    x_end = int(x_end)
    img = img_volume[z_st:z_end, y_st:y_end, x_st:x_end]
    return img

def detect_blobs(
    padded_proposal,
    area_min=0,
    area_max=1e12,
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




@hug.get()
def label_analyzer(
    src: DataURI, 
    dst: DataURI,
    workspace: String,
    mode: String,
    pipelines_id: DataURI,
    analyzers_id: DataURI,
    annotations_id: DataURI,
    feature_id: DataURI,
    split_ops : dict,
    background_label : Int
) -> "SEGMENTATION":

    return label_splitter(src, dst, workspace, mode, pipelines_id, analyzers_id, annotations_id, feature_id, split_ops, background_label)

@hug.get()
def label_splitter(
    src: DataURI, 
    dst: DataURI,
    workspace: String,
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
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif mode == '2':
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif mode == '3':
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
    #objects = new_labels
    num_objects = total_labels
    logger.debug(f"Number of objects {num_objects}")
    objlabels = np.arange(1, num_objects + 1)

    obj_windows = measure.find_objects(objs)
    feature = []
    depth = []
    height = []
    width = []
    
    #seg_crops = []
    feature_surface_area = []
    feature_volume = []
    feature_sphericity = []
    for w in obj_windows:
        feature.append(
            (w[0].stop - w[0].start)
            * (w[1].stop - w[1].start)
            * (w[2].stop - w[2].start)
        )
        depth.append(w[0].stop - w[0].start)
        height.append(w[1].stop - w[1].start)
        width.append(w[2].stop - w[2].start)

        if (depth[-1] > 2) & (height[-1] > 2) & (width[-1] > 2):    
            #seg_crops.append(seg[w[0].start:w[0].stop,w[1].start:w[1].stop,w[2].start:w[2].stop])
            #plt.figure()
            #plt.imshow(seg_crops[-1][(w[0].stop-w[0].start) // 2,:])  
            try:  
                _, surface_area, volume, sphericity = get_surface(seg[w[0].start:w[0].stop,w[1].start:w[1].stop,w[2].start:w[2].stop])
                feature_surface_area.append(surface_area)
                feature_volume.append(volume) 
                feature_sphericity.append(sphericity)
            except:
                d, h, w = depth[-1],height[-1],width[-1]
                feature_surface_area.append((d*h) + 4 * (h*w) + (d*w))
                feature_volume.append(d*h*w) 
                sphericity_of_cube = 0.523599  # rounded to 6 fig
                feature_sphericity.append(sphericity_of_cube)
        else:
            d, h, w = depth[-1],height[-1],width[-1]
            feature_surface_area.append((d*h) + 4 * (h*w) + (d*w))
            feature_volume.append(d*h*w) 
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
            "seg_surface_area" : feature_surface_area,
            "seg_volume" : feature_volume,
            "seg_sphericity" : feature_sphericity
        }
    )

    sel_start, sel_end = 0, len(features_df)
    features_array = np.array(
        [
            [
                np.int32(np.float32(features_df.iloc[i]["z"])),
                np.int32(np.float32(features_df.iloc[i]["y"])),
                np.int32(np.float32(features_df.iloc[i]["x"])),
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
                np.float32(np.float32(features_df.iloc[i]["seg_sphericity"]))
            ]
            for i in range(sel_start, sel_end)
        ]
    )

    logger.debug(split_ops)
    rules = []
    calculate = False
    for k in split_ops.keys():
        if k != 'context':
            split_op_card = split_ops[k]
            split_feature_index = int(split_op_card["split_feature_index"])
            split_op = int(split_op_card["split_op"])
            split_threshold = float(split_op_card["split_threshold"])
            
            if int(split_op) > 0:
                calculate = True
                s = int(split_op) - 1  # split_op starts at 1
                feature_names = ["z", "y", "x", "Sum", "Mean", "Std", "Var", "bb_vol", "bb_vol_log10", "bb_vol_depth", "bb_vol_depth","bb_vol_height", "bb_vol_width", "ori_vol", "ori_vol_log10", "ori_vol_depth", "ori_vol_depth","ori_vol_height", "ori_vol_width", "seg_surface_area", "seg_volume", "seg_sphericity"]
                feature_index = int(
                    split_feature_index
                )  # feature_names.index(split_feature_index)
                rules.append((int(feature_index), s, split_threshold))
                logger.debug(f"Adding split rule: {split_feature_index} {split_op} {split_threshold}")
            
    if calculate:
        masked_out, result_features = apply_rules(
            features_array, -1, rules, np.array(objlabels), num_objects
        )
        logger.debug(f"Masking out: {masked_out}")
        bg_label = max(np.unique(new_labels))
        logger.debug(f"Masking out bg label {bg_label}")
        for i, l in enumerate(masked_out):
            if l != -1:
                new_labels[new_labels == l] = 0

        new_labels[new_labels == bg_label] = 0
        new_labels = (new_labels > 0) * 1.0
    else:
        result_features = features_array    
    map_blocks(pass_through, new_labels, out=dst, normalize=False)


    bvols = windows_to_bvols(obj_windows)

    return result_features, features_array,bvols 

def apply_rules(
    features: np.ndarray, label: int, rules: tuple, out: np.ndarray, num_objects: int
):
    logger.debug("Applying rules")
    mask = np.ones(num_objects, dtype=np.bool)

    for f, s, t in rules:
        if s == 0:
            np.logical_and(mask, features[:, f] > t, out=mask)
            result_features = features[features[:, f] > t]
        else:
            np.logical_and(mask, features[:, f] < t, out=mask)
            result_features = features[features[:, f] < t]

    out[mask] = label
    return out, result_features


# def plot_clustered_img(
#     proj,
#     colors,
#     images=None,
#     ax=None,
#     thumb_frac=0.02,
#     cmap="gray",
#     title="Clustered Images",
# ):

#     num_classes = len(np.unique(colors))
#     palette = np.array(sns.color_palette("hls", num_classes))
#     # ax = ax or plt.gca()
#     if images is not None:
#         min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
#         shown_images = np.array([2 * proj.max(0)])
#         for i in range(proj.shape[0]):
#             dist = np.sum((proj[i] - shown_images) ** 2, 1)
#             if np.min(dist) < min_dist_2 / 5:
#                 continue
#             shown_images = np.vstack([shown_images, proj[i]])
#             imagebox = offsetbox.AnnotationBbox(
#                 offsetbox.OffsetImage(images[i], cmap=cmap, zorder=1), proj[i]
#             )
#             ax.add_artist(imagebox)

#     ax.scatter(proj[:, 0], proj[:, 1], lw=0, s=40, zorder=200)


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
    src: DataURI, 
    dst: DataURI, 
    workspace: String, 
    label_index: Int, 
    area_min : Int, 
    area_max: Int,
    mode: String,
    pipelines_id: DataURI,
    analyzers_id: DataURI,
    annotations_id: DataURI,

) -> "SEGMENTATION":
    logger.debug(f"Finding connected components on segmentation: {pipelines_id}")
    
    
    if mode == '1':
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif mode == '2':
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif mode == '3':
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")
    
    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        seg = DM.sources[0][:]
        logger.debug(f"src_dataset shape {seg[:].shape}")

    src_dataset_arr = seg.astype(np.uint32) & 15
    
    # print(f"{DataModel.g.current_workspace}")
    # src = DataModel.g.dataset_uri(pipelines_id, group="pipelines")
    # print(src)
    # with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
    #     src_dataset_arr = DM.sources[0][:]
    #     logger.debug(f"src_dataset shape {src_dataset_arr[:].shape}")

    single_label_level = (src_dataset_arr == label_index) * 1.0

    print(single_label_level.shape)
    bbs_tables, selected_entities = detect_blobs(single_label_level)
    print(bbs_tables)
    print(selected_entities)

    result_list = []
    for i in range(len(bbs_tables[0])):
        if (bbs_tables[0].iloc[i]["area"] > area_min) & (bbs_tables[0].iloc[i]["area"] < area_max):
            result_list.append(
                [
                    bbs_tables[0].iloc[i]["z"],
                    bbs_tables[0].iloc[i]["x"],
                    bbs_tables[0].iloc[i]["y"],
                    bbs_tables[0].iloc[i]["area"],
                ]
            )

    map_blocks(pass_through, single_label_level, out=dst, normalize=False)
    
    print(result_list)

    return result_list

@hug.get()
def segmentation_stats(
    src: DataURI, 
    dst: DataURI,
    modeA: String,
    modeB: String,
    workspace: String, 
    pipelines_id_A: DataURI,
    analyzers_id_A: DataURI,
    annotations_id_A: DataURI,
    pipelines_id_B: DataURI,
    analyzers_id_B: DataURI,
    annotations_id_B: DataURI,
    label_index_A: Int, 
    label_index_B: Int, 
) -> "IMAGE":

    if modeA == '1':
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id_A), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif modeA == '2':
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id_A), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif modeA == '3':
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id_A), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")
    
    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        seg = DM.sources[0][:]
        logger.debug(f"src_dataset shape {seg[:].shape}")

    src_dataset_arr_A = seg.astype(np.uint32) & 15
    
    if modeB == '1':
        src = DataModel.g.dataset_uri(ntpath.basename(pipelines_id_B), group="pipelines")
        logger.debug(f"Analyzer calc on pipeline src {src}")
    elif modeB == '2':
        src = DataModel.g.dataset_uri(ntpath.basename(analyzers_id_B), group="analyzer")
        logger.debug(f"Analyzer calc on analyzer src {src}")
    elif modeB == '3':
        src = DataModel.g.dataset_uri(ntpath.basename(annotations_id_B), group="annotations")
        logger.debug(f"Analyzer calc on annotation src {src}")
    
    with DatasetManager(src, out=None, dtype="int32", fillvalue=0) as DM:
        segB = DM.sources[0][:]
        logger.debug(f"src_dataset shape {segB[:].shape}")

    src_dataset_arr_B = segB.astype(np.uint32) & 15

    single_label_level_A = (src_dataset_arr_A == label_index_A) * 1.0
    single_label_level_B = (src_dataset_arr_B == label_index_B) * 1.0
    
    print(f"Count: {np.sum(single_label_level_A * single_label_level_B)}")

    #from survos2.entity.trainer import score_dice
    #print(f"Dice loss {score_dice(single_label_level_A, single_label_level_B)}")

    from torchmetrics import JaccardIndex, Dice
    dice = Dice(average='micro')
    jaccard = JaccardIndex(num_classes=2)
        
    A_t = torch.IntTensor(single_label_level_A)
    B_t = torch.IntTensor(single_label_level_B)
    
    print(f"Jaccard (IOU): {jaccard(A_t, B_t)}")
    print(f"Dice (torchmetrics): {dice(A_t, B_t)}")

    
    dice_score = dice(A_t, B_t)
    iou_score = jaccard(A_t, B_t)
    
    result_list = [float(dice_score.numpy()), float(iou_score.numpy())]

    return result_list



@hug.get()
def binary_image_stats(
    src: DataURI, 
    dst: DataURI,
    workspace: String, 
    feature_id: DataURI, 
    threshold: Float = 0.5,
    area_min: Int = 0,
    area_max: Int = 1e12
) -> "IMAGE":
    logger.debug(f"Calculating stats on feature: {feature_id}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset_arr = DM.sources[0][:]

    src_thresh = (src_dataset_arr > threshold) * 1.0
    bbs_tables, selected_entities = detect_blobs(src_thresh, area_min=area_min, area_max=area_max)
    print(bbs_tables)
    print(selected_entities)
    result_list = []

    for i in range(len(bbs_tables[0])):
        if (bbs_tables[0].iloc[i]["area"] < area_max) & (bbs_tables[0].iloc[i]["area"] > area_min):
            result_list.append(
                [
                    bbs_tables[0].iloc[i]["z"],
                    bbs_tables[0].iloc[i]["x"],
                    bbs_tables[0].iloc[i]["y"],
                    bbs_tables[0].iloc[i]["area"],
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
def remove_masked_objects(
    src: DataURI, dst: DataURI,
    feature_id: DataURI,
    object_id: DataURI,
    invert : Boolean,
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

    if invert:
        mask = 1.0 - mask

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
    logger.debug(f"Total number of entities after masking {len(refined_entity_df)}")
    return result_list


@hug.get()
def patch_stats(
    src: DataURI, 
    dst: DataURI,
    workspace: String,
    object_id: DataURI,
    feature_id: DataURI,
    stat_name: String,
    box_size: Int 
) -> "OBJECTS":
    logger.debug(f"Calculating stats on cubical patches: {object_id}")
    
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
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
    elif stat_name == "3":
        stat_op = np.median
        title = "Median"
    elif stat_name == "4":
        stat_op = np.sum
        title = "Sum"
    
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

    return point_features, encode_numpy(plot_image)


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

    ax.scatter(proj[:, 0], proj[:, 1], lw=0, s=2, zorder=200)

    ax.axis("off")
    ax.axis("tight")
    # ax.set_title(
    #     "Plot of k-means clustering of small click windows using ResNet-18 Features"
    # )

    for i in range(num_classes):
        xtext, ytext = np.median(proj[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=18)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="y"), PathEffects.Normal()])
    
@hug.get()
def object_analyzer(
    workspace: String, 
    object_id: DataURI, 
    feature_id: DataURI, 
    feature_extraction_method : String,
    embedding_method: String,
    dst: DataURI,
    bvol_dim: IntOrVector,
    axis: Int,
    embedding_params : dict,
    min_cluster_size : Int,
    flipxy : SmartBoolean,
    
) -> "OBJECTS":
    logger.debug(f"Calculating clustering on patches located at entities: {object_id}")
    from survos2.entity.cluster.cluster_plotting import cluster_scatter
    from survos2.entity.cluster.clusterer import (
        PatchCluster,
        prepare_patches_for_clustering,
    )
    from survos2.entity.cluster.cnn_features import prepare_3channel

    # get features
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_feature = DM.sources[0][:]
        logger.debug(f"Feature shape {ds_feature.shape}")

    # get entities
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=flipxy)

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=False))

    slice_idx = bvol_dim[0]
    print(f"Slicing bvol at index: {slice_idx} on axis {axis} for bvol of dim {bvol_dim}")
    
    feature_mat, selected_images = prepare_patches_for_clustering(
        ds_feature, entities, bvol_dim=bvol_dim, axis=axis, slice_idx=slice_idx, method=feature_extraction_method, flipxy=False
    )
    num_clusters = 5

    print(f"Using feature matrix of size {feature_mat.shape}")
    selected_3channel = prepare_3channel(
        selected_images,
        patch_size=(selected_images[0].shape[0], selected_images[0].shape[1]),
    )
    patch_clusterer = PatchCluster(num_clusters, embedding_params["n_components"])
    patch_clusterer.prepare_data(feature_mat)
    patch_clusterer.fit()
    patch_clusterer.predict()

    embedding_params["n_components"]= 2
    if embedding_method=='TSNE':
        patch_clusterer.embed_TSNE(perplexity=embedding_params["perplexity"], n_iter=embedding_params["n_iter"])
    else:
        # params={'n_neighbors':10,
        #         'min_dist':0.3,
        #         'n_components':2,
        #         'metric':'correlation', 
        #         'densmap':True,  
        #         'dens_lambda':3.0
        # }
        embedding_params.pop("context")
        print(embedding_params)
        patch_clusterer.embed_UMAP(params=embedding_params)

    patch_clusterer.density_cluster(min_cluster_size=min_cluster_size)
    labels = patch_clusterer.density_clusterer.labels_

    print(f"Metrics (DB, Sil): {patch_clusterer.cluster_metrics()}")
    preds = patch_clusterer.get_predictions()
    selected_images_arr = np.array(selected_3channel)
    print(f"Plotting {selected_images_arr.shape} patch images.")
    selected_images_arr = selected_images_arr[:, :, :, 1]

    # Plot to image
    fig = Figure()
    
    ax = fig.gca()
    ax.axis("off")
    fig.tight_layout(pad=0)
    #ax.margins(0)
    # ax.set_xticks([], [])
    # ax.set_yticks([], [])

    clustered = (labels >= 0)
    standard_embedding = patch_clusterer.embedding
    print(standard_embedding)
    standard_embedding = np.array(standard_embedding)
    # ax.scatter(standard_embedding[~clustered, 0],
    #             standard_embedding[~clustered, 1],
    #             color=(0.5, 0.5, 0.5),
    #             s=1,
    #             alpha=0.5)
    # ax.scatter(standard_embedding[clustered, 0],
    #             standard_embedding[clustered, 1],
    #             c=labels[clustered],
    #             s=1,
    #             cmap='Spectral');
    if bvol_dim[0] < 32:
        skip_px = 1
    else:
        skip_px = 2

    # plot_clustered_img(
    #     standard_embedding,
    #     labels,
    #     ax=ax,
    #     images=selected_images_arr[:, ::skip_px, ::skip_px],
    # )
    # #fig.suptitle("Clustering")
    # from survos2.entity.cluster.cluster_plotting import image_grid, image_grids
    # figs = image_grids(selected_images_arr, labels)
    # canvas = FigureCanvasAgg(figs[0])
    # canvas.draw()
    # plot_image = np.asarray(canvas.buffer_rgba())
    return None, labels, entities, encode_numpy(selected_images_arr), encode_numpy(standard_embedding)




@hug.get()
def binary_classifier(
    workspace: String,
    feature_id: DataURI, 
    proposal_id: DataURI,
    mask_id : DataURI, 
    object_id: DataURI,
    background_id: DataURI,
    dst: DataURI,
    bvol_dim: IntOrVector,
    score_thresh : Float,
    area_min : Int,
    area_max : Int,
    model_fullname : String):
    """
    Takes proposal seg and a set of gt entities
    
    """
    # load the proposal seg
    # load the gt entities
    # make the gt training data patches
    # extract the features of the classifier
    # train the classifier
    # find the components in the proposal seg given the filter components settings and apply mask
    # run the classifier and generate detections
    # be able to load the obj or the background as entities

    from survos2.entity.instance.det import make_augmented_entities
    from survos2.entity.instance.detector import ObjectVsBgClassifier, ObjectVsBgCNNClassifier

    # get image feature
    src = DataModel.g.dataset_uri(ntpath.basename(feature_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        feature = DM.sources[0][:]
        logger.debug(f"Feature shape {feature.shape}")

    # get feature for proposal segmentation
    src = DataModel.g.dataset_uri(ntpath.basename(proposal_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        proposal_segmentation = DM.sources[0][:]
        logger.debug(f"Feature shape {proposal_segmentation.shape}")
    
    # get feature for background mask
    src = DataModel.g.dataset_uri(ntpath.basename(mask_id), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        bg_mask = DM.sources[0][:]
        logger.debug(f"Feature shape {bg_mask.shape}")

    # get object entities
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=False)
    entities_arr = np.array(entities_df)
    entities_arr[:,3] = np.array([[1] * len(entities_arr)])
    entities = np.array(make_entity_df(entities_arr, flipxy=False))
    print(entities)

    # get background entities
    src = DataModel.g.dataset_uri(ntpath.basename(background_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=False)
    entities_arr = np.array(entities_df)
    entities_arr[:,3] = np.array([[0] * len(entities_arr)])
    bg_entities = np.array(make_entity_df(entities_arr, flipxy=False))
    print(bg_entities)

    padding = np.array(bvol_dim) // 2
    feature = pad_vol(feature, padding)
    proposal_segmentation = pad_vol(proposal_segmentation, padding)
    bg_mask = pad_vol(bg_mask, padding)
    entities = offset_points(entities, padding)
    bg_entities = offset_points(bg_entities, padding)

    entity_meta = {
        "0": {
            "name": "class1",
            "size": np.array(16),
            "core_radius": np.array((7, 7, 7)),
        },
    }

    aug_pts = np.concatenate((entities,bg_entities))
    augmented_entities = aug_pts
    #augmented_entities = make_augmented_entities(aug_pts)
    logger.debug(f"Produced augmented entities of shape {augmented_entities.shape}")

    combined_clustered_pts, classwise_entities = organize_entities(
        proposal_segmentation, augmented_entities, entity_meta, plot_all=False)
    wparams = {}
    wparams["entities_offset"] = (0, 0, 0)

    wf = PatchWorkflow(
        [feature, proposal_segmentation],
        combined_clustered_pts,
        classwise_entities,
        bg_mask,
        wparams,
        combined_clustered_pts,
    )   
    
    # gt_entities, random_bg_entities =  generate_augmented_entities(wf, 
    #                                                                 generate_random_bg_entities=True,
    #                                                                 num_before_masking=num_before_masking,
    #                                                                 stratified_selection=True,
    #                                                                 class_proportion = {0:1, 1: 1, 2: 1.0, 5:1})
    
    #print(entities.shape, random_bg_entities.shape)    
    
    if model_fullname != 'None':
        logger.debug(f"Loading model for fpn features: {model_fullname}")
    else:
        model_fullname = None

    instdet = ObjectVsBgCNNClassifier(wf, 
                           augmented_entities,
                           proposal_segmentation, 
                           feature, 
                           bg_mask,
                           padding=bvol_dim,
                           area_min=area_min,
                           area_max=area_max,
                           plot_debug=True)

    classifier_type='cnn'
    if classifier_type=='classical':
        instdet.reset_bg_mask()
        instdet.make_component_tables()
        instdet.prepare_detector_data()
        instdet.train_validate_model(model_file=model_fullname)
        instdet.predict(instdet.class1_dets, score_thresh=score_thresh, model_file=model_fullname, offset=False)
        instdet.analyze_result(instdet.class1_gold_entities, instdet.class1_dets, instdet.detections)
        logger.debug("\nProduced foreground detections: \n")
        logger.debug(instdet.class1_dets)
        result_entities = offset_points(result_entities, np.array((-16,-16,-16)))
    elif classifier_type=='cnn':
        instdet.reset_bg_mask()
        instdet.make_component_tables()
        instdet.prepare_detector_data()
        instdet.train_validate_model(model_file=model_fullname, workspace=workspace)
        instdet.predict(instdet.class1_dets, score_thresh=score_thresh, model_file=model_fullname, offset=False)
        result_entities = offset_points(instdet.class1_dets, -2 * np.array(padding))    
        #instdet.analyze_result(instdet.class1_gold_entities, instdet.class1_dets, instdet.detections)
        #logger.debug("\nProduced foreground detections: \n")
        #logger.debug(instdet.class1_dets)
        
    # offset the points back to remove the effect of padding
    
    #result_entities = np.array(make_entity_df(result_entities, flipxy=False))
    return result_entities



@hug.get()
def point_generator(
    bg_mask_id : DataURI,
    num_before_masking : Int,
):
    mask_name = ntpath.basename(bg_mask_id)
    
    if mask_name == 'None':
        src = DataModel.g.dataset_uri('001_raw', group="features")
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            bg_mask = np.zeros_like(DM.sources[0][:])
            logger.debug(f"Feature shape {bg_mask.shape}")

    else:
        # get feature for background mask
        src = DataModel.g.dataset_uri(ntpath.basename(bg_mask_id), group="features")
    
        logger.debug(f"Getting features {src}")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            bg_mask = DM.sources[0][:]
            logger.debug(f"Feature shape {bg_mask.shape}")

    random_entities = generate_random_points_in_volume(
        bg_mask, num_before_masking, border=(0, 0, 0)
    ).astype(np.uint32)

    if mask_name != 'None':
        from survos2.entity.utils import remove_masked_entities
        print(
            f"Before masking random entities generated of shape {random_entities.shape}"
        )
        result_entities = remove_masked_entities(bg_mask, random_entities)
        print(f"After masking: {random_entities.shape}")
    else:
        result_entities = random_entities
    result_entities[:, 3] = np.array([6] * len(result_entities))
    result_entities = np.array(make_entity_df(result_entities, flipxy=True))

    return result_entities


    

# def analyze_detector_predictions(
#     gt_entities,
#     detected_entities,
#     bvol_dim=(24, 24, 24),
#     debug_verbose=True,
# ):
#     print(f"Evaluating detections of shape {detected_entities.shape}")

#     preds = centroid_to_bvol(detected_entities, bvol_dim=bvol_dim)
#     targs = centroid_to_bvol(gt_entities, bvol_dim=bvol_dim)
#     eval_result = eval_matches(preds, targs, debug_verbose=debug_verbose)
#     return detected_entities



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



@lru_cache(maxsize=2)
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



