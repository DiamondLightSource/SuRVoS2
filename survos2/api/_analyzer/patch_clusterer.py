import ntpath
from typing import List
import os
import matplotlib.patheffects as PathEffects
import numpy as np

import seaborn as sns

from loguru import logger
from matplotlib import offsetbox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from survos2.entity.entities import make_entity_bvol, make_entity_df
from survos2.entity.instance.detector import ObjectVsBgClassifier, ObjectVsBgCNNClassifier
from survos2.entity.patches import PatchWorkflow, make_patches, organize_entities, pad_vol
from survos2.entity.sampler import (
    offset_points,
    viz_bvols,
)

from survos2.frontend.components.entity import setup_entity_table

from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.utils import encode_numpy


from fastapi import APIRouter, Body, Query, Request

analyzer = APIRouter()





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
            [PathEffects.Stroke(linewidth=5, foreground="y"), PathEffects.Normal()]
        )


@analyzer.get("/object_analyzer", response_model=None)
def object_analyzer(
    workspace: str = Body(),
    object_id: str = Body(),
    feature_id: str = Body(),
    feature_extraction_method: str = Body(),
    embedding_method: str = Body(),
    dst: str = Body(),
    bvol_dim: List[int] = Body(),
    axis: int = Body(),
    embedding_params: dict = Body(),
    min_cluster_size: int = Body(),
    flipxy: bool = Body(),
) -> "OBJECTS":
    """Patch-based clustering tool. Provided locations are sampled in multiple locations in 2d, image
    features are calculated from the patches and the features are clustered and plotted. Allows selection of a
    set of clusters to return a subset of the input locations.

    Args:
        workspace (str): Workspace to use.
        object_id (str): Object URI to use as sample locations.
        feature_id (str): Feature image to constrain the point locations to.
        feature_extraction_method (str): Either Resnet CNN features or HOG features.
        embedding_method (str): TSNE or UMAP.
        dst (str): Destination URI (not used).
        bvol_dim (Union[int, List]): Dimension of bounding volume to use.
        axis (int): Which axis to sample 2D patches along.
        embedding_params (dict): Parameters for embedding method.
        min_cluster_size (int): HDBScan is used to cluster the embedded points. This is the minimum number of points in a cluster.
        flipxy (bool): Flip the XY coordinates of the input points.

    Returns:
        Tuple of (_, list, list of entities, np.ndarray, np.ndarray)
    """
    DataModel.g.current_workspace = workspace

    logger.debug(f"Calculating clustering on patches located at entities: {object_id}")
    from survos2.entity.cluster.cluster_plotting import cluster_scatter
    from survos2.entity.cluster.clusterer import PatchCluster, prepare_patches_for_clustering
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
    objects_path = ds_objects._path
    tabledata, entities_df = setup_entity_table(
        os.path.join(objects_path, entities_fullname), flipxy=flipxy
    )

    entities = np.array(make_entity_df(np.array(entities_df), flipxy=False))

    slice_idx = bvol_dim[0]
    logger.debug(f"Slicing bvol at index: {slice_idx} on axis {axis} for bvol of dim {bvol_dim}")

    feature_mat, selected_images = prepare_patches_for_clustering(
        ds_feature,
        entities,
        bvol_dim=bvol_dim,
        axis=axis,
        slice_idx=slice_idx,
        method=feature_extraction_method,
        flipxy=False,
    )
    num_clusters = 5

    logger.debug(f"Using feature matrix of size {feature_mat.shape}")
    selected_3channel = prepare_3channel(
        selected_images,
        patch_size=(selected_images[0].shape[0], selected_images[0].shape[1]),
    )
    patch_clusterer = PatchCluster(num_clusters, embedding_params["n_components"])
    patch_clusterer.prepare_data(feature_mat)
    patch_clusterer.fit()
    patch_clusterer.predict()

    embedding_params["n_components"] = 2
    if embedding_method == "TSNE":
        patch_clusterer.embed_TSNE(
            perplexity=embedding_params["perplexity"], n_iter=embedding_params["n_iter"]
        )
    else:
        # embedding_params.pop("context")
        patch_clusterer.embed_UMAP(params=embedding_params)

    patch_clusterer.density_cluster(min_cluster_size=min_cluster_size)
    labels = patch_clusterer.density_clusterer.labels_

    logger.debug(f"Metrics (DB, Sil): {patch_clusterer.cluster_metrics()}")
    preds = patch_clusterer.get_predictions()
    selected_images_arr = np.array(selected_3channel)
    logger.debug(f"Plotting {selected_images_arr.shape} patch images.")
    selected_images_arr = selected_images_arr[:, :, :, 1]

    # Plot to image
    fig = Figure()

    ax = fig.gca()
    ax.axis("off")
    fig.tight_layout(pad=0)

    clustered = labels >= 0
    standard_embedding = patch_clusterer.embedding
    standard_embedding = np.array(standard_embedding)

    if bvol_dim[0] < 32:
        skip_px = 1
    else:
        skip_px = 2

    labels = np.array(labels)
    labels = labels.tolist()
    entities = np.array(entities)
    entities = entities.tolist()

    return (
        None,
        labels,
        entities,
        encode_numpy(selected_images_arr),
        encode_numpy(standard_embedding),
    )




@analyzer.get("/patch_stats", response_model=None)
def patch_stats(
    src: str,
    dst: str,
    workspace: str,
    object_id: str,
    feature_id: str,
    stat_name: str,
    box_size: int,
) -> "OBJECTS":
    """Image statistics calculated on patches sampled from locations.

    Args:
        src (str): Source URI.
        dst (str): Destination URI.q
        workspace (str): Workspace to use.
        object_id (str): Object URI to use as source points.
        feature_id (str): Feature image to constrain the point locations to.
        stat_name (str): Which statistic to calculate.
        box_size (int): Side dimension of box to sample from.

    Returns:
        tuple: Tuple of (list of float, np.ndarray).
    """

    DataModel.g.current_workspace = workspace

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


def plot_to_image(arr, title="Histogram", vert_line_at=None):
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    # ax.axis("off")
    y, x, _ = ax.hist(arr, bins=16)
    fig.suptitle(title)

    if vert_line_at:
        ax.axvline(x=vert_line_at, ymin=0, ymax=y.max(), color="r")

    canvas.draw()
    plot_image = np.asarray(canvas.buffer_rgba())
    return plot_image






@analyzer.get("/binary_classifier", response_model=None)
def binary_classifier(
    workspace: str = Body(),
    feature_id: str = Body(),
    proposal_id: str = Body(),
    mask_id: str = Body(),
    object_id: str = Body(),
    background_id: str = Body(),
    dst: str = Body(),
    bvol_dim: List[int] = Body(),
    score_thresh: float = Body(),
    area_min: int = Body(),
    area_max: int = Body(),
    model_fullname: str = Body(),
):
    """
    Takes proposal seg and a set of gt entities

    """
    DataModel.g.current_workspace = workspace
    # load the proposal seg
    # load the gt entities
    # make the gt training data patches
    # extract the features of the classifier
    # train the classifier
    # find the components in the proposal seg given the filter components settings and apply mask
    # run the classifier and generate detections
    # be able to load the obj or the background as entities

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
    entities_arr[:, 3] = np.array([[1] * len(entities_arr)])
    entities = np.array(make_entity_df(entities_arr, flipxy=False))

    # get background entities
    src = DataModel.g.dataset_uri(ntpath.basename(background_id), group="objects")
    logger.debug(f"Getting background entities {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")

    tabledata, entities_df = setup_entity_table(entities_fullname, flipxy=False)
    entities_arr = np.array(entities_df)
    entities_arr[:, 3] = np.array([[0] * len(entities_arr)])
    bg_entities = np.array(make_entity_df(entities_arr, flipxy=False))

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

    aug_pts = np.concatenate((entities, bg_entities))
    augmented_entities = aug_pts
    # augmented_entities = make_augmented_entities(aug_pts)
    logger.debug(f"Produced augmented entities of shape {augmented_entities.shape}")

    combined_clustered_pts, classwise_entities = organize_entities(
        proposal_segmentation, augmented_entities, entity_meta, plot_all=False
    )
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

    if model_fullname != "None":
        logger.debug(f"Loading model for fpn features: {model_fullname}")
    else:
        model_fullname = None

    classifier_type = "classical"
    if classifier_type == "classical":
        instdet = ObjectVsBgClassifier(
            wf,
            augmented_entities,
            proposal_segmentation,
            feature,
            bg_mask,
            padding=bvol_dim,
            area_min=area_min,
            area_max=area_max,
            plot_debug=True,
        )
        instdet.reset_bg_mask()
        instdet.make_component_tables()  # find connected components in proposal seg
        instdet.prepare_detector_data()
        instdet.train_validate_model(model_file=None)
        instdet.predict(
            instdet.class1_dets, score_thresh=score_thresh, model_file=model_fullname, offset=False
        )
        # instdet.analyze_result(instdet.class1_gold_entities, instdet.class1_dets, instdet.detections)
        logger.debug("\nProduced foreground detections: \n")
        logger.debug(instdet.detections)
        # result_entities = instdet.detections
        result_entities = offset_points(
            instdet.detections, -3 * np.array(padding)
        )  # offset_points(result_entities, np.array((-16, -16, -16)))
    elif classifier_type == "cnn":
        instdet = ObjectVsBgCNNClassifier(
            wf,
            augmented_entities,
            proposal_segmentation,
            feature,
            bg_mask,
            padding=bvol_dim,
            area_min=area_min,
            area_max=area_max,
            plot_debug=True,
        )
        instdet.reset_bg_mask()
        instdet.make_component_tables()
        instdet.prepare_detector_data()
        instdet.train_validate_model(model_file=model_fullname, workspace=workspace)
        instdet.predict(
            instdet.class1_dets, score_thresh=score_thresh, model_file=model_fullname, offset=False
        )
        result_entities = offset_points(instdet.class1_dets, -2 * np.array(padding))

    result_entities = np.array(result_entities)
    result_entities = result_entities.tolist()

    return result_entities