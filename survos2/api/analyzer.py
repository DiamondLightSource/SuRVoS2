import logging
import ntpath
import os.path as op

import dask.array as da
import hug
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
from loguru import logger
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
    Int,
    IntList,
    SmartBoolean,
    String,
)

from survos2.api.utils import dataset_repr, get_function_api, save_metadata
from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.frontend.components.entity import setup_entity_table
from survos2.frontend.nb_utils import summary_stats
from survos2.improc import map_blocks
from survos2.improc.utils import DatasetManager
from survos2.io import dataset_from_uri
from survos2.model import DataModel
from survos2.utils import encode_numpy

__analyzer_fill__ = 0
__analyzer_dtype__ = "uint32"
__analyzer_group__ = "analyzer"
__analyzer_names__ = ["simple_stats", "simple_stats2"]


from survos2.api.utils import dataset_repr, get_function_api, save_metadata




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


@hug.get()
def object_analyzer(
    workspace: String, object_id: DataURI, feature_ids: DataURIList, dst: DataURI,
) -> "OBJECTS":
    logger.debug(f"Calculating clustering on patches located at entities: {object_id}")
    logger.debug(f"With features: {feature_ids}")
    from survos2.entity.cluster.cluster_plotting import cluster_scatter
    from survos2.entity.cluster.clusterer import (
        PatchCluster,
        prepare_patches_for_clustering,
    )
    from survos2.entity.cluster.cnn_features import prepare_3channel

    # get features
    src = DataModel.g.dataset_uri(ntpath.basename(feature_ids[0]), group="features")
    logger.debug(f"Getting features {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_feature = DM.sources[0][:]
        # logger.debug(f"summary_stats {src_dataset[:]}")

    # get entities
    src = DataModel.g.dataset_uri(ntpath.basename(object_id), group="objects")
    logger.debug(f"Getting objects {src}")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds_objects = DM.sources[0]
    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname)

    entities = np.array(entities_df)
    feature_mat, selected_images = prepare_patches_for_clustering(
        ds_feature, entities, padding=(32, 32, 32)
    )

    n_components = 10
    num_clusters = 15
    perplexity = 100
    n_iter = 1000
    skip_px = 2

    print(f"Using feature matrix of size {feature_mat.shape}")
    selected_3channel = prepare_3channel(
        selected_images,
        patch_size=(selected_images[0].shape[0], selected_images[0].shape[1]),
    )

    patch_clusterer = PatchCluster(num_clusters, n_components)
    patch_clusterer.prepare_data(feature_mat)
    patch_clusterer.fit()
    patch_clusterer.predict()
    patch_clusterer.embed_TSNE(perplexity=perplexity, n_iter=n_iter)

    print(f"Metrics (DB, Sil, C-H): {patch_clusterer.cluster_metrics()}")
    preds = patch_clusterer.get_predictions()
    # cluster_scatter(patch_clusterer.embedding, preds)
    selected_images_arr = np.array(selected_3channel)
    print(f"Plotting {selected_images_arr.shape} patch images.")
    selected_images_arr = selected_images_arr[:, :, :, 1]

    # fig, ax = plt.subplots(figsize=(17, 17))
    # plt.title(
    #    "Windows around a click clustered using deep features and embedded in 2d using T-SNE".format()
    # )

    # Plot to image
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.axis("off")
    # fig.tight_layout(pad=0)
    # ax.margins(0)
    # ax.imshow(src_dataset[:][100,0:100,0:100])

    plot_clustered_img(
        patch_clusterer.embedding,
        preds,
        ax=ax,
        images=selected_images_arr[:, ::skip_px, ::skip_px],
    )

    fig.suptitle("Test")
    canvas.draw()
    plot_image = np.asarray(canvas.buffer_rgba())

    return encode_numpy(plot_image)


@hug.get()
def simple_stats(workspace: String, feature_ids: DataURIList, dst: DataURI,) -> "IMAGE":
    logger.debug(f"Calculating stats on features: {feature_ids}")
    src = DataModel.g.dataset_uri(ntpath.basename(feature_ids[0]), group="features")
    logger.debug(f"Getting features {src}")

    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        logger.debug(f"summary_stats {src_dataset[:]}")

    def plot_to_image(img):
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.axis("off")
        # fig.tight_layout(pad=0)
        # ax.margins(0)
        # axax.imshow(src_dataset[:][100, 0:100, 0:100])
        ax.hist(img, bins=16)
        fig.suptitle("Test histogram")
        canvas.draw()
        plot_image = np.asarray(canvas.buffer_rgba())
        return plot_image

    plot_image = plot_to_image(src_dataset[:][10, 100:300, 100:300])

    return encode_numpy(plot_image)


@hug.get()
def simple_stats2(
    workspace: String, object_id: DataURI, feature_ids: DataURIList, dst: DataURI,
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

    entities_fullname = ds_objects.get_metadata("fullname")
    tabledata, entities_df = setup_entity_table(entities_fullname)
    sel_start, sel_end = 0, len(entities_df)
    logger.info(f"Viewing entities {entities_fullname} from {sel_start} to {sel_end}")
    scale = 1.0

    centers = np.array(
        [
            [
                np.int(np.float(entities_df.iloc[i]["z"]) * scale),
                np.int(np.float(entities_df.iloc[i]["x"]) * scale),
                np.int(np.float(entities_df.iloc[i]["y"]) * scale),
            ]
            for i in range(sel_start, sel_end)
        ]
    )
    print(centers)

    point_features = [(ds_feature[c[0], c[2], c[1]]) for c in centers]
    return point_features


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
    datasets = ws.existing_datasets(workspace, group=__analyzer_group__) #, filter=filter)
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
