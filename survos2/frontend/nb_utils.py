"""
Utilities for notebooks, e.g. for popping up napari to view different types of data.    

"""
import os
import json
import sys
import numpy as np
import napari

import matplotlib
from matplotlib import pyplot as plt
from survos2.frontend.utils import quick_norm
import seaborn as sns

from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2 import survos


def view_dataset(dataset_name, group, z=None):
    src = DataModel.g.dataset_uri(dataset_name, group=group)

    with DatasetManager(src, out=None, dtype="float32") as DM:
        src_dataset = DM.sources[0]
        src_arr = src_dataset[:]
    if z:
        plt.figure()
        plt.imshow(src_arr[z, :])

    return src_arr


def add_anno(anno_vol, new_name, workspace_name):
    result = survos.run_command(
        "annotations",
        "add_level",
        uri=None,
        workspace=workspace_name,
    )
    new_anno_id = result[0]["id"]

    src = DataModel.g.dataset_uri(new_anno_id, group="annotations")

    with DatasetManager(src, out=src, dtype="int32", fillvalue=0) as DM:
        out_dataset = DM.out
        out_dataset[:] = anno_vol

    print(f"Created new annotation with id: {new_anno_id}")


def add_feature(feature_vol, new_name, workspace_name):
    result = survos.run_command(
        "features", "create", uri=None, workspace=workspace_name, feature_type="raw"
    )
    new_feature_id = result[0]["id"]
    result = survos.run_command(
        "features",
        "rename",
        uri=None,
        feature_id=new_feature_id,
        new_name=new_name,
        workspace=workspace_name,
    )

    src = DataModel.g.dataset_uri(new_feature_id, group="features")

    with DatasetManager(src, out=src, dtype="float32", fillvalue=0) as DM:
        out_dataset = DM.out
        out_dataset[:] = feature_vol

    print(f"Created new feature with id: {new_feature_id}")


def slice_plot(
    img_volume,
    pts=None,
    bg_vol=None,
    slice_idxs=(0, 0, 0),
    suptitle="",
    plot_color=False,
    boxpadding=-1,
    unique_color_plot=False,
    figsize=(12, 12),
):
    z, x, y = slice_idxs
    print(f"Plotting at location: {z}, {x}, {y}")
    plt.figure(figsize=figsize)
    plt.suptitle(suptitle, fontsize=20)

    if boxpadding != -1:
        from survos2.entity.sampler import crop_pts_bb

        bb = [
            z - boxpadding[0],
            z + boxpadding[0],
            x - boxpadding[1],
            x + boxpadding[1],
            y - boxpadding[2],
            y + boxpadding[2],
        ]
        pts = crop_pts_bb(pts, bb)

    if plot_color:
        img = np.zeros((img_volume.shape[1], img_volume.shape[2], 3))
        if bg_vol is None:
            img[:, :, 1] = img_volume[z, :]
        else:
            img[:, :, 1] = img_volume[z, :]
            img[:, :, 2] = bg_vol[z, :]
        plt.imshow(img)
    else:
        if bg_vol is None:
            plt.imshow(img_volume[z, :], cmap="gray")
        else:
            plt.imshow(img_volume[z, :] + bg_vol[z, :], cmap="gray")
        plt.title(f"XY, Z: {z}")
        if pts is not None:

            if unique_color_plot:

                unique = np.unique(pts[:, 3])
                for i, u in enumerate(unique):
                    xs = pts[:, 1][pts[:, 3] == u]
                    ys = pts[:, 2][pts[:, 3] == u]
                    color_list = [plt.cm.jet(i / float(len(unique))) for i in range(len(unique))]

                    plt.scatter(xs, ys, c=color_list[i], label=u, cmap="jet")
            else:
                plt.scatter(pts[:, 1], pts[:, 2])
        plt.legend()

    plt.title(f"XY, Z: {z}")

    plt.figure(figsize=(figsize[0] - 1, figsize[1] - 1))
    if bg_vol is None:
        plt.imshow(img_volume[:, :, y], cmap="gray")
    else:
        plt.imshow(img_volume[:, :, y] + bg_vol[:, :, y], cmap="gray")
    plt.title(f"ZX, Y:{y}")
    if pts is not None:
        plt.scatter(pts[:, 1], pts[:, 0], c=pts[:, 3], cmap="jet")

    plt.figure(figsize=(figsize[0] - 1, figsize[1] - 1))
    if bg_vol is None:
        plt.imshow(img_volume[:, x, :], cmap="gray")
    else:
        plt.imshow(img_volume[:, x, :] + bg_vol[:, x, :], cmap="gray")

    plt.title(f"ZY, X:{x}")
    if pts is not None:
        plt.scatter(pts[:, 2], pts[:, 0], c=pts[:, 3], cmap="jet")


def slice_plot_bw(img_volume, pts=None, bg_vol=None, slice_idxs=(0, 0, 0), suptitle=""):
    z, x, y = slice_idxs
    print(z, x, y)
    plt.figure(figsize=(12, 12))
    plt.suptitle(suptitle, fontsize=20)
    if bg_vol is None:
        plt.imshow(img_volume[z, :], cmap="gray")
    else:
        plt.imshow(img_volume[z, :] + bg_vol[z, :], cmap="gray")
    plt.title(f"XY, Z: {z}")
    if pts is not None:
        plt.scatter(pts[:, 1], pts[:, 2])

    plt.figure(figsize=(12, 12))
    if bg_vol is None:
        plt.imshow(img_volume[:, :, y], cmap="gray")
    else:
        plt.imshow(img_volume[:, :, y] + bg_vol[:, :, y], cmap="gray")
    plt.title(f"ZX, Y:{y}")
    if pts is not None:
        plt.scatter(pts[:, 1], pts[:, 0])

    plt.figure(figsize=(12, 12))
    if bg_vol is None:
        plt.imshow(img_volume[:, x, :], cmap="gray")
    else:
        plt.imshow(img_volume[:, x, :] + bg_vol[:, x, :], cmap="gray")

    plt.title(f"ZY, X:{x}")
    if pts is not None:
        plt.scatter(
            pts[:, 2],
            pts[:, 0],
        )


def view_volume(imgvol, name=""):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(imgvol, name=name)

    return viewer


def view_volumes(imgvols, name=""):
    with napari.gui_qt():
        viewer = napari.Viewer()
        names = [str(i) for i in range(len(imgvols))]

        for i in range(len(imgvols)):
            viewer.add_image(imgvols[i], name=names[i])

    return viewer


def view_label(label_vol, name="Label"):
    with napari.gui_qt():

        viewer = napari.Viewer()
        viewer.add_labels(label_vol, name=name)

        return viewer


def view_labels(img_vols, label_vol, name=""):
    with napari.gui_qt():

        viewer = napari.Viewer()
        label_layer = viewer.add_labels(
            label_vol,
            name="segmentation",
        )

        return viewer


def view_vols_labels(img_vols, label_vol, name=""):
    with napari.gui_qt():

        viewer = napari.Viewer()
        names = [str(i) for i in range(len(img_vols))]

        label_layer = viewer.add_labels(
            label_vol,
            name="Labels",
            # properties=label_properties,
            # color=color,
        )

        for i in range(len(img_vols)):
            viewer.add_image(quick_norm(img_vols[i]), name=names[i])

        return viewer


def view_vols_points(img_vols, entities, names=None, flipxy=True):
    with napari.gui_qt():
        viewer = napari.Viewer()

        if names is None:
            names = [str(i) for i in range(len(img_vols))]

        for i in range(len(img_vols)):
            viewer.add_image(img_vols[i], name=names[i])

        sel_start, sel_end = 0, len(entities)

        if flipxy:
            centers = np.array(
                [
                    [
                        np.int(np.float(entities.iloc[i]["z"])),
                        np.int(np.float(entities.iloc[i]["y"])),
                        np.int(np.float(entities.iloc[i]["x"])),
                    ]
                    for i in range(sel_start, sel_end)
                ]
            )

        else:
            centers = np.array(
                [
                    [
                        np.int(np.float(entities.iloc[i]["z"])),
                        np.int(np.float(entities.iloc[i]["x"])),
                        np.int(np.float(entities.iloc[i]["y"])),
                    ]
                    for i in range(sel_start, sel_end)
                ]
            )

        num_classes = len(np.unique(entities["class_code"])) + 5
        print(f"Number of entity classes {num_classes}")
        palette = np.array(sns.color_palette("hls", num_classes))  # num_classes))
        face_color_list = [palette[class_code] for class_code in entities["class_code"]]

        viewer.add_points(
            centers,
            size=[10] * len(centers),
            face_color=face_color_list,
            n_dimensional=True,
        )

        return viewer


class NumpyEncoder(json.JSONEncoder):
    """
    usage:
    j=json.dumps(results,cls=NumpyEncoder)
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def isstring(s):
    # if we use Python 3
    if sys.version_info[0] >= 3:
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def summary_stats(arr):
    return (
        np.min(arr),
        np.max(arr),
        np.mean(arr),
        np.std(arr),
        np.median(arr),
        arr.shape,
    )


def make_directories(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def show_images(images, titles=None, figsize=(12, 12), suptitle=""):
    n_images = len(images)

    if titles is None:
        titles = [f"{im.shape} {str(im.dtype)}" for im in images]

    fig = plt.figure(figsize=figsize)

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(1, n_images, n + 1)
        if image.ndim == 2:
            plt.gray()

        plt.imshow(image)
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def show_image_grid(images, titles=None, row_len=8, figsize=(12, 12), suptitle=""):
    n_images = len(images)
    num_col = int(n_images / row_len)

    if titles is None:
        titles = [f"{im.shape} {str(im.dtype)}" for im in images]

    fig = plt.figure(figsize=figsize)

    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(1, n_images, n + 1)
        if image.ndim == 2:
            plt.gray()

        plt.imshow(image)
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def grid_of_images_and_clicks(
    image_list, clicks, point, n_rows, n_cols, image_titles="", figsize=(20, 20)
):
    images = [image_list[i] for i in range(n_rows * n_cols)]
    f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):

            axarr[i, j].imshow(images[i * n_cols + j], cmap="gray", origin="lower")
            axarr[i, j].set_title(img_titles[i * n_cols + j])
            axarr[i, j].scatter([point[0]], [point[1]], c="red")

    plt.rcParams["axes.grid"] = False
    plt.tight_layout()


# plotting the selected grid around the click points
def grid_of_images2(
    image_list,
    n_rows,
    n_cols,
    image_titles="",
    bigtitle="",
    figsize=(20, 20),
    color="black",
):
    images = [image_list[i] for i in range(n_rows * n_cols)]

    if image_titles == "":
        image_titles = [str(t) for t in list(range(len(images)))]

    f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):
            axarr[i, j].imshow(images[i * n_cols + j], cmap="gray")
            axarr[i, j].grid(False)
            # axarr[i, j].set_title(image_titles[i * n_cols + j], fontsize=10, color=color)
            axarr[i, j].tick_params(
                labeltop=False, labelleft=False, labelbottom=False, labelright=False
            )
    f.suptitle(bigtitle, color=color)


def grid_of_images(image_list, n_rows, n_cols, image_titles="", figsize=(20, 20)):
    images = [image_list[i] for i in range(n_rows * n_cols)]

    f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):
            axarr[i, j].imshow(images[i * n_cols + j])
            if image_titles == "":
                axarr[i, j].set_title(str(i * n_cols + j))
            else:
                axarr[i, j].set_title("Label: " + str(image_titles[(i * n_cols + j)]))

    plt.tight_layout()


def show_images_and_points(
    images,
    points,
    cluster_classes,
    class_names=None,
    titles=None,
    figsize=(12, 4),
    uselegend=True,
    cleanmargin=False,
):
    n_ims = len(images)
    if titles is None:
        titles = ["(%d)" % i for i in range(1, n_ims + 1)]
    fig = plt.figure(figsize=figsize)
    n = 1
    plt.style.use("ggplot")
    if class_names is None:
        class_names = [str(i) for i in cluster_classes]
    for image, title in zip(images, titles):

        a = fig.add_subplot(1, n_ims, n)
        plt.imshow(image, cmap="gray")

        cmap = matplotlib.colors.ListedColormap(
            ["cyan", "orange", "maroon", "navy", "mediumvioletred", "palegreen"], N=None
        )

        scat = a.scatter(points[:, 1], points[:, 2], c=cluster_classes, cmap=cmap, s=80, alpha=1.0)
        if uselegend:
            lgnd = a.legend(
                handles=scat.legend_elements()[0],
                labels=class_names,
                fontsize=28,
                loc="upper right",
            )
            for handle in lgnd.legendHandles:
                handle._legmarker.set_markersize(26.0)
        if not cleanmargin:
            a.set_title(title)
        else:
            a.set_xticks([])
            a.set_yticks([])
        n += 1

    return fig
