import numpy as np
import pandas as pd
from collections import Counter
from statistics import mode, StatisticsError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


def centroid_3d(arr):
    length = arr.shape[0]

    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])

    return sum_x / length, sum_y / length, sum_z / length


def show_images_and_points(
    images, points, cluster_classes, class_names=None, titles=None, figsize=(12, 4)
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
        scat = a.scatter(
            points[:, 1], points[:, 2], c=cluster_classes, cmap="jet_r", s=50, alpha=1.0
        )
        a.legend(handles=scat.legend_elements()[0], labels=class_names)

        a.set_title(title)
        n += 1


def rescale_3d(X, x_scale, y_scale, z_scale):

    X_rescaled = np.zeros_like(X)
    X_rescaled[:, 0] = X[:, 0] * x_scale
    X_rescaled[:, 1] = X[:, 1] * y_scale
    X_rescaled[:, 2] = X[:, 2] * z_scale

    return X_rescaled


def chip_cluster(
    orig_pts,
    chip,
    offset_x,
    offset_y,
    min_cluster_size=5,
    min_samples=1,
    eps=5,
    method="hdbscan",
    plot_all=False,
    debug_verbose=False,
    quantile_threshold=0.95,
):
    """Cluster and simplify point cloud associated with a chip volume.

    Args:
        orig_pts (np.ndarray): array of 3d points
        chip (np.ndarray): chip volume
        offset_x (int): x offset in original volume
        offset_y (int): y offset in original volume
        min_cluster_size (int, optional): minimum number of points in  a cluster. Defaults to 5.
        method (str, optional): cluster method. Defaults to 'hdbscan'.
        plot_all (bool, optional): debug plotting. Defaults to False.
        debug_verbose (bool, optional): debug messages. Defaults to False.

    Returns:
        np.ndarray, np.ndarray: cluster centroids, cluster classes
    """

    X = orig_pts.copy()
    img_sample = chip[chip.shape[0] // 2, :]
    subsample = False
    print(f"Image {chip.shape}")
    print(f"Clustering pts {orig_pts.shape}")

    if plot_all:

        plt.figure(figsize=(14, 14))
        plt.gca().invert_yaxis()
        plt.scatter(
            orig_pts[:, 1] - offset_x,
            orig_pts[:, 2] - offset_y,
            s=5,
            linewidth=0,
            c=orig_pts[:, 3],
            alpha=1.0,
        )
        plt.title("Raw Points Before")

    if subsample:
        X = np.array(X)

        samp_prop = 0.3
        num_samp = int(samp_prop * len(X))

        X = np.floor(np.random.permutation(X))[0:num_samp, :]

    scale_minmax = False

    if scale_minmax:

        scale_x = 1.0 / np.max(X[:, 0])
        scale_y = 1.0 / np.max(X[:, 1])
        scale_z = 1.0 / np.max(X[:, 2])
        if debug_verbose:
            print("Scaling by {} {} {}".format(scale_x, scale_y, scale_z))

        # X_rescaled = rescale_3d(orig_pts, scale_x, scale_y, scale_z)
        x_scale = xend - xstart
        y_scale = yend - ystart
        slicestart = 0
        sliceend = vol_shape_z
        z_scale = (sliceend - slicestart) * 5  # HACK (not sure how best to scale z)
        X_rescaled = rescale_3d(X, scale_x, scale_y, scale_z)
    else:
        X_rescaled = X.copy()

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

    #
    # Point cloud cluster
    #

    if method == "hdbscan":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=1
        ).fit(X_rescaled)
        label_code = clusterer.labels_
        num_clusters_found = len(np.unique(label_code))
        threshold = pd.Series(clusterer.outlier_scores_).quantile(quantile_threshold)
        outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

        X_rescaled_cl = np.delete(X_rescaled, outliers, axis=0)
        label_code_cl = np.delete(label_code, outliers, axis=0)
        cluster_probs_cl = np.delete(clusterer.probabilities_, outliers, axis=0)
        num_outliers_removed = X_rescaled.shape[0] - X_rescaled_cl.shape[0]

        if debug_verbose:
            print("Limits: {} {} {} ".format(xlim, ylim, zlim))
            print(np.min(X[:, 0]), np.min(X[:, 1]), np.min(X[:, 2]))
            print(
                "Orig: {} Clean: {} Num points rem: {}".format(
                    X_rescaled.shape[0], X_rescaled_cl.shape[0], num_outliers_removed
                )
            )
            print(
                "Proportion removed: {}".format(
                    num_outliers_removed / X_rescaled.shape[0]
                )
            )

    elif method == "dbscan":

        clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(X_rescaled)
        label_code_cl = clusterer.labels_
        X_rescaled_cl = X_rescaled[label_code_cl != -1]
        num_clusters_found = len(np.unique(label_code_cl))

    cluster_coords = []
    cluster_sizes = []

    for l in np.unique(label_code_cl)[0:]:
        cluster_coords.append(X_rescaled_cl[label_code_cl == l])

    cluster_coords = np.array(cluster_coords)
    cluster_sizes = np.array([len(cluster_coord) for cluster_coord in cluster_coords])
    print(f"Cluster sizes {cluster_sizes.shape}")

    cluster_centroids = np.array(
        [centroid_3d(cluster_coord) for cluster_coord in cluster_coords]
    )

    cluster_centroids = np.array(cluster_centroids)
    cluster_sizes = np.array(cluster_sizes)

    title_str = "Number of original clicks: {0} Number of final centroids: {1} Av clicks cluster {2}".format(
        X_rescaled.shape[0],
        cluster_sizes.shape[0],
        X_rescaled.shape[0] / cluster_sizes.shape[0],
    )

    slice_center = int(chip.shape[0] / 2.0)
    cc2 = np.roll(cluster_centroids, shift=2, axis=1)
    slice_top, slice_bottom = slice_center + 5, slice_center - 5

    centroid_coords_woffset = cc2.copy()
    centroid_coords_woffset[:, 1] = centroid_coords_woffset[:, 1] - offset_y
    centroid_coords_woffset[:, 0] = centroid_coords_woffset[:, 0] - offset_x
    cc = []

    for c in cluster_coords:
        cluster_classes = list(c[:, 3].astype(np.uint32))
        try:
            classes_mode = mode(cluster_classes)
        except StatisticsError as e:
            classes_mode = np.random.choice(cluster_classes)
        cc.append(classes_mode)
        # print(f"Assigned class for cluster: {classes_mode}")

    if debug_verbose:
        print(f"Number of clusters: {len(cluster_coords)}")
        # print(f"Cluster classes {cc}")
        print(f"Len cluster classes {len(cc)}")

    clustered = np.zeros((cluster_centroids.shape[0], 4))
    clustered[:, 0:3] = cluster_centroids
    clustered[:, 3] = cc

    if plot_all:

        plt.figure(figsize=(14, 14))
        plt.gca().invert_yaxis()
        plt.scatter(
            centroid_coords_woffset[:, 0] - offset_x,
            centroid_coords_woffset[:, 1] - offset_y,
            s=5,
            linewidth=0,
            c=cc,
            alpha=1.0,
        )
        plt.title("Raw Points After")

        show_images_and_points(
            [
                img_sample,
            ],
            cluster_centroids,
            cc,
            figsize=(12, 12),
        )

    print(f"Produced clustered output of shape: {clustered.shape}")

    return clustered


# add more clustering
def chip_cluster2(
    orig_pts,
    chip,
    offset_x,
    offset_y,
    MIN_CLUSTER_SIZE=5,
    method="hdbscan",
    plot_all=False,
    debug_verbose=False,
):
    X = orig_pts.copy()
    img_sample = chip[0, :]
    subsample = False

    print(f"Image {chip.shape}")
    print(f"Clustering pts {orig_pts.shape}")

    if plot_all:
        plt.figure(figsize=(14, 14))
        plt.scatter(
            orig_pts[:, 1] - offset_x,
            orig_pts[:, 2] - offset_y,
            s=60,
            linewidth=0,
            c=orig_pts[:, 3],
            alpha=1.0,
        )
        plt.title("Before")

    if subsample:
        X = np.array(X)

        samp_prop = 0.3
        num_samp = int(samp_prop * len(X))

        X = np.floor(np.random.permutation(X))[0:num_samp, :]

    scale_minmax = False

    if scale_minmax:
        scale_x = 1.0 / np.max(X[:, 0])
        scale_y = 1.0 / np.max(X[:, 1])
        scale_z = 1.0 / np.max(X[:, 2])
        if debug_verbose:
            print("Scaling by {} {} {}".format(scale_x, scale_y, scale_z))

        # X_rescaled = rescale_3d(orig_pts, scale_x, scale_y, scale_z)
        x_scale = xend - xstart
        y_scale = yend - ystart
        slicestart = 0
        sliceend = vol_shape_z
        z_scale = (sliceend - slicestart) * 5  # HACK (not sure how best to scale z)
        X_rescaled = rescale_3d(X, scale_x, scale_y, scale_z)
    else:
        X_rescaled = X.copy()

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

    #
    # Point cloud cluster
    #

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit(X_rescaled)

        label_code = clusterer.labels_

        num_clusters_found = len(np.unique(label_code))
        threshold = pd.Series(clusterer.outlier_scores_).quantile(0.65)
        outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

        # X_rescaled_cl = np.delete(X_rescaled,  outliers,axis=0)
        # label_code_cl = np.delete(label_code,  outliers,axis=0)
        # cluster_probs_cl = np.delete(clusterer.probabilities_, outliers, axis=0)
        X_rescaled_cl = X_rescaled
        label_code_cl = label_code
        cluster_probs_cl = clusterer.probabilities_
        num_outliers_removed = X_rescaled.shape[0] - X_rescaled_cl.shape[0]

        if debug_verbose:
            # print(X_rescaled_cl.shape, label_code_cl.shape)
            print("Limits: {} {} {} ".format(xlim, ylim, zlim))
            print(np.min(X[:, 0]), np.min(X[:, 1]), np.min(X[:, 2]))

            print(
                "Orig: {} Clean: {} Num points rem: {}".format(
                    X_rescaled.shape[0], X_rescaled_cl.shape[0], num_outliers_removed
                )
            )
            print(
                "Proportion removed: {}".format(
                    num_outliers_removed / X_rescaled.shape[0]
                )
            )

        if plot_all:
            plt.figure(figsize=(14, 14))
            pal = sns.color_palette("husl", num_clusters_found + 1)
            # pal = sns.color_palette('cubehelix', num_clusters_found + 1)
            # pal = sns.color_palette('Paired', num_clusters_found + 1)

            cleaned_colors = [
                sns.desaturate(pal[col], sat)
                for col, sat in zip(label_code_cl, cluster_probs_cl)
            ]
            plt.imshow(img_sample, cmap="gray")
            plt.title("After")
            plt.scatter(
                X_rescaled_cl[:, 1] - offset_x,
                X_rescaled_cl[:, 2] - offset_y,
                s=60,
                linewidth=0,
                c=cleaned_colors,
                alpha=1.0,
            )

    elif method == "dbscan":
        clusterer = DBSCAN(eps=3, min_samples=2).fit(X_rescaled)
        label_code = clusterer.labels_

    cluster_coords = []
    cluster_sizes = []

    for l in np.unique(label_code_cl)[0:]:

        cluster_coords.append(X_rescaled_cl[label_code_cl == l])

    cluster_coords = np.array(cluster_coords)

    cluster_sizes = np.array([len(cluster_coord) for cluster_coord in cluster_coords])
    print(f"Cluster sizes {cluster_sizes.shape}")

    cluster_centroids = np.array(
        [centroid_3d(cluster_coord) for cluster_coord in cluster_coords]
    )

    cluster_centroids = np.array(cluster_centroids)

    # sns.distplot(cluster_centroids[np.isfinite(cluster_centroids)], rug=True)

    cluster_sizes = np.array(cluster_sizes)

    # sns.distplot(cluster_sizes[np.isfinite(cluster_sizes)], rug=True)

    title_str = "Number of original clicks: {0} Number of final centroids: {1} Av clicks cluster {2}".format(
        X_rescaled.shape[0],
        cluster_sizes.shape[0],
        X_rescaled.shape[0] / cluster_sizes.shape[0],
    )

    slice_center = int(chip.shape[0] / 2.0)
    cc2 = np.roll(cluster_centroids, shift=2, axis=1)
    slice_top, slice_bottom = slice_center + 5, slice_center - 5

    centroid_coords_woffset = cc2.copy()
    centroid_coords_woffset[:, 1] = centroid_coords_woffset[:, 1] - offset_y
    centroid_coords_woffset[:, 0] = centroid_coords_woffset[:, 0] - offset_x
    print(centroid_coords_woffset)
    cc = []

    for c in cluster_coords:
        cluster_classes = list(c[:, 3].astype(np.uint32))

        try:
            classes_mode = mode(cluster_classes)

        except StatisticsError as e:
            classes_mode = np.random.choice(cluster_classes)

        cc.append(classes_mode)
        # print(f"Assigned class for cluster: {classes_mode}")

    if debug_verbose:
        print(f"Number of clusters: {len(cluster_coords)}")
        print(f"Cluster classes {cc}")
        print(f"Len cluster classes {len(cc)}")

    # if plot_all:
    #    show_images_and_points([img_sample,], centroid_coords_woffset, cc, figsize=(12,12))

    return cluster_centroids, cc
