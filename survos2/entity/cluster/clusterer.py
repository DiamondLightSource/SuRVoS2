import numpy as np
import umap
import pprint
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics

from survos2.entity.entities import make_bvol_df, offset_points
from entityseg.cluster.patch_cluster import patch_2d_features, extract_cnn_features
from entityseg.cluster.utils import pad_vol, quick_norm
from entityseg.cluster.cnn_features import prepare_3channel
from entityseg.cluster.cluster_plotting import cluster_scatter, plot_clustered_img

from survos2.entity.entities import offset_points
from survos2.entity.sampler import centroid_to_bvol
from survos2.frontend.nb_utils import grid_of_images2
from survos2.entity.instance.dataset import BoundingVolumeDataset


def select_clusters(
    predictions,
    selected_images,
    target_cluster_num=[0],
    bg_cluster_num=[1],
    plot_cluster_grid=False,
):
    num_classes = len(np.unique(predictions))
    predictions_bincount = np.bincount(predictions)
    ii = np.nonzero(predictions_bincount)[0]
    cluster_sizes = list(zip(ii, predictions_bincount[ii]))
    print("Number of clusters {}\n\nCluster sizes:\n".format(num_classes))
    pprint.pprint(cluster_sizes)

    selected_images = np.array(selected_images)
    if plot_cluster_grid:
        grid_dim = 3
        for cluster_idx, cluster_sz in cluster_sizes:
            print(cluster_idx, cluster_sz)
            cluster_idx = int(cluster_idx)
            sel_idx = np.array(np.where(predictions == cluster_idx)[0])
            sel_1 = np.random.permutation(selected_images[sel_idx])
            # plt.figure()

            if cluster_sz > grid_dim ** 2:
                grid_of_images2(
                    sel_1,
                    grid_dim,
                    grid_dim,
                    bigtitle=int(cluster_idx),
                    color="white",
                    figsize=(6, 6),
                )

    # Choose target cluster indexes
    identified_cluster_num = bg_cluster_num.copy()
    identified_cluster_num.extend(target_cluster_num)
    other_cluster_num = list(range(num_classes))
    other_cluster_num = [
        e for e in other_cluster_num if e not in identified_cluster_num
    ]

    print(
        "Objects {}\nBackground {}\nOther {}".format(
            target_cluster_num, bg_cluster_num, other_cluster_num
        )
    )
    bg_sel_idx = np.any([predictions == idx for idx in bg_cluster_num], axis=0)
    target_sel_idx = np.any([predictions == idx for idx in target_cluster_num], axis=0)
    other_sel_idx = np.any([predictions == idx for idx in other_cluster_num], axis=0)

    return target_sel_idx, bg_sel_idx, other_sel_idx


def select_clusters2(
    predictions,
    patch_dict,
    target_cluster_num=[0],
    bg_cluster_num=[1],
    plot_cluster_grid=False,
):
    num_classes = len(np.unique(predictions))
    predictions_bincount = np.bincount(predictions)
    ii = np.nonzero(predictions_bincount)[0]
    cluster_sizes = list(zip(ii, predictions_bincount[ii]))
    print("Number of clusters {}\n\nCluster sizes:\n".format(num_classes))
    pprint.pprint(cluster_sizes)
    selected_images = np.array([patch_dict[idx][1] for idx in list(patch_dict.keys())])

    if plot_cluster_grid:
        grid_dim = 3
        for cluster_idx, cluster_sz in cluster_sizes:
            print(cluster_idx, cluster_sz)
            cluster_idx = int(cluster_idx)
            sel_idx = np.array(np.where(predictions == cluster_idx)[0])
            sel_1 = np.random.permutation(selected_images[sel_idx])
            # plt.figure()

            if cluster_sz > grid_dim ** 2:
                grid_of_images2(
                    sel_1,
                    grid_dim,
                    grid_dim,
                    bigtitle="",
                    color="white",
                    figsize=(6, 6),
                )
                plt.title(cluster_idx)

    # Choose target cluster indexes
    identified_cluster_num = bg_cluster_num.copy()
    identified_cluster_num.extend(target_cluster_num)
    other_cluster_num = list(range(num_classes))
    other_cluster_num = [
        e for e in other_cluster_num if e not in identified_cluster_num
    ]

    print("Objects {}\nBackground {}".format(target_cluster_num, bg_cluster_num))
    clustered_images = np.array(selected_images)

    bg_sel_idx = np.any([predictions == idx for idx in bg_cluster_num], axis=0)
    bg_samples = clustered_images[bg_sel_idx]
    target_sel_idx = np.any([predictions == idx for idx in target_cluster_num], axis=0)
    target_samples = clustered_images[target_sel_idx]
    other_sel_idx = np.any([predictions == idx for idx in other_cluster_num], axis=0)
    other_samples = clustered_images[other_sel_idx]

    print(
        "Number of selected object samples {}\nNumber of selected bg samples {} and other samples {}".format(
            len(target_samples), len(bg_samples), len(other_samples)
        )
    )

    return bg_sel_idx, target_sel_idx, other_sel_idx


def cluster_and_embed_patch_features(
    feature_mat,
    selected_images,
    n_components=10,
    num_clusters=15,
    params={'perplexity':150,'n_iter':2000},
    skip_px=2,
    embed_type="TSNE",
):
    print(f"Using feature matrix of size {feature_mat.shape}")
    selected_3channel = prepare_3channel(
        selected_images,
        patch_size=(selected_images[0].shape[0], selected_images[0].shape[1]),
    )
    patch_clusterer = PatchCluster(num_clusters, n_components)
    patch_clusterer.prepare_data(feature_mat)
    patch_clusterer.fit()
    patch_clusterer.predict()

    if embed_type == "TSNE":
        patch_clusterer.embed_TSNE(**params)
    elif embed_type == "UMAP":
        patch_clusterer.embed_UMAP(params)
    else:
        print("Embedding type not supported")
        return

    print(f"Metrics (DB, Sil): {patch_clusterer.cluster_metrics()}")
    preds = patch_clusterer.get_predictions()
    cluster_scatter(patch_clusterer.embedding, preds)

    selected_images_arr = np.array(selected_3channel)
    print(f"Plotting {selected_images_arr.shape} patch images.")
    selected_images_arr = selected_images_arr[:, :, :, 1]

    fig, ax = plt.subplots(figsize=(17, 17))
    plt.title(
        f"Windows around a click clustered using deep features and embedded in 2d using {embed_type}"
    )

    plot_clustered_img(
        patch_clusterer.embedding,
        preds,
        images=selected_images_arr[:, ::skip_px, ::skip_px],
    )

    return preds, patch_clusterer


def prepare_patches_for_clustering(
    vol, entities, bvol_dim=(32, 32, 32), axis=0, slice_idx=16, flipxy=False, gpu_id=0
):

    print(f"Volume to cluster shape: {vol.shape} {bvol_dim}")
    padded_vol = pad_vol(vol, bvol_dim)
    selected_entity_loc_offset = offset_points(entities, -np.array(bvol_dim))

    entity_bvols = centroid_to_bvol(
        selected_entity_loc_offset, bvol_dim=bvol_dim, flipxy=True
    )
    bvd = BoundingVolumeDataset(
        padded_vol, entity_bvols, selected_entity_loc_offset[:, 3]
    )
    if axis==0:    
        selected_images = np.array([im[0][slice_idx, :] for im in bvd])
    elif axis==1:
        selected_images = np.array([im[0][:,slice_idx, :] for im in bvd])
    elif axis==2:
        selected_images = np.array([im[0][:,slice_idx] for im in bvd])

    features = extract_cnn_features(selected_images)

    return features, selected_images


def prepare_patches_for_clustering2(
    vol, entities, bvol_dim=(32, 32, 32), flipxy=False, gpu_id=0
):
    # padded_vol = pad_vol(vol, bvol_dim)
    # selected_entity_loc = np.array(offset_points(entities, (32, 32, 32)))

    padded_vol = vol
    selected_entity_loc = entities
    entity_bvols = centroid_to_bvol(
        selected_entity_loc, bvol_dim=bvol_dim, flipxy=flipxy
    )
    # entity_mask = viz_bvols(padded_vol, entity_bvols)
    # slice_plot(entity_mask, None, padded_vol, (60,400,400))
    print(entity_bvols.shape)
    entity_bvols = np.hstack((entity_bvols, np.zeros((entity_bvols.shape[0], 1))))
    entity_bvols_df = make_bvol_df(entity_bvols)
    print(entity_bvols_df)
    vec_mat, selected_images, patch_dict, bbs_info = patch_2d_features(
        quick_norm(padded_vol), entity_bvols_df, gpu_id=gpu_id
    )
    return vec_mat, selected_images, patch_dict


def prepare_subset(clustered_images, preds, target_cluster_idxs, label_value=0):
    target_samples = clustered_images[
        np.any([preds == idx for idx in target_cluster_idxs], axis=0)
    ]

    # print("Number of selected object samples {}\nNumber of selected bg samples {}".format(
    #    len(target_samples), len(other_samples)))
    train_samples = []
    # train_samples.extend(bg_samples)
    train_samples.extend(target_samples)
    train_samples = np.array(train_samples)
    print(len(train_samples))

    # generate target labels
    # bg_class = [0] * len(bg_samples)
    target_class = [0] * len(target_samples)

    # train_y =  bg_class + target_class
    train_y = target_class
    train_y = np.array(train_y)

    print(len(train_y))
    train_y_df = pd.DataFrame(train_y)
    train_y_df

    train_y_labels = [int(t) for t in train_y]
    train_samples.shape, len(train_y_labels)

    return train_samples, train_y_labels


class PatchCluster:
    def __init__(
        self,
        n_clusters,
        n_components,
        method="k-means++",
        n_init=15,
        epochs=300,
        cores=None,
    ):
        self.method = method
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.setup_model(method, n_clusters, n_init, epochs, cores)

    def setup_model(
        self, method="k-means++", n_clusters=5, n_init=15, epochs=300, cores=1
    ):
        self.n_clusters = n_clusters
        if method == "k-means++":
            self.model = KMeans(
                init="k-means++",
                n_clusters=n_clusters,
                n_init=n_init,
                max_iter=epochs,
            )

    def prepare_data(self, data):
        print(
            "Reducing Dimensionality using PCA to {} number of components.".format(
                self.n_components
            )
        )
        reduced_data = PCA(n_components=self.n_components).fit_transform(data)
        self.reduced_data = reduced_data

    def fit(self):
        print("Calculating Clusters for {} clusters.".format(self.n_clusters))
        self.model.fit(self.reduced_data)

    def predict(self):
        print("Predicting Clusters")
        self.predictions = self.model.predict(self.reduced_data)

    def get_predictions(self):
        print(f"Returning predictions of shape {self.predictions.shape}")
        return self.predictions

    def embed_TSNE(self, n_components=2, verbose=1, perplexity=80, n_iter=1400):
        tsne = TSNE(
            n_components=n_components,
            verbose=verbose,
            perplexity=perplexity,
            n_iter=n_iter,
        )
        self.embedding = tsne.fit_transform(self.reduced_data)
        print(f"TSNE resulted in projected data of shape {self.embedding.shape}")

    def embed_UMAP(self, params):
        umap_embedder = umap.UMAP(**params)
        self.embedding = umap_embedder.fit_transform(self.reduced_data)
        print(f"UMAP resulted in projected data of shape {self.embedding.shape}")

    def density_cluster(self, min_cluster_size=3):
        import hdbscan

        self.density_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, gen_min_span_tree=True
        )
        self.density_clusterer.fit(self.embedding)

    def clustering_evaluation(self, n_cluster_range=(2, 20)):
        distortions = []
        silhouette = []

        for k in range(n_cluster_range[0], n_cluster_range[1]):
            self.setup_model(n_clusters=k)
            self.fit()
            distortions.append(self.model.inertia_)
            silhouette.append(
                metrics.silhouette_score(
                    self.reduced_data, self.predictions, metric="euclidean"
                )
            )

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(n_cluster_range[0], n_cluster_range[1]), distortions)
        plt.grid(True)
        plt.title("Elbow curve")

        fig = plt.figure(figsize=(15, 5))
        plt.plot(range(n_cluster_range[0], n_cluster_range[1]), silhouette)
        plt.grid(True)
        plt.title("Silhouette")

    def cluster_metrics(self):
        cluster_metrics = {
            "davies_bouldin": metrics.davies_bouldin_score(
                self.reduced_data, self.predictions
            ),
            "silhouette": metrics.silhouette_score(
                self.reduced_data, self.predictions, metric="euclidean"
            ),
        }

        return cluster_metrics
