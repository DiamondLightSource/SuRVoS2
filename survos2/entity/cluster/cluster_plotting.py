import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
from matplotlib import offsetbox
from matplotlib import pyplot as plt

sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


def cluster_scatter(x, colors):
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.axis("off")
    ax.axis("tight")
    ax.set_title(
        "Plot of k-means clustering of small click windows using ResNet-18 Features"
    )

    txts = []

    for i in range(num_classes):
        xtext, ytext = np.median(x[colors == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=24)

        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )

        txts.append(txt)


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

    ax = ax or plt.gca()

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

    ax.scatter(
        proj[:, 0], proj[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)], zorder=200
    )
