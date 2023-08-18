import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
from matplotlib import offsetbox
from matplotlib import pyplot as plt
from survos2.frontend.nb_utils import grid_of_images2, show_images, show_image_grid

sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


def cluster_scatter(x, colors, text_labels=False):
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int32)])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.axis("off")
    ax.axis("tight")

    if text_labels:
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
    print(f"Plotting {num_classes} clusters.")
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
                offsetbox.OffsetImage(images[i], cmap=cmap),
                proj[i],
            )
            imagebox.set_zorder(-1)
            ax.add_artist(imagebox)

    ax.scatter(proj[:, 0], proj[:, 1], lw=0, s=40, c=palette[colors.astype(np.int32)], zorder=1)
    txts = []

    for i in range(num_classes):
        xtext, ytext = np.median(proj[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24, zorder=200)
        print(f"Class {i}")
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )

    txts.append(txt)
    print(txts)



def image_grid(
    image_list,
    n_cols,
    fig,
    axarr,
    image_titles="",
    bigtitle="",
    color="black",
):
    print(f"List of length {len(image_list)}")
    if len(image_list) < n_cols:
        print("Too few images.")
    else:
        n_rows = min((len(image_list) // n_cols) + 1, 5)
        print(f"Number of rows: {n_rows}, Number of columns {n_cols}")
        images = [image_list[i] for i in range(len(image_list))]

        if image_titles == "":
            image_titles = [str(t) for t in list(range(len(images)))]

        for i in range(n_rows):
            for j in range(n_cols):
                index = i * n_cols + j
                print(index)
                if index < len(images):
                    axarr[i, j].imshow(images[index], cmap="gray")

                axarr[i, j].grid(False)
                # axarr[i, j].set_title(image_titles[i * n_cols + j], fontsize=10, color=color)
                axarr[i, j].tick_params(
                    labeltop=False, labelleft=False, labelbottom=False, labelright=False
                )
                axarr[i, j].margins(0)
                axarr[i, j].axes.axis("off")

        fig.suptitle(bigtitle, color=color)

    return fig, axarr


# def image_grids(images, labels):
#     figs = []
#     for i in np.unique(labels):
#         print(i)
#         fig, _ = image_grid(images[labels == i], 3, figsize=(8, 8))
#         figs.append(fig)
#     return figs
