import math
from itertools import islice
from typing import Collection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import skimage
import torch
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage
from skimage import data, img_as_ubyte, measure

from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.frontend.nb_utils import show_images, summary_stats
from torch import Tensor
from torchvision.ops import roi_align

from survos2 import survos
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from loguru import logger
from survos2.frontend.main import init_ws, roi_ws



import torch


def accuracy(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0

def load_model(detmod, file_path):
    def load_model_parameters(full_path):
        checkpoint = torch.load(full_path)
        return checkpoint

    checkpoint = load_model_parameters(file_path)
    detmod.load_state_dict(checkpoint["model_state"], strict=False)
    detmod.eval()

    print(f"Loaded model from {file_path}")
    return detmod
    

def remove_masked_entities(bg_mask, entities):
    pts_vol = np.zeros_like(bg_mask)
    for pt in entities:
        pts_vol[pt[0], pt[1], pt[2]] = 1
    pts_vol = pts_vol * (1.0 - bg_mask)
    zs, xs, ys = np.where(pts_vol == 1)
    masked_entities = []
    for i in range(len(zs)):
        pt = [zs[i], ys[i], xs[i], 6]
        masked_entities.append(pt)
    return np.array(masked_entities)


def quick_norm(vol):
    vol -= np.min(vol)
    vol = vol / np.max(vol)
    return vol


def pad_vol(vol, padding):
    padded_vol = np.zeros(
        (
            vol.shape[0] + padding[0] * 2,
            vol.shape[1] + padding[1] * 2,
            vol.shape[2] + padding[2] * 2,
        )
    )

    padded_vol[
        padding[0] : vol.shape[0] + padding[0],
        padding[1] : vol.shape[1] + padding[1],
        padding[2] : vol.shape[2] + padding[2],
    ] = vol

    return padded_vol


def remove_padding(vol, padding):
    unpadded_vol = vol[
        padding[0] : vol.shape[0] + padding[0],
        padding[1] : vol.shape[1] + padding[1],
        padding[2] : vol.shape[2] + padding[2],
    ]
    return unpadded_vol


def get_largest_cc(I):
    cc = np.zeros_like(I)
    if np.sum(I) > 0:
        img = I > 0
        label_im, nb_labels = ndimage.label(img)
        sizes = ndimage.sum(I, label_im, range(nb_labels + 1))
        max_sz = np.max(sizes)
        lab_sz = sizes[label_im]
        cc = lab_sz == max_sz
        cc = cc.astype(int)

    return cc


def get_surface(img_vol, plot3d=False):
    try:
        verts, faces, normals, values = measure.marching_cubes((img_vol > 0) * 1.0, 0)

        mesh = Poly3DCollection(verts[faces])

        if plot3d:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection="3d")

            mesh.set_edgecolor("k")
            ax.add_collection3d(mesh)

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.set_xlim(0, img_vol.shape[1])
            ax.set_ylim(0, img_vol.shape[2])
            ax.set_zlim(0, img_vol.shape[0])

            plt.tight_layout()
            plt.show()

        s = skimage.measure.mesh_surface_area(verts, faces)
        v = np.sum(img_vol)

        sphericity = (36 * math.pi * (v ** 2)) / (s ** 3)
    except:
        s = 0
        v = 0
        sphericity = 0
        mesh = None

    return mesh, s, v, sphericity


def remove_padding(vol, padding):
    unpadded_vol = vol[
        padding[0] : vol.shape[0] + padding[0],
        padding[1] : vol.shape[1] + padding[1],
        padding[2] : vol.shape[2] + padding[2],
    ]
    return unpadded_vol


def accuracy(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    accuracy = np.divide(TP + TN, N)
    return accuracy * 100.0



