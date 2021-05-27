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


def add_anno(anno_vol, new_name, workspace_name):
    result = survos.run_command(
        "annotations",
        "add_level",
        uri=None,
        workspace=workspace_name,
    )
    print(result)
    new_anno_id = result[0]["id"]
    result = survos.run_command(
        "annotations",
        "rename",
        uri=None,
        feature_id=new_anno_id,
        new_name=new_name,
        workspace=workspace_name,
    )

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


def get_entities(objects_id):
    msg = {"objects_id": objects_id}
    logger.debug(f"view_objects {msg['objects_id']}")
    src = DataModel.g.dataset_uri(msg["objects_id"], group="objects")
    with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
        ds = DM.sources[0]

    logger.debug(f"Using dataset {ds}")
    entities_fullname = ds.get_metadata("fullname")
    logger.info(f"Viewing entities {entities_fullname}")
    from survos2.frontend.components.entity import setup_entity_table

    tabledata, entities_df = setup_entity_table(entities_fullname)
    return entities_df


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
    img = I > 0
    label_im, nb_labels = ndimage.label(img)
    sizes = ndimage.sum(I, label_im, range(nb_labels + 1))
    max_sz = np.max(sizes)
    lab_sz = sizes[label_im]
    cc = lab_sz == max_sz
    cc = cc.astype(int)

    return cc


def get_surface(img_vol, plot3d=False):
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

    return mesh, s, v, sphericity


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


def get3dcc(pred_vol):
    # pred_t= torch.FloatTensor(pred_vol)
    # pred_t = torch.abs(pred_t.unsqueeze(0))
    # input_tens = input_tens / torch.max(input_tens)
    # pred_ssm_t = kornia.dsnt.spatial_softmax_2d(pred_t, temperature=0.1)

    # print(f"SSM stats: {summary_stats(pred_ssm_t.numpy())}")
    # pred_ssm = pred_ssm_t.squeeze(0).detach().numpy()

    pred_vol_mask = (((pred_vol > 0.45)) * 1.0).astype(np.uint16)
    # pred_vol_mask = pred_ssm.astype(np.uint16)
    # print(pred_vol_mask.shape)

    connectivity = 6
    labels_out = cc3d.connected_components(
        pred_vol_mask, connectivity=connectivity
    )  # 26-connected

    return labels_out


def prepare_bv(mask):
    label_vol = get3dcc(pred_vol)
    list_of_roi = regionprops(label_vol)
    return list_of_roi


def plot_bb_2d(img, pred_info):
    ax = None
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.imshow(img)

    bbs = pred_info["bbs"]
    preds = pred_info["preds"]
    scores = pred_info["scores"]
    bb_ids = pred_info["bb_ids"]

    for bbox, c, scr, bb_id in zip(bbs, preds, scores, bb_ids):
        txt = c
        draw_bb(
            ax, [bbox[1], bbox[0], bbox[3], bbox[2]], text=f"{txt} {scr:.2f} \n {bb_id}"
        )
