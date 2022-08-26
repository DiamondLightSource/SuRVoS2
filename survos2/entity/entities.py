import collections
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import as_strided as ast

from survos2.entity.sampler import viz_bvols, viz_bb, centroid_to_bvol, offset_points
from survos2.frontend.control.launcher import Launcher
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager

import tempfile


def load_boxes_via_file(boxes_arr, flipxy=True):
    boxes_df = make_entity_boxes(boxes_arr, flipxy=flipxy)
    tmp_fullpath = os.path.abspath(os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv"))
    boxes_df.to_csv(tmp_fullpath, line_terminator="")
    print(boxes_df)
    object_scale = 1.0
    object_offset = (0.0, 0.0, 0.0)
    object_crop_start = (0.0, 0.0, 0.0)
    object_crop_end = (1e9, 1e9, 1e9)

    params = dict(
        order=1,
        workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        fullname=tmp_fullpath,
    )

    result = Launcher.g.run("objects", "create", **params)
    if result:
        dst = DataModel.g.dataset_uri(result["id"], group="objects")
        params = dict(
            dst=dst,
            fullname=tmp_fullpath,
            scale=object_scale,
            offset=object_offset,
            crop_start=object_crop_start,
            crop_end=object_crop_end,
        )
        logger.debug(f"Creating objects with params {params}")
        Launcher.g.run("objects", "boxes", **params)

    os.remove(tmp_fullpath)


def load_entities_via_file(entities_arr, flipxy=True):
    entities_df = make_entity_df(entities_arr, flipxy=flipxy)
    tmp_fullpath = os.path.abspath(os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + ".csv"))
    entities_df.to_csv(tmp_fullpath, line_terminator="")

    object_scale = 1.0
    object_offset = (0.0, 0.0, 0.0)
    object_crop_start = (0.0, 0.0, 0.0)
    object_crop_end = (1e9, 1e9, 1e9)

    params = dict(
        order=0,
        workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        fullname=tmp_fullpath,
    )

    result = Launcher.g.run("objects", "create", **params)
    if result:
        dst = DataModel.g.dataset_uri(result["id"], group="objects")
        params = dict(
            dst=dst,
            fullname=tmp_fullpath,
            scale=object_scale,
            offset=object_offset,
            crop_start=object_crop_start,
            crop_end=object_crop_end,
        )
        logger.debug(f"Creating objects with params {params}")
        Launcher.g.run("objects", "points", **params)

    os.remove(tmp_fullpath)


def get_entities_df(objects_id):
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


def make_entity_mask(vol, dets, flipxy=True, bvol_dim=(32, 32, 32)):
    from survos2.entity.utils import pad_vol

    offset_dets = offset_points(dets, (-bvol_dim[0], -bvol_dim[1], -bvol_dim[2]))
    offset_det_bvol = centroid_to_bvol(offset_dets, bvol_dim=bvol_dim, flipxy=flipxy)
    padded_vol = pad_vol(vol, bvol_dim)
    det_mask = viz_bvols(padded_vol, offset_det_bvol)

    return det_mask, offset_dets, padded_vol


def calc_bounding_vol(m):
    return [
        m[0][0],
        m[0][0] + m[1][0],
        m[0][1],
        m[0][1] + m[1][1],
        m[0][2],
        m[0][2] + m[1][2],
    ]


def calc_bounding_vols(main_bv):
    for k, v in main_bv.items():
        main_bv[k]["bb"] = calc_bounding_vol(v["key_coords"])
    return main_bv


def uncrop_pad(img, orig_img_shape, crop_bb):
    blank_img = np.zeros(orig_img_shape)
    blank_img[
        crop_bb[0] : crop_bb[0] + img.shape[0],
        crop_bb[2] : crop_bb[2] + img.shape[1],
        crop_bb[4] : crop_bb[4] + img.shape[2],
    ] = img
    return blank_img


def offset_points(pts, patch_pos):
    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]

    logger.debug(f"Offset: {offset_x}, {offset_y}, {offset_z}")

    z = pts[:, 0].copy() - offset_z
    x = pts[:, 1].copy() - offset_x
    y = pts[:, 2].copy() - offset_y

    c = pts[:, 3].copy()

    offset_pts = np.stack([z, x, y, c], axis=1)

    return offset_pts


def make_entity_df(pts, flipxy=True):
    """
    Converts an array with 4 columns for z,x,y and class code into a typed dataframe.
    A Entity Dataframe has 'z','x','y','class_code'.

    """

    if flipxy:
        entities_df = pd.DataFrame({"z": pts[:, 0], "x": pts[:, 2], "y": pts[:, 1], "class_code": pts[:, 3]})
    else:
        entities_df = pd.DataFrame({"z": pts[:, 0], "x": pts[:, 1], "y": pts[:, 2], "class_code": pts[:, 3]})

    entities_df = entities_df.astype({"x": "int32", "y": "int32", "z": "int32", "class_code": "int32"})
    return entities_df


def make_entity_df2(pts):
    entities_df = pd.DataFrame({"z": pts[:, 0], "x": pts[:, 2], "y": pts[:, 1], "class_code": pts[:, 3]})

    entities_df = entities_df.astype({"x": "float32", "y": "float32", "z": "int32", "class_code": "int32"})

    return entities_df


def make_entity_bvol(bbs, flipxy=False):
    if flipxy:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "area": bbs[:, 1],
                "z": bbs[:, 2],
                "x": bbs[:, 4],
                "y": bbs[:, 3],
                "bb_s_z": bbs[:, 5],
                "bb_s_x": bbs[:, 6],
                "bb_s_y": bbs[:, 7],
                "bb_f_z": bbs[:, 8],
                "bb_f_x": bbs[:, 9],
                "bb_f_y": bbs[:, 10],
            }
        )
    else:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "area": bbs[:, 1],
                "z": bbs[:, 2],
                "x": bbs[:, 3],
                "y": bbs[:, 4],
                "bb_s_z": bbs[:, 5],
                "bb_s_x": bbs[:, 6],
                "bb_s_y": bbs[:, 7],
                "bb_f_z": bbs[:, 8],
                "bb_f_x": bbs[:, 9],
                "bb_f_y": bbs[:, 10],
            }
        )

    entities_df = entities_df.astype(
        {
            "x": "int32",
            "y": "int32",
            "z": "int32",
            "class_code": "int32",
            "bb_s_z": "int32",
            "bb_s_x": "int32",
            "bb_s_y": "int32",
            "bb_f_z": "int32",
            "bb_f_x": "int32",
            "bb_f_y": "int32",
            "area": "int32",
        }
    )
    return entities_df


def make_entity_boxes(bbs, flipxy=False):
    if flipxy:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "z": bbs[:, 1],
                "x": bbs[:, 3],
                "y": bbs[:, 2],
                "bb_s_z": bbs[:, 4],
                "bb_s_x": bbs[:, 5],
                "bb_s_y": bbs[:, 6],
                "bb_f_z": bbs[:, 7],
                "bb_f_x": bbs[:, 8],
                "bb_f_y": bbs[:, 9],
            }
        )
    else:
        entities_df = pd.DataFrame(
            {
                "class_code": bbs[:, 0],
                "z": bbs[:, 1],
                "x": bbs[:, 2],
                "y": bbs[:, 3],
                "bb_s_z": bbs[:, 4],
                "bb_s_x": bbs[:, 5],
                "bb_s_y": bbs[:, 6],
                "bb_f_z": bbs[:, 7],
                "bb_f_x": bbs[:, 8],
                "bb_f_y": bbs[:, 9],
            }
        )

    entities_df = entities_df.astype(
        {
            "class_code": "int32",
            "z": "int32",
            "x": "int32",
            "y": "int32",
            "bb_s_z": "int32",
            "bb_s_x": "int32",
            "bb_s_y": "int32",
            "bb_f_z": "int32",
            "bb_f_x": "int32",
            "bb_f_y": "int32",
        }
    )
    return entities_df


def make_bvol_df(bbs, flipxy=False):
    entities_df = pd.DataFrame(
        {
            "bb_s_z": bbs[:, 0],
            "bb_s_x": bbs[:, 1],
            "bb_s_y": bbs[:, 2],
            "bb_f_z": bbs[:, 3],
            "bb_f_x": bbs[:, 4],
            "bb_f_y": bbs[:, 5],
            "class_code": bbs[:, 6],
        }
    )

    entities_df = entities_df.astype(
        {
            "bb_s_z": "int32",
            "bb_s_x": "int32",
            "bb_s_y": "int32",
            "bb_f_z": "int32",
            "bb_f_x": "int32",
            "bb_f_y": "int32",
            "class_code": "int32",
        }
    )
    return entities_df


def make_bounding_vols(entities, patch_size=(14, 14, 14)):
    p_z, p_x, p_y = patch_size

    bbs = []

    for z, x, y, c in entities:
        bb_s_z = z - p_z
        bb_s_x = x - p_x
        bb_s_y = y - p_y
        bb_f_z = z + p_z
        bb_f_x = x + p_x
        bb_f_y = y + p_y
        area = (2 * p_z) * (2 * p_x) * (2 * p_y)
        bbs.append([c, area, z, x, y, bb_s_z, bb_s_y, bb_s_x, bb_f_z, bb_f_y, bb_f_x])
    bbs = np.array(bbs)

    return bbs
