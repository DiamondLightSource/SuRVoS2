import itertools
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loguru import logger
from survos2.entity.anno.geom import centroid_3d, rescale_3d
from survos2.frontend.nb_utils import summary_stats


def entitybvol_to_cropbvol(bvol):
    """
    from z1 z2 x1 x2 y1 y2
    to z1 x1 y1 z2 x2 y2
    """
    b = np.zeros_like(bvol)
    b[0] = bvol[0]
    b[1] = bvol[3]
    b[2] = bvol[1]
    b[3] = bvol[4]
    b[4] = bvol[2]
    b[5] = bvol[5]

    return b


def detnetbvol_to_cropbvol(bvol):
    """
    from x1 y1 x2 y2 z1 z2
    to z1 x1 y1 z2 x2 y2

    """
    b = np.zeros_like(bvol)
    b[0] = bvol[4]
    b[1] = bvol[0]
    b[2] = bvol[1]
    b[3] = bvol[5]
    b[4] = bvol[2]
    b[5] = bvol[3]
    return b


def cropbvol_to_detnet_bvol(bvol):
    """
    from z1 x1 y1 z2 x2 y2
    to x1 y1 x2 y2 z1 z2
    """
    b = np.zeros_like(bvol)
    b[0] = bvol[1]
    b[1] = bvol[2]
    b[2] = bvol[4]
    b[3] = bvol[5]
    b[4] = bvol[0]
    b[5] = bvol[3]
    return b


def produce_patches(padded_vol, padded_anno, offset_locs, bvol_grid):
    patches = []
    patches_pts = []
    patches_bvols = []
    patches_anno = []

    for i in range(len(bvol_grid)):
        patch, patch_pts = crop_vol_and_pts_bb(
            padded_vol, offset_locs, entitybvol_to_cropbvol(bvol_grid[i]), offset=True
        )
        patch_anno, patch_pts = crop_vol_and_pts_bb(
            padded_anno, offset_locs, entitybvol_to_cropbvol(bvol_grid[i]), offset=True
        )
        patch_bvol = centroid_to_detnet_bvol(patch_pts, bvol_dim=(10, 10, 10))
        patches.append(patch)
        patches_bvols.append(patch_bvol)
        patches_pts.append(patch_pts)
        patches_anno.append(patch_anno)

    return patches, patches_anno, patches_bvols, patches_pts


def grid_of_points(padded_vol, padding, grid_dim=(4, 16, 16), sample_grid=False):

    z_dim, x_dim, y_dim = grid_dim
    spacez = np.linspace(0, padded_vol.shape[0] - (2 * padding[0]), z_dim)
    spacex = np.linspace(0, padded_vol.shape[1] - (2 * padding[1]), x_dim)
    spacey = np.linspace(0, padded_vol.shape[2] - (2 * padding[2]), y_dim)

    zv, xv, yv = np.meshgrid(spacez, spacex, spacey)
    print(zv.shape, xv.shape, yv.shape)

    zv = zv + padding[0]
    xv = xv + padding[1]
    yv = yv + padding[2]

    gridarr = np.stack((zv, xv, yv)).astype(np.uint32)
    gridarr[:, 1, 1, 1]

    zv_f = zv.reshape((z_dim * x_dim * y_dim))
    xv_f = xv.reshape((z_dim * x_dim * y_dim))
    yv_f = yv.reshape((z_dim * x_dim * y_dim))

    class_code = [0] * len(zv_f)

    trans_pts = np.stack((zv_f, xv_f, yv_f, class_code)).T.astype(np.uint32)

    return trans_pts


def generate_random_points(vol, num_pts, patch_size):
    pts = np.random.random((num_pts, 4))
    z_size, x_size, y_size = patch_size
    pts[:, 0] = pts[:, 0] * (vol.shape[0] - z_size * 2) + z_size // 2
    pts[:, 1] = pts[:, 1] * (vol.shape[1] - y_size * 2) + x_size // 2
    pts[:, 2] = pts[:, 2] * (vol.shape[2] - x_size * 2) + y_size // 2
    pts = np.abs(pts)
    return pts


def generate_random_points_in_volume(vol, num_pts, border=(32, 32, 32)):
    pts = np.random.random((num_pts, 4))
    pts[:, 0] = pts[:, 0] * (vol.shape[0] - (2 * border[0])) + border[0]
    pts[:, 1] = pts[:, 1] * (vol.shape[1] - (2 * border[1])) + border[1]
    pts[:, 2] = pts[:, 2] * (vol.shape[2] - (2 * border[2])) + border[2]
    pts = np.abs(pts)
    return pts


def offset_points(pts, offset, scale=32, random_offset=False):
    trans_pts = pts.copy()

    trans_pts[:, 0] = pts[:, 0] + offset[0]
    trans_pts[:, 1] = pts[:, 1] + offset[1]
    trans_pts[:, 2] = pts[:, 2] + offset[2]

    if random_offset:

        offset_rand = np.random.random(trans_pts.shape) * scale
        offset_rand[:, 3] = np.zeros((len(trans_pts)))

        trans_pts = trans_pts + offset_rand

    return trans_pts


def centroid_to_detnet_bvol(centers, bvol_dim=(10, 10, 10), flipxy=False):
    """Centroid to bounding volume

    Parameters
    ----------
    centers : np.ndarray, (nx3)
        3d coordinates of the point to use as the centroid of the bounding box
    bvol_dim : tuple, optional
        Dimensions of the bounding volume centered at the points given by centers, by default (10, 10, 10)
    flipxy : bool, optional
        Flip x and y coordinates, by default False

    Returns
    -------
    np.ndarray, (nx6)
        (x_start, y_start, x_fin, y_fin, z_start, z_fin)
    """
    d, w, h = bvol_dim

    if flipxy:
        bvols = np.array(
            [
                (cx - w, cy - h, cx + w, cy + h, cz - d, cz + d)
                for cz, cx, cy, _ in centers
            ]
        )
    else:
        bvols = np.array(
            [
                (cx - w, cy - h, cx + w, cy + h, cz - d, cz + d)
                for cz, cy, cx, _ in centers
            ]
        )

    return bvols


def centroid_to_bvol(centers, bvol_dim=(10, 10, 10), flipxy=False):
    """Centroid to bounding volume

    Parameters
    ----------
    centers : np.ndarray, (nx3)
        3d coordinates of the point to use as the centroid of the bounding box
    bvol_dim : tuple, optional
        Dimensions of the bounding volume centered at the points given by centers, by default (10, 10, 10)
    flipxy : bool, optional
        Flip x and y coordinates, by default False

    Returns
    -------
    np.ndarray, (nx6)
        (z_start, x_start, y_start, z_fin, x_fin, y_fin)
    """
    d, w, h = bvol_dim
    if flipxy:
        bvols = np.array(
            [
                (cz - d, cx - w, cy - h, cz + d, cx + w, cy + h)
                for cz, cx, cy, _ in centers
            ]
        )
    else:
        bvols = np.array(
            [
                (cz - d, cx - w, cy - h, cz + d, cx + w, cy + h)
                for cz, cy, cx, _ in centers
            ]
        )

    return bvols


def centroid_to_boxes(centers, bvol_dim=(10, 10, 10), flipxy=False):
    """Centroid to bounding volume

    Parameters
    ----------
    centers : np.ndarray, (nx3)
        3d coordinates of the point to use as the centroid of the bounding box
    bvol_dim : tuple, optional
        Dimensions of the bounding volume centered at the points given by centers, by default (10, 10, 10)
    flipxy : bool, optional
        Flip x and y coordinates, by default False

    Returns
    -------
    np.ndarray, (nx6)
        (z_start, x_start, y_start, z_fin, x_fin, y_fin)
    """
    d, w, h = bvol_dim
    if flipxy:
        bvols = np.array(
            [
                (0, cz, cx, cy, cz - d, cx - w, cy - h, cz + d, cx + w, cy + h)
                for cz, cx, cy, _ in centers
            ]
        )
    else:
        bvols = np.array(
            [
                (0, cz, cx, cy, cz - d, cx - w, cy - h, cz + d, cx + w, cy + h)
                for cz, cy, cx, _ in centers
            ]
        )

    return bvols


def sample_volumes(sel_entities, precropped_vol):
    sampled_vols = []
    for i in range(len(sel_entities)):
        ent = sel_entities.iloc[i]
        bb = np.array(
            [
                ent["bb_s_z"],
                ent["bb_f_z"],
                ent["bb_s_x"],
                ent["bb_f_x"],
                ent["bb_s_y"],
                ent["bb_f_y"],
            ]
        ).astype(np.uint32)
        sampled_vols.append(sample_bvol(precropped_vol, bb))
    return sampled_vols


def viz_bvols(input_array, bvols, flip_coords=False):
    bvol_mask = np.zeros_like(input_array)
    print(f"Making {len(bvols)} bvols")
    for bvol in bvols:
        # print(bvol)
        bvol = bvol.astype(np.int32)
        z_s = np.max((0, bvol[0]))
        z_f = np.min((bvol[3], input_array.shape[0]))
        x_s = np.max((0, bvol[1]))
        x_f = np.min((bvol[4], input_array.shape[1]))
        y_s = np.max((0, bvol[2]))
        y_f = np.min((bvol[5], input_array.shape[2]))
        # print(f"Sampling {z_s}, {z_f}, {x_s}, {x_f}, {y_s}, {y_f}")

        if flip_coords:
            bvol_mask[z_s:z_f, y_s:y_f, x_s:x_f] = 1.0
        else:
            bvol_mask[z_s:z_f, x_s:x_f, y_s:y_f] = 1.0

    return bvol_mask


def viz_bvols2(input_array, bvols, flip_coords=False, edge_thickness=2):
    bvol_mask = np.zeros_like(input_array)
    print(f"Making {len(bvols)} bvols")
    for bvol in bvols:
        # print(bvol)
        bvol = bvol.astype(np.int32)
        z_s = np.max((0, bvol[0]))
        z_f = np.min((bvol[3], input_array.shape[0]))
        x_s = np.max((0, bvol[1]))
        x_f = np.min((bvol[4], input_array.shape[1]))
        y_s = np.max((0, bvol[2]))
        y_f = np.min((bvol[5], input_array.shape[2]))
        # print(f"Sampling {z_s}, {z_f}, {x_s}, {x_f}, {y_s}, {y_f}")

        if flip_coords:
            bvol_mask[z_s:z_f, y_s:y_f, x_s:x_f] = 1.0
            bvol_mask[z_s + 1 : z_f - 1, y_s + 1 : y_f - 1, x_s + 1 : x_f - 1] = 0.90
            bvol_mask[
                z_s + edge_thickness : z_f - edge_thickness,
                y_s + edge_thickness : y_f - edge_thickness,
                x_s + edge_thickness : x_f - edge_thickness,
            ] = 0.55
        else:
            bvol_mask[z_s:z_f, x_s:x_f, y_s:y_f] = 0.45
            bvol_mask[z_s + 1 : z_f - 1, x_s + 1 : x_f - 1, y_s + 1 : y_f - 1] = 0.85
            bvol_mask[
                z_s + edge_thickness : z_f - edge_thickness,
                y_s + edge_thickness : y_f - edge_thickness,
                x_s + edge_thickness : x_f - edge_thickness,
            ] = 0.45

    return bvol_mask


def sample_region_at_pt(img_volume, pt, dim):
    z, x, y = pt
    d, w, h = dim

    z_st = np.max((0, z - d))
    z_end = np.min((z + d, img_volume.shape[0]))
    x_st = np.max((0, x - w))
    x_end = np.min((x + w, img_volume.shape[1]))
    y_st = np.max((0, y - h))
    y_end = np.min((y + h, img_volume.shape[2]))

    return img_volume[z_st:z_end, x_st:x_end, y_st:y_end]


def sample_bvol(img_volume, bvol):
    z_st, z_end, x_st, x_end, y_st, y_end = bvol
    return img_volume[z_st:z_end, x_st:x_end, y_st:y_end]


def get_vol_in_cent_box(img_volume, z_st, z_end, x, y, w, h):
    return img_volume[z_st:z_end, x - w : x + w, y - h : y + h]


def sample_roi(img_vol, tabledata, i=0, vol_size=(32, 32, 32)):
    # Sampling ROI from an entity table
    print(f"Sampling from vol of shape {img_vol.shape}")
    pad_slice, pad_y, pad_x = np.array(vol_size) // 2

    z, x, y = tabledata["z"][i], tabledata["x"][i], tabledata["y"][i]
    logger.info(f"Sampling location {z} {x} {y}")
    # make a bv
    bb_zb = np.clip(int(z) - pad_slice, 0, img_vol.shape[0])
    bb_zt = np.clip(int(z) + pad_slice, 0, img_vol.shape[0])
    bb_xl = np.clip(int(x) - pad_slice, 0, img_vol.shape[1])
    bb_xr = np.clip(int(x) + pad_slice, 0, img_vol.shape[1])
    bb_yl = np.clip(int(y) - pad_slice, 0, img_vol.shape[2])
    bb_yr = np.clip(int(y) + pad_slice, 0, img_vol.shape[2])

    vol1 = get_vol_in_bbox(img_vol, bb_zb, bb_zt, bb_xl, bb_xr, bb_yl, bb_yr)

    print(f"Sampled vol of shape {vol1.shape}")
    if vol1.shape[0] == 0 or vol1.shape[1] == 0 or vol1.shape[2] == 0:
        vol1 = np.zeros(vol_size)
    return vol1


def get_vol_in_bbox(image_volume, slice_start, slice_end, xst, xend, yst, yend):
    return image_volume[slice_start:slice_end, xst:xend, yst:yend]


def get_centered_vol_in_bbox(image_volume, slice_start, slice_end, x, y, w, h):
    return image_volume[slice_start:slice_end, x - w : x + w, y - h : y + h]


def crop_vol_in_bbox(image_volume, slice_start, slice_end, x, y, w, h):
    return image_volume[slice_start:slice_end, x : x + w, y : y + h]


def get_centered_img_in_bbox(image_volume, sliceno, x, y, w, h):
    w = w // 2
    h = h // 2
    return image_volume[int(sliceno), x - w : x + w, y - h : y + h]


def get_img_in_bbox(image_volume, sliceno, x, y, w, h):
    return image_volume[int(sliceno), x - w : x + w, y - h : y + h]


@dataclass
class MarkedPatches:
    """Set of N patches, with associated per-patch 3d points
    There is also a per-patch location which is the location the patch was sampled from in the original volume.
    """

    vols: np.ndarray  # (N, Z, X, Y) image data within patch
    vols_pts: np.ndarray  # (N, Z, X, Y) cropped point geometry within patch
    vols_locs: np.ndarray  # (N, Z, X, Y, C) centroid location of patch and class code
    vols_bbs: np.ndarray  # (N, Z_start, Z_fin, X_start, X_fin, Y_start, Y_fin)bounding box for patch


# todo: list of patch sizes
# todo: pad
def sample_marked_patches(
    img_volume, locs, pts, patch_size=(32, 32, 32), debug_verbose=False
):
    """Samples a large image volume into a MarkedPatches object.
    Uses bounding volumes, and crops the image volume and associated geometry
    into a list of cropped volumes and cropped geometry.

    Parameters
    ----------
    img_volume : {np.ndarray}
        image volume
    locs : {np.array of N x 4}
        N point locations, with a label in the final column
    pts : {np.array of P x k}
         point cloud of size P (the first 3 columns are used as the z,x,y coords)
    patch_size : {tuple, int x 3)
        -- Size of patch to sample (default: {(32,32,32)}), optional
    debug_verbose : bool, optional
        [description], by default False

    Returns
    -------
    MarkedPatches
        volumes with associated geometry
    """
    vols = []
    img_titles = []
    vols_pts = []
    vols_locs = []
    vols_bbs = []

    print(
        f"Generating {len(locs)} patch volumes from image of shape {img_volume.shape}"
    )

    for j in range(len(locs)):

        if locs[j].shape[0] == 4:
            sliceno, x, y, c = locs[j]
        else:
            sliceno, x, y = locs[j]

        d, w, h = patch_size

        w = w // 2
        h = h // 2

        sliceno = int(sliceno)
        x = int(np.ceil(x))
        y = int(np.ceil(y))

        slice_start = np.max([0, sliceno - int(patch_size[0] / 2.0)])
        slice_end = np.min([sliceno + int(patch_size[0] / 2.0), img_volume.shape[0]])

        out_of_bounds = np.unique(
            np.hstack(
                (
                    np.where(pts[:, 1] <= x - w)[0],
                    np.where(pts[:, 1] >= x + w)[0],
                    np.where(pts[:, 2] <= y - h)[0],
                    np.where(pts[:, 2] >= y + h)[0],
                    np.where(pts[:, 0] <= slice_start)[0],
                    np.where(pts[:, 0] >= slice_end)[0],
                )
            )
        )

        pts_c = pts.copy()
        sel_pts = np.delete(pts_c, out_of_bounds, axis=0)

        if debug_verbose:
            print("Shape of original pt data {}".format(pts.shape))
            print("Number of out of bounds pts: {}".format(out_of_bounds.shape))

        img = get_centered_vol_in_bbox(img_volume, slice_start, slice_end, y, x, h, w)

        sel_pts[:, 0] = sel_pts[:, 0] - slice_start
        sel_pts[:, 1] = sel_pts[:, 1] - (x - w)
        sel_pts[:, 2] = sel_pts[:, 2] - (y - h)

        if img.shape == patch_size:
            # print(f"Number of points cropped in bounding box: {sel_pts.shape}")
            vols.append(img)

        else:
            incomplete_img = np.zeros(patch_size)
            incomplete_img[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
            # print(img.shape)
            vols.append(incomplete_img)
        vols_pts.append(sel_pts)
        vols_bbs.append([slice_start, slice_end, x - w, x + w, y - h, y + h])
        vols_locs.append(locs[j])

    vols = np.array(vols)
    vols_pts = np.array(vols_pts)
    vols_bbs = np.array(vols_bbs)
    vols_locs = np.array(vols_locs)

    marked_patches = MarkedPatches(vols, vols_pts, vols_locs, vols_bbs)
    print(f"Generated {len(locs)} MarkedPatches of shape {vols.shape}")

    return marked_patches


def crop_vol_and_pts(
    img_data,
    pts,
    location=(60, 700, 700),
    patch_size=(40, 300, 300),
    debug_verbose=False,
    offset=False,
):
    patch_size = np.array(patch_size).astype(np.uint32)
    location = np.array(location).astype(np.uint32)

    # z, x_bl, x_ur, y_bl, y_ur = location[0], location[1], location[1]+patch_size[1], location[2], location[2]+patch_size[2]

    slice_start = np.max([0, location[0]])
    slice_end = np.min([location[0] + patch_size[0], img_data.shape[0]])

    out_of_bounds_w = np.hstack(
        (
            np.where(pts[:, 2] >= location[2] + patch_size[2])[0],
            np.where(pts[:, 2] <= location[2])[0],
            np.where(pts[:, 1] >= location[1] + patch_size[1])[0],
            np.where(pts[:, 1] <= location[1])[0],
            np.where(pts[:, 0] <= location[0])[0],
            np.where(pts[:, 0] >= location[0] + patch_size[0])[0],
        )
    )

    cropped_pts = np.array(np.delete(pts, out_of_bounds_w, axis=0))

    if offset:

        cropped_pts[:, 0] = cropped_pts[:, 0] - location[0]
        cropped_pts[:, 1] = cropped_pts[:, 1] - location[1]
        cropped_pts[:, 2] = cropped_pts[:, 2] - location[2]

    if debug_verbose:
        print(
            "\n z x y w h: {}".format(
                (location[0], location[1], location[2], patch_size[1], patch_size[2])
            )
        )
        print("Slice start, slice end {} {}".format(slice_start, slice_end))
        print("Cropped points array shape: {}".format(cropped_pts.shape))

    img = crop_vol_in_bbox(
        img_data,
        slice_start,
        slice_end,
        location[2],
        location[1],
        patch_size[2],
        patch_size[1],
    )

    return img, cropped_pts


def crop_pts_bb(
    pts, bounding_box, location=(0, 0, 0), debug_verbose=False, offset=False
):

    z_st, z_end, x_st, x_end, y_st, y_end = bounding_box
    print(z_st, z_end, x_st, x_end, y_st, y_end)

    out_of_bounds_w = np.hstack(
        (
            np.where(pts[:, 0] <= z_st)[0],
            np.where(pts[:, 0] >= z_end)[0],
            np.where(pts[:, 1] <= x_st)[0],
            np.where(pts[:, 1] >= x_end)[0],
            np.where(pts[:, 2] <= y_st)[0],
            np.where(pts[:, 2] >= y_end)[0],
        )
    )

    cropped_pts = np.array(np.delete(pts, out_of_bounds_w, axis=0))

    if offset:
        location = (z_st, x_st, y_st)
        cropped_pts[:, 0] = cropped_pts[:, 0] - location[0]
        cropped_pts[:, 1] = cropped_pts[:, 1] - location[1]
        cropped_pts[:, 2] = cropped_pts[:, 2] - location[2]
        print(f"Offset location {location}")

    if debug_verbose:
        print("Cropped points array shape: {}".format(cropped_pts.shape))

    return cropped_pts


def crop_vol_and_pts_bb(
    img_volume, pts, bounding_box, debug_verbose=False, offset=False
):

    # TODO: clip bbox to img_volume
    z_st, z_end, y_st, y_end, x_st, x_end = bounding_box

    location = (z_st, x_st, y_st)
    out_of_bounds_w = np.hstack(
        (
            np.where(pts[:, 0] <= z_st)[0],
            np.where(pts[:, 0] >= z_end)[0],
            np.where(pts[:, 1] <= x_st)[0],
            np.where(pts[:, 1] >= x_end)[0],
            np.where(pts[:, 2] <= y_st)[0],
            np.where(pts[:, 2] >= y_end)[0],
        )
    )

    cropped_pts = np.array(np.delete(pts, out_of_bounds_w, axis=0))

    if offset:
        cropped_pts[:, 0] = cropped_pts[:, 0] - location[0]
        cropped_pts[:, 1] = cropped_pts[:, 1] - location[1]
        cropped_pts[:, 2] = cropped_pts[:, 2] - location[2]

    img = sample_bvol(img_volume, bounding_box)

    return img, cropped_pts


# old
def crop_vol_and_pts_centered(
    img_volume,
    pts,
    location=(60, 700, 700),
    patch_size=(40, 300, 300),
    debug_verbose=False,
    offset=False,
):
    patch_size = np.array(patch_size).astype(np.uint32)
    location = np.array(location).astype(np.uint32)
    # z, x_bl, x_ur, y_bl, y_ur = location[0], location[1], location[1]+patch_size[1], location[2], location[2]+patch_size[2]
    slice_start = np.max([0, location[0]])
    slice_end = np.min([location[0] + patch_size[0], img_volume.shape[0]])

    out_of_bounds_w = np.hstack(
        (
            np.where(pts[:, 2] >= location[2] + patch_size[2])[0],
            np.where(pts[:, 2] <= location[2])[0],
            np.where(pts[:, 1] >= location[1] + patch_size[1])[0],
            np.where(pts[:, 1] <= location[1])[0],
            np.where(pts[:, 0] <= location[0])[0],
            np.where(pts[:, 0] >= location[0] + patch_size[0])[0],
        )
    )

    cropped_pts = np.array(np.delete(pts, out_of_bounds_w, axis=0))

    if offset:

        cropped_pts[:, 0] = cropped_pts[:, 0] - location[0]
        cropped_pts[:, 1] = cropped_pts[:, 1] - location[1]
        cropped_pts[:, 2] = cropped_pts[:, 2] - location[2]

    if debug_verbose:
        print(
            "\n z x y w h: {}".format(
                (location[0], location[1], location[2], patch_size[1], patch_size[2])
            )
        )
        print("Slice start, slice end {} {}".format(slice_start, slice_end))
        print("Cropped points array shape: {}".format(cropped_pts.shape))

    img = crop_vol_in_bbox(
        img_volume,
        slice_start,
        slice_end,
        location[2],
        location[1],
        patch_size[2],
        patch_size[1],
    )

    return img, cropped_pts


def sample_patch_slices(img_vol, entities_df):
    entities_locs = np.array(entities_df[["slice", "x", "y"]])
    mp = sample_marked_patches(
        img_vol, entities_locs, entities_locs, patch_size=(64, 64, 64)
    )
    vol_list = mp.vols
    vol_locs = mp.vols_locs
    vol_pts = mp.vols_pts

    slice_list = np.array([v[vol_list[0].shape[0] // 2, :, :] for v in vol_list])

    print(f"Generated slice {slice_list.shape}")

    return slice_list, vol_pts


def gather_single_class(img_vol, entities_locs, class_code, patch_size=(64, 64, 64)):
    entities_locs_singleclass = entities_locs.loc[
        entities_locs["class_code"].isin([class_code])
    ]
    entities_locs_singleclass = np.array(entities_locs_singleclass[["slice", "x", "y"]])

    mp = sample_marked_patches(
        img_vol,
        entities_locs_singleclass,
        entities_locs_singleclass,
        patch_size=patch_size,
    )

    vol_list = mp.vols
    vol_locs = mp.vols_locs
    vol_pts = mp.vols_pts

    return vol_list, vol_locs, vol_pts


def sample_patch2d(img_volume, pts, patch_size=(40, 40)):
    img_shortlist = []
    img_titles = []
    print(f"Sampling {len(pts)} pts from image volume of shape {img_volume.shape}")

    for j in range(len(pts)):
        sliceno, y, x = pts[j]
        w, h = patch_size
        img = get_centered_img_in_bbox(
            img_volume, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h
        )
        img_shortlist.append(img)
        img_titles.append(str(int(x)) + "_" + str(int(y)) + "_" + str(sliceno))

    return img_shortlist, img_titles
