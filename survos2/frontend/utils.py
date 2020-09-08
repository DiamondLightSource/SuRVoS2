import numpy as np
from numpy import nonzero, zeros_like, zeros
from numpy.random import permutation
from napari import gui_qt
from napari import Viewer as NapariViewer
import napari
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time


def resource(*args):
    rdir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(rdir, "frontend/resources", *args))


def sample_from_bw(bwimg, sample_prop):
    pp = nonzero(bwimg)
    points = zeros([len(pp[0]), 2])
    points[:, 0] = pp[0]
    points[:, 1] = pp[1]
    num_samp = sample_prop * points.shape[0]
    points = np.floor(permutation(points))[0:num_samp, :]

    return points


def quick_norm(imgvol1):
    imgvol1 -= np.min(imgvol1)
    imgvol1 = imgvol1 / np.max(imgvol1)
    return imgvol1


# center-size, single slice
# def get_img_in_bbox(image_volume, sliceno, x,y,w,h):
#        return image_volume[int(sliceno), x-w:x+w, y-h:y+h]


# center-size for x/y but interval for slice
# def get_vol_in_bbox(image_volume, slice_start, slice_end, x,y,w,h):
#        return image_volume[slice_start:slice_end, x-w:x+w, y-h:y+h]


def prepare_point_data(pts, patch_pos):

    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]

    print(f"Offset: {offset_x}, {offset_y}, {offset_z}")

    z = pts[:, 0].copy() - offset_z
    x = pts[:, 1].copy() - offset_x
    y = pts[:, 2].copy() - offset_y

    c = pts[:, 3].copy()

    offset_pts = np.stack([z, x, y, c], axis=1)

    return offset_pts
