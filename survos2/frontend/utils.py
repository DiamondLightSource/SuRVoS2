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
from PyQt5.QtCore import QThread, QTimer


class WorkerThread(QThread):
    def run(self):
        def work():
            cfg.ppw.clientEvent.emit(
                {
                    "source": "update_annotation",
                    "data": "update_annotation",
                    "value": None,
                }
            )
            cfg.ppw.clientEvent.emit(
                {"source": "update_annotation", "data": "refresh", "value": None}
            )
            QThread.sleep(5)

        timer = QTimer()
        timer.timeout.connect(work)
        timer.start(50000)
        self.exec_()


def coords_in_view(coords, image_shape):
    if (
        coords[0] >= 0
        and coords[1] >= 0
        and coords[0] < image_shape[0]
        and coords[1] < image_shape[1]
    ):
        return True
    else:
        return False


def hex_string_to_rgba(hex_string):
    hex_value = hex_string.lstrip("#")
    rgba_array = (
        np.append(np.array([int(hex_value[i : i + 2], 16) for i in (0, 2, 4)]), 255.0)
        / 255.0
    )
    return rgba_array


def get_color_mapping(result):
    for r in result:
        level_name = r["name"]
        if r["kind"] == "level":
            cmapping = {}
            print(r["labels"].items())
            for ii, (k, v) in enumerate(r["labels"].items()):
                # remapped_label = label_ids[ii]
                remapped_label = int(k) - 1
                print(remapped_label)
                cmapping[remapped_label] = hex_string_to_rgba(v["color"])
                cmapping[remapped_label + (remapped_label * 16)] = hex_string_to_rgba(
                    v["color"]
                )
    return cmapping


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
