import os
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector
from numpy import nonzero, zeros
from numpy.random import permutation
from PyQt5.QtCore import QPoint, QSettings, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from survos2.server.state import cfg
from loguru import logger
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel, Workspace

DEFAULT_DIR_KEY = "default_dir"
DEFAULT_DATA_KEY = "default_data_dir"



def remove_masked_pts(bg_mask, entities):
    pts_vol = np.zeros_like(bg_mask)
    
    #bg_mask = binary_dilation(bg_mask * 1.0, disk(2).astype(np.bool))
    for pt in entities:
        if pt[0] < bg_mask.shape[0] and pt[1] < bg_mask.shape[1] and pt[2] < bg_mask.shape[2]:
            pts_vol[pt[0], pt[1], pt[2]] = 1
        else:
            print(pt)
    bg_mask = bg_mask > 0
    pts_vol = pts_vol * bg_mask
    zs, xs, ys = np.where(pts_vol == 1)
    masked_entities = []
    for i in range(len(zs)):
        pt = [zs[i], xs[i], ys[i]]
        masked_entities.append(pt)
    return np.array(masked_entities)




def get_array_from_dataset(src_dataset, axis=0):
    if cfg.retrieval_mode == 'slice':
        print(f"src_dataset shape {src_dataset.shape}")
        dataset = src_dataset.copy(order="C")
        dataset = np.transpose(dataset, np.array(cfg.order)).astype(np.float32)
        src_arr = dataset[cfg.current_slice, :, :]
    elif cfg.retrieval_mode == 'volume':
        src_arr = src_dataset[:]

    return src_arr


class WorkerThread(QThread):
    def run(self):
        def work():
            cfg.ppw.clientEvent.emit(
                {"source": "save_annotation", "data": "save_annotation", "value": None,}
            )
            cfg.ppw.clientEvent.emit(
                {"source": "save_annotation", "data": "refresh", "value": None}
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


def get_color_mapping(result, level_id="001_level"):
    logger.debug(f"Getting color mapping for level {level_id}")
    labels = []
    for r in result:
        if r["kind"] == "level":
            if r["id"] == level_id:
                cmapping = {}
                for ii, (k, v) in enumerate(r["labels"].items()):
                    labels.append(ii)
                    remapped_label = int(k) - 1
                    cmapping[remapped_label] = hex_string_to_rgba(v["color"])
                    cmapping[
                        remapped_label + (remapped_label * 16)
                    ] = hex_string_to_rgba(v["color"])
                return cmapping, labels


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


class SComboBox(QComboBox):
    def __init__(self, *args):
        super(SComboBox, self).__init__(*args)


class ComboDialog(QDialog):
    def __init__(self, options=None, parent=None, title="Select option"):
        super(ComboDialog, self).__init__(parent=parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)

        self.combo = SComboBox()
        for option in options:
            self.combo.addItem(option)
        layout.addWidget(self.combo)

        # OK and Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setGeometry(300, 300, 300, 100)

    @staticmethod
    def getOption(options, parent=None, title="Select option"):
        dialog = ComboDialog(options, parent=parent, title=title)
        result = dialog.exec_()
        option = dialog.combo.currentText()
        return (option, result == QDialog.Accepted)

    @staticmethod
    def getOptionIdx(options, parent=None, title="Select option"):
        dialog = ComboDialog(options, parent=parent, title=title)
        result = dialog.exec_()
        option = dialog.combo.currentIndex()
        return (option, result == QDialog.Accepted)


class MplCanvas(QWidget):

    roi_updated = pyqtSignal(tuple)

    def __init__(self, orient=0, axisoff=True, autoscale=False, **kwargs):
        super(MplCanvas, self).__init__()
        self.orient = orient
        self.setLayout(QVBoxLayout())

        # Figure
        self.fig, self.ax, self.canvas = self.figimage(axisoff=axisoff)
        self.rs = RectangleSelector(
            self.ax,
            self.line_select_callback,
            drawtype="box",
            useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        self.pressed = False
        self.ax.autoscale(enable=autoscale)
        self.layout().addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self.on_press)

    def replot(self):
        self.ax.clear()
        self.ax.cla()

    def redraw(self):
        self.canvas.draw_idle()

    def figimage(self, scale=1, dpi=None, axisoff=True):
        fig = plt.figure(figsize=(10, 10))
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if axisoff:
            fig.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.99)
        canvas.draw()
        return fig, ax, canvas

    def on_press(self, event):
        self.setFocus()
        if event.button == 1 or event.button == 3 and not self.rs.active:
            self.redraw()
            self.rs.set_active(True)
        else:
            self.rs.set_active(False)

    def line_select_callback(self, eclick, erelease):
        self.roi_updated.emit(self.rs.extents)


class FileWidget(QWidget):

    path_updated = pyqtSignal(str)

    def __init__(
        self, extensions="*.h5", home=None, folder=False, save=True, parent=None
    ):
        super(FileWidget, self).__init__(parent)
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(hbox)

        self.extensions = extensions
        self.folder = folder
        self.save = save

        if save and home == "~":
            home = os.path.expanduser("~")
        elif home is None:
            home = "Click to select Folder" if folder else "Click to select File"

        self.path = QLineEdit(home)
        self.path.setReadOnly(True)
        self.path.mousePressEvent = self.find_path
        self.selected = False

        hbox.addWidget(self.path)

    def find_path(self, ev):
        if ev.button() != 1:
            return
        self.open_dialog()

    def open_dialog(self):
        selected = False
        path = None
        settings = QSettings()
        folder = settings.value(DEFAULT_DATA_KEY)
        if not folder:
            folder = settings.value(DEFAULT_DIR_KEY)

        path, _ = QFileDialog.getOpenFileName(
            self, "Select input source", folder, filter=self.extensions
        )
        if path is not None and len(path) > 0:
            selected = True
            settings.setValue(DEFAULT_DATA_KEY, os.path.dirname(path))

        if selected:
            self.path.setText(path)
            self.path_updated.emit(path)
            self.selected = True

    def value(self):
        return self.path.text() if self.selected else None
