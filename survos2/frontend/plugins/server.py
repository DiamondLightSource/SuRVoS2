import re
import h5py as h5
import mrcfile
from skimage import io
import numpy as np
from loguru import logger
from qtpy.QtWidgets import QFileDialog, QGridLayout, QGroupBox, QLabel
import os

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from paramiko.client import SSHClient, WarningPolicy
from paramiko.ssh_exception import AuthenticationException
import socket
import h5py as h5

import yaml
from skimage import io
import mrcfile
import time
from datetime import date
from loguru import logger
from numpy import clip, product
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


from survos2.frontend.main import init_ws
from survos2.frontend.utils import ComboDialog, FileWidget, MplCanvas
from survos2.model.workspace import WorkspaceException
from survos2.frontend.plugins.base import ComboBox

from survos2.config import Config


from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotations import LevelComboBox as AnnoComboBox
from survos2.frontend.plugins.base import (
    ComboBox,
    LazyComboBox,
    Plugin,
    VBox,
    register_plugin,
)
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from napari.qt.progress import progress

CHROOT = Config["model.chroot"]


FILE_TYPES = ["HDF5", "MRC", "TIFF"]
HDF_EXT = ".h5"
MRC_EXT = ".rec"
TIFF_EXT = ".tiff"


LOAD_DATA_EXT = "*.h5 *.hdf5 *.tif *.tiff *.rec *.mrc"


class LoadDataDialog(QDialog):
    """Dialog box that contains a data preview for a 3d HDF5 dataset.

    Preview window allows selection of a ROI using a mouse or manual input.
    An estimated data size is calculated based upon ROI size and downsampling factor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_limits = None
        self.roi_changed = False

        self.setWindowTitle("Select Data to Load")
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setLayout(main_layout)
        container = QWidget(self)
        hbox = QHBoxLayout(container)
        container.setMaximumWidth(950)
        container.setMaximumHeight(530)
        container.setLayout(hbox)
        container.setObjectName("loaderContainer")
        container.setStyleSheet(
            "QWidget#loaderContainer {"
            "  background-color: #1e1e1e; "
            "  border-radius: 10px;"
            "}"
        )
        lvbox = QVBoxLayout()
        rvbox = QVBoxLayout()
        lvbox.setAlignment(Qt.AlignTop)
        rvbox.setAlignment(Qt.AlignTop)
        hbox.addLayout(lvbox, 1)
        hbox.addLayout(rvbox, 1)

        main_layout.addWidget(container)

        lvbox.addWidget(QLabel("Preview Dataset"))

        slider_vbox = self.setup_slider()
        lvbox.addLayout(slider_vbox)

        self.canvas = MplCanvas()
        self.canvas.roi_updated.connect(self.on_roi_box_update)
        lvbox.addWidget(self.canvas)

        # INPUT
        rvbox.addWidget(QLabel("Input Dataset:"))
        self.winput = FileWidget(extensions=LOAD_DATA_EXT, save=False)
        rvbox.addWidget(self.winput)
        rvbox.addWidget(QLabel("Internal HDF5 data path:"))
        self.int_h5_pth = QLabel("None selected")
        rvbox.addWidget(self.int_h5_pth)
        roi_fields = self.setup_roi_fields()
        rvbox.addWidget(QWidget(), 1)
        rvbox.addWidget(roi_fields)

        # Save | Cancel
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        rvbox.addWidget(self.buttonBox)
        self.winput.path_updated.connect(self.load_data)
        self.slider.sliderReleased.connect(self.update_image)
        self.slider.valueChanged.connect(self.update_slider_z_label)

    def setup_roi_fields(self):
        """Setup the dialog fields associated with ROI selection.

        Returns:
            PyQt5.QWidgets.QGroupBox: The GroupBox containing the fields.
        """
        apply_roi_button = QPushButton("Apply ROI")
        reset_button = QPushButton("Reset ROI")
        roi_fields = QGroupBox("Select Region of Interest:")
        roi_layout = QGridLayout()
        roi_layout.addWidget(
            QLabel("Drag a box in the image window or type manually"), 0, 0, 1, 3
        )
        roi_layout.addWidget(QLabel("Axis"), 1, 0)
        roi_layout.addWidget(QLabel("Start Value:"), 1, 1)
        roi_layout.addWidget(QLabel("End Value:"), 1, 2)
        roi_layout.addWidget(apply_roi_button, 1, 3)
        roi_layout.addWidget(reset_button, 2, 3)
        roi_layout.addWidget(QLabel("x:"), 2, 0)
        self.xstart_linedt = QLineEdit("0")
        self.xstart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.xstart_linedt, 2, 1)
        self.xend_linedt = QLineEdit("0")
        self.xend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.xend_linedt, 2, 2)
        roi_layout.addWidget(QLabel("y:"), 3, 0)
        self.ystart_linedt = QLineEdit("0")
        self.ystart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.ystart_linedt, 3, 1)
        self.yend_linedt = QLineEdit("0")
        self.yend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.yend_linedt, 3, 2)
        roi_layout.addWidget(QLabel("z:"), 4, 0)
        self.zstart_linedt = QLineEdit("0")
        self.zstart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.zstart_linedt, 4, 1)
        self.zend_linedt = QLineEdit("0")
        self.zend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.zend_linedt, 4, 2)
        roi_layout.addWidget(QLabel("Downsample Factor:"), 5, 0)
        self.downsample_spinner = QSpinBox()
        self.downsample_spinner.setRange(1, 10)
        self.downsample_spinner.setSpecialValueText("None")
        self.downsample_spinner.setMaximumWidth(60)
        self.downsample_spinner.valueChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.downsample_spinner, 5, 1)
        roi_layout.addWidget(QLabel("Estimated datasize (MB):"), 5, 3)
        self.data_size_label = QLabel("0")
        roi_layout.addWidget(self.data_size_label, 5, 4)
        roi_fields.setLayout(roi_layout)
        apply_roi_button.clicked.connect(self.on_roi_apply_clicked)
        reset_button.clicked.connect(self.on_roi_reset_clicked)
        return roi_fields

    def setup_slider(self):
        """Creates a horizontal slider in a VBoX with labels showing max and min values.

        Returns:
            PyQt5.QWidgets.QVBoxLayout: A QVBoXLayout containing the slider and labels.
        """
        self.slider = QSlider(1)
        slider_vbox = QVBoxLayout()
        slider_hbox = QHBoxLayout()
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setSpacing(0)
        self.slider_min_label = QLabel(alignment=Qt.AlignLeft)
        self.slider_z_label = QLabel(alignment=Qt.AlignCenter)
        self.slider_max_label = QLabel(alignment=Qt.AlignRight)
        slider_vbox.addWidget(self.slider)
        slider_vbox.addLayout(slider_hbox)
        slider_hbox.addWidget(self.slider_min_label, Qt.AlignLeft)
        slider_hbox.addWidget(self.slider_z_label, Qt.AlignCenter)
        slider_hbox.addWidget(self.slider_max_label, Qt.AlignRight)
        slider_vbox.addStretch()
        return slider_vbox

    @pyqtSlot()
    def update_slider_z_label(self):
        """Changes Z value label when slider moved."""
        idx = self.sender().value()
        self.slider_z_label.setNum(idx)

    @pyqtSlot()
    def on_roi_reset_clicked(self):
        """Resets data preview and ROI fields when reset button clicked."""
        self.data_limits = None
        self.reset_roi_fields()
        self.update_image(load=True)

    @pyqtSlot()
    def on_roi_apply_clicked(self):
        """Updates data preview window to fit new ROI when 'Apply' button clicked."""
        self.data_limits = self.get_roi_limits()
        self.roi_changed = self.check_if_roi_changed(self.data_limits)
        self.update_image()

    @pyqtSlot()
    def on_roi_param_changed(self):
        """Gets ROI limits and updates estimated data size whenever value is changed."""
        limits = self.get_roi_limits()
        x_start, x_end, y_start, y_end, z_start, z_end = self.clip_roi_box_vals(limits)
        x_size = x_end - x_start
        y_size = y_end - y_start
        z_size = z_end - z_start
        self.update_est_data_size(z_size, y_size, x_size)

    def get_roi_limits(self):
        """Reads the values of the ROI parameter fields.

        Returns:
            tuple: The six parameters x_start, x_end, y_start, y_end, z_start, z_end defining a 3d ROI
        """
        x_start = self.get_linedt_value(self.xstart_linedt)
        x_end = self.get_linedt_value(self.xend_linedt)
        y_start = self.get_linedt_value(self.ystart_linedt)
        y_end = self.get_linedt_value(self.yend_linedt)
        z_start = self.get_linedt_value(self.zstart_linedt)
        z_end = self.get_linedt_value(self.zend_linedt)
        return x_start, x_end, y_start, y_end, z_start, z_end

    def get_linedt_value(self, linedt):
        """Helper function that converts text in a LineEdit to int if it exists

        Args:
            linedt (PyQt5.QWidgets.LineEdit): A linedit widget to read.

        Returns:
            int: Value of text in LineEdt or 0
        """
        if linedt.text():
            return int(linedt.text())
        return 0

    def load_data(self, path):
        """Launches dialog box to select dataset from within HDF5 file and loads in data if selected.

        Args:
            path (str): Path to the HDF5 file.
        """
        if path is not None and len(path) > 0:
            dataset = None
            if path.endswith(".h5") or path.endswith(".hdf5"):
                available_hdf5 = self.available_hdf5_datasets(path)
                selected, accepted = ComboDialog.getOption(
                    available_hdf5, parent=self, title="HDF5: Select internal path"
                )
                if accepted == QDialog.Rejected:
                    return
                dataset = selected
                self.int_h5_pth.setText(selected)

            self.data = self.volread(path)
            self.dataset = dataset
            if isinstance(self.data, h5.Group):
                self.data_shape = self.data[self.dataset].shape
            else:
                self.data_shape = self.data.shape
            logger.info(self.data_shape)
            self.reset_roi_fields()
            self.update_image(load=True)

    def reset_roi_fields(self):
        """Resets all the ROI dimension parameters to equal the data shape."""
        self.xstart_linedt.setText("0")
        self.xend_linedt.setText(str(self.data_shape[2]))
        self.ystart_linedt.setText("0")
        self.yend_linedt.setText(str(self.data_shape[1]))
        self.zstart_linedt.setText("0")
        self.zend_linedt.setText(str(self.data_shape[0]))
        self.roi_changed = False

    def check_if_roi_changed(self, roi_limits):
        """Checks if any of the ROI dimension parameters are different from the data shape.

        Args:
            roi_limits (tuple): The six parameters x_start, x_end, y_start, y_end, z_start, z_end defining a 3d ROI

        Returns:
            bool: True if the ROI dimension parameters are different from the data shape.
        """
        x_start, x_end, y_start, y_end, z_start, z_end = roi_limits
        if not x_start == y_start == z_start == 0:
            return True
        if (
            (x_end != self.data_shape[2])
            or (y_end != self.data_shape[1])
            or (z_end != self.data_shape[0])
        ):
            return True
        return False

    def on_roi_box_update(self, size_tuple):
        """Updates ROI dimension parameters with data from ROI box drawn by dragging mouse on preview window.

        Args:
            size_tuple (tuple): Tuple of values received from ROI box, x_start, x_end, y_start, y_end
        """
        # Append the z values
        z_start = int(self.zstart_linedt.text())
        z_end = int(self.zend_linedt.text())
        size_tuple += (z_start, z_end)
        # Clip the values
        x_start, x_end, y_start, y_end, z_start, z_end = self.clip_roi_box_vals(
            size_tuple
        )
        self.xstart_linedt.setText(str(x_start))
        self.xend_linedt.setText(str(x_end))
        self.ystart_linedt.setText(str(y_start))
        self.yend_linedt.setText(str(y_end))
        self.zstart_linedt.setText(str(z_start))
        self.zend_linedt.setText(str(z_end))

    def clip_roi_box_vals(self, vals):
        """Clip ROI values to ensure that they lie within the data shape.

        Args:
            vals (tuple): Tuple of six ROI parameters x_start, x_end, y_start, y_end, z_start, z_end

        Returns:
            tuple: Tuple of six clipped ROI parameters x_start, x_end, y_start, y_end, z_start, z_end
        """
        x_start, x_end, y_start, y_end, z_start, z_end = map(round, vals)
        x_start, x_end = clip([x_start, x_end], 0, self.data_shape[2])
        y_start, y_end = clip([y_start, y_end], 0, self.data_shape[1])
        z_start, z_end = clip([z_start, z_end], 0, self.data_shape[0])
        return x_start, x_end, y_start, y_end, z_start, z_end

    def volread(self, path):
        """Helper to return a file handle to an HDF5 file.

        Args:
            path (str): Path to the HDF5 file.

        Raises:
            Exception: If the file is not and HDF5 file.

        Returns:
            h5py.File: File handle to an HDF5 file.
        """
        _, file_extension = os.path.splitext(path)
        data = None
        logger.info("Loading file handle")
        if file_extension in [".hdf5", ".h5"]:
            data = h5.File(path, "r")
        elif file_extension in [".tif", ".tiff"]:
            data = io.imread(path)
        elif file_extension in [".rec", ".mrc"]:
            mrc = mrcfile.mmap(path, mode="r+")
            data = mrc.data
        else:
            raise Exception("File format not supported")
        return data

    def scan_datasets_group(self, group, shape=None, dtype=None, path=""):
        """Recursive function that finds the datasets in an HDF5 file.

        Args:
            group (h5py.File or h5py.Group): A File Handle (Root Group) or Group.
            shape (tuple, optional): Specify datasets of a certain shape. Defaults to None.
            dtype (str, optional): Specify datasets of certain type. Defaults to None.
            path (str, optional): Internal HDF5 path. Defaults to ''.

        Returns:
            list: The datasets contained with the HDF5 file.
        """
        datasets = []
        for name, ds in group.items():
            curr_path = "{}/{}".format(path, name)
            if hasattr(ds, "shape"):
                if (
                    len(ds.shape) == 3
                    and (shape is None or ds.shape == shape)
                    and (dtype is None or ds.dtype == dtype)
                ):
                    datasets.append(curr_path)
            else:
                extra = self.scan_datasets_group(ds, shape=shape, path=curr_path)
                if len(extra) > 0:
                    datasets += extra
        return datasets

    def available_hdf5_datasets(self, path, shape=None, dtype=None):
        """Wrapper round the scan_datasets_gruop function."""
        datasets = []
        with h5.File(path, "r") as f:
            datasets = self.scan_datasets_group(f, shape=shape, dtype=dtype)
        return datasets

    @pyqtSlot()
    def update_image(self, load=False):
        """Updates the image in the preview window.

        Args:
            load (bool, optional): Set to True if loading a new dataset. Defaults to False.
        """
        # Only update index if called by slider
        if isinstance(self.sender(), QSlider):
            idx = self.sender().value()
        else:
            idx = None
        # Set limits
        if self.data_limits:
            x_start, x_end, y_start, y_end, z_start, z_end = self.data_limits
            x_size = x_end - x_start
            y_size = y_end - y_start
            z_size = z_end - z_start
        else:
            z_size, y_size, x_size = self.data_shape
            x_start, x_end, y_start, y_end, z_start, z_end = (
                0,
                x_size,
                0,
                y_size,
                0,
                z_size,
            )
        # Show central slice if loading data or changing roi
        if idx is None or load:
            idx = z_size // 2
            self.slider.blockSignals(True)
            self.slider.setMinimum(z_start)
            self.slider_min_label.setNum(z_start)
            self.slider.setMaximum(z_end - 1)
            self.slider_max_label.setNum(z_end)
            self.slider.setValue(idx)
            self.slider_z_label.setNum(idx)
            self.slider.blockSignals(False)
            self.canvas.ax.set_ylim([y_size + 1, -1])
            self.canvas.ax.set_xlim([-1, x_size + 1])
        if isinstance(self.data, h5.Group):
            img = self.data[self.dataset][idx]
        else:
            img = self.data[idx]
        self.canvas.ax.set_facecolor((1, 1, 1))
        self.canvas.ax.imshow(img[y_start:y_end, x_start:x_end], "gray")
        self.canvas.ax.grid(False)
        #self.canvas.redraw()

    def update_est_data_size(self, z_size, y_size, x_size):
        """Updates the estimated datasize label according to the dimensions and the downsampling factor.

        Args:
            z_size (int): Length of z dimension.
            y_size (int): Length of y dimension.
            x_size (int): Length of x dimension.
        """
        data_size_tup = tuple(map(int, (z_size, y_size, x_size)))
        est_data_size = (product(data_size_tup) * 4) / 10 ** 6
        est_data_size /= self.downsample_spinner.value()
        self.data_size_label.setText(f"{est_data_size:.2f}")


class SSHWorker(QObject):

    button_message_signal = pyqtSignal(list)
    update_ip_linedt_signal = pyqtSignal(str)
    error_signal = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, params, text, run_config):
        super().__init__()
        self.ssh_host, self.ssh_user, self.ssh_port = params
        self.text = text
        self.run_config = run_config

    @pyqtSlot()
    def start_server_over_ssh(self):
        try:
            client = SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(WarningPolicy())
            client.connect(
                self.ssh_host,
                port=self.ssh_port,
                username=self.ssh_user,
                password=self.text,
            )
            transport = client.get_transport()
            ip, _ = transport.getpeername()
            if ip:
                self.update_ip_linedt_signal.emit(ip)
                logger.info(f"IP for {self.ssh_host} detected as {ip}.")
            ws_name = self.run_config["workspace_name"]
            server_port = self.run_config["server_port"]
            # TODO Check if the server port is already in use
            logger.info(
                f"Checking if server port: {server_port} at ip: {ip} is already in use."
            )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((ip, int(server_port)))
            if result == 0:
                logger.error(f"Port {server_port} is already open.")
                self.button_message_signal.emit(
                    [f"Port: {server_port} at ip: {ip} is already in use!", "maroon", 3]
                )
                self.error_signal.emit()
                sock.close()
                client.close()
                return
            else:
                logger.info(f"Port {server_port} is not open.")
            sock.close()
            cuda_command = "module load cuda/10.1\n"
            command = (
                "/dls_sw/apps/SuRVoS2/s2_conda/bin/python -u "
                "/dls/science/groups/das/SuRVoS/s2/s2_dec/SuRVoS2/survos.py "
                f"start_server {ws_name} {server_port} > {date.today()}_survos2.log &\n"
            )
            logger.info(f"Running command on remote machine: {command}")

            session = transport.open_session()
            session.setblocking(0)  # Set to non-blocking mode
            session.get_pty()
            session.invoke_shell()
            # Send commands
            session.send(cuda_command)
            session.send(command)
            # Loop for 15 seconds
            self.button_message_signal.emit(
                [f"Starting server on {self.ssh_host}. Please Wait!", "navy", 14]
            )
            start = time.time()
            while time.time() - start < 15:
                if session.recv_ready():
                    data = session.recv(512)
                    print(data.decode(), flush=True)
                time.sleep(1)  # Yield CPU so we don't take up 100% usage...
            self.finished.emit()

        except AuthenticationException:
            logger.error("SSH Authentication failed!")
            self.button_message_signal.emit(["Incorrect Password!", "maroon", 3])
            self.error_signal.emit()


class FrontEndRunner(QWidget):
    """Main FrontEnd Runner window for creating workspace and starting SuRVoS2."""

    def __init__(self, run_config, workspace_config, pipeline_config, *args, **kwargs):
        super().__init__()


# if __name__ == "__main__":
#     signal.signal(signal.SIGINT, signal.SIG_DFL)


# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
# QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
# app = QApplication([])
# app.setOrganizationName("DLS")
# app.setApplicationName("SuRVoS2")
# front_end_runner = FrontEndRunner(run_config, workspace_config, pipeline_config)
# app.exec_()

# if front_end_runner.server_process is not None:
#     front_end_runner.server_process.kill()


@register_plugin
class ServerPlugin(Plugin):
    __icon__ = "fa.qrcode"
    __pname__ = "server"
    __views__ = ["slice_viewer"]
    __tab__ = "server"

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        run_config = {
            "server_ip": "127.0.0.1",
            "server_port": "8134",
            "workspace_name": "test_hunt_d4b",
            "use_ssh": False,
            "ssh_host": "ws168.diamond.ac.uk",
            "ssh_port": "22",
        }

        workspace_config = {
            "dataset_name": "data",
            "datasets_dir": "/path/to/my/data/dir",
            "vol_fname": "myfile.h5",
            "workspace_name": "my_survos_workspace",
            "downsample_by": "1",
        }

        from survos2.server.config import cfg

        pipeline_config = dict(cfg)

        self.run_config = run_config
        self.workspace_config = workspace_config
        self.pipeline_config = pipeline_config

        self.server_process = None
        self.client_process = None

        self.layout = QVBoxLayout()
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()

        tabwidget.addTab(tab1, "Setup and Start Survos")
        self.create_workspace_button = QPushButton("Create workspace")

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)

        chroot_fields = self.get_chroot_fields()
        tab1.layout.addWidget(chroot_fields)

        workspace_fields = self.get_workspace_fields()
        tab1.layout.addWidget(workspace_fields)

        self.setup_adv_run_fields()
        self.adv_run_fields.hide()

        run_fields = self.get_run_fields()
        tab1.layout.addWidget(run_fields)

        output_config_button = QPushButton("Save config")

        self.create_workspace_button.clicked.connect(self.create_workspace_clicked)
        output_config_button.clicked.connect(self.output_config_clicked)
        self.layout.addWidget(tabwidget)

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("SuRVoS Settings Editor")
        current_fpth = os.path.dirname(os.path.abspath(__file__))
        self.setWindowIcon(QIcon(os.path.join(current_fpth, "resources", "logo.png")))
        self.setLayout(self.layout)
        self.show()

    def get_chroot_fields(self):
        chroot_fields = QGroupBox("Set Dataset Root:")
        chroot_layout = QGridLayout()
        self.given_chroot_linedt = QLineEdit(CHROOT)
        chroot_layout.addWidget(self.given_chroot_linedt, 1, 0, 1, 2)

        set_chroot_button = QPushButton("Set")
        chroot_layout.addWidget(set_chroot_button, 1, 2)
        chroot_fields.setLayout(chroot_layout)

        set_chroot_button.clicked.connect(self.set_chroot)
        return chroot_fields

    def get_workspace_fields(self):
        """Gets the QGroupBox that contains all the fields for setting up the workspace.

        Returns:
            PyQt5.QWidgets.GroupBox: GroupBox with workspace fields.
        """
        select_data_button = QPushButton("Select")
        workspace_fields = QGroupBox("Create Workspace:")
        wf_layout = QGridLayout()
        wf_layout.addWidget(QLabel("Data File Path:"), 0, 0)
        current_data_path = Path(
            self.workspace_config["datasets_dir"], self.workspace_config["vol_fname"]
        )
        self.data_filepth_linedt = QLineEdit(str(current_data_path))
        wf_layout.addWidget(self.data_filepth_linedt, 1, 0, 1, 2)
        wf_layout.addWidget(select_data_button, 1, 2)
        wf_layout.addWidget(QLabel("HDF5 Internal Data Path:"), 2, 0, 1, 1)
        ws_dataset_name = self.workspace_config["dataset_name"]
        internal_h5_path = (
            ws_dataset_name
            if str(ws_dataset_name).startswith("/")
            else "/" + ws_dataset_name
        )
        self.h5_intpth_linedt = QLineEdit(internal_h5_path)
        wf_layout.addWidget(self.h5_intpth_linedt, 2, 1, 1, 1)
        wf_layout.addWidget(QLabel("Workspace Name:"), 3, 0)
        self.ws_name_linedt_1 = QLineEdit(self.workspace_config["workspace_name"])
        wf_layout.addWidget(self.ws_name_linedt_1, 3, 1)
        wf_layout.addWidget(QLabel("Downsample Factor:"), 4, 0)
        self.downsample_spinner = QSpinBox()
        self.downsample_spinner.setRange(1, 10)
        self.downsample_spinner.setSpecialValueText("None")
        self.downsample_spinner.setMaximumWidth(60)
        self.downsample_spinner.setValue(int(self.workspace_config["downsample_by"]))
        wf_layout.addWidget(self.downsample_spinner, 4, 1, 1, 1)
        # ROI
        self.setup_roi_fields()
        wf_layout.addWidget(self.roi_fields, 4, 2, 1, 2)
        self.roi_fields.hide()

        wf_layout.addWidget(self.create_workspace_button, 5, 0, 1, 3)
        workspace_fields.setLayout(wf_layout)
        select_data_button.clicked.connect(self.launch_data_loader)
        return workspace_fields

    def setup_roi_fields(self):
        """Sets up the QGroupBox that displays the ROI dimensions, if selected."""
        self.roi_fields = QGroupBox("ROI:")
        roi_fields_layout = QHBoxLayout()
        # z
        roi_fields_layout.addWidget(QLabel("z:"), 0)
        self.zstart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.zstart_roi_val, 1)
        roi_fields_layout.addWidget(QLabel("-"), 2)
        self.zend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.zend_roi_val, 3)
        # y
        roi_fields_layout.addWidget(QLabel("y:"), 4)
        self.ystart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.ystart_roi_val, 5)
        roi_fields_layout.addWidget(QLabel("-"), 6)
        self.yend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.yend_roi_val, 7)
        # x
        roi_fields_layout.addWidget(QLabel("x:"), 8)
        self.xstart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.xstart_roi_val, 9)
        roi_fields_layout.addWidget(QLabel("-"), 10)
        self.xend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.xend_roi_val, 11)

        self.roi_fields.setLayout(roi_fields_layout)

    def setup_adv_run_fields(self):
        """Sets up the QGroupBox that displays the advanced optiona for starting SuRVoS2."""
        self.adv_run_fields = QGroupBox("Advanced Run Settings:")
        adv_run_layout = QGridLayout()
        adv_run_layout.addWidget(QLabel("Server IP Address:"), 0, 0)
        self.server_ip_linedt = QLineEdit(self.run_config["server_ip"])
        adv_run_layout.addWidget(self.server_ip_linedt, 0, 1)
        adv_run_layout.addWidget(QLabel("Server Port:"), 1, 0)
        self.server_port_linedt = QLineEdit(self.run_config["server_port"])
        adv_run_layout.addWidget(self.server_port_linedt, 1, 1)
        
        # SSH Info
        self.ssh_button = QRadioButton("Use SSH")
        self.ssh_button.setAutoExclusive(False)
        adv_run_layout.addWidget(self.ssh_button, 0, 2)
        ssh_flag = self.run_config.get("use_ssh", False)
        if ssh_flag:
            self.ssh_button.setChecked(True)
        self.ssh_button.toggled.connect(self.toggle_ssh)

        self.adv_ssh_fields = QGroupBox("SSH Settings:")
        adv_ssh_layout = QGridLayout()
        adv_ssh_layout.setColumnStretch(2, 2)
        ssh_host_label = QLabel("Host")
        self.ssh_host_linedt = QLineEdit(self.run_config.get("ssh_host", ""))
        adv_ssh_layout.addWidget(ssh_host_label, 0, 0)
        adv_ssh_layout.addWidget(self.ssh_host_linedt, 0, 1, 1, 2)
        ssh_user_label = QLabel("Username")
        self.ssh_username_linedt = QLineEdit(self.get_login_username())
        adv_ssh_layout.addWidget(ssh_user_label, 1, 0)
        adv_ssh_layout.addWidget(self.ssh_username_linedt, 1, 1, 1, 2)
        ssh_port_label = QLabel("Port")
        self.ssh_port_linedt = QLineEdit(self.run_config.get("ssh_port", ""))
        adv_ssh_layout.addWidget(ssh_port_label, 2, 0)
        adv_ssh_layout.addWidget(self.ssh_port_linedt, 2, 1, 1, 2)
        self.adv_ssh_fields.setLayout(adv_ssh_layout)
        #adv_run_layout.addWidget(self.adv_ssh_fields, 1, 2, 2, 5)
        
        self.adv_run_fields.setLayout(adv_run_layout)

    def get_run_fields(self):
        """Gets the QGroupBox that contains the fields for starting SuRVoS.

        Returns:
            PyQt5.QWidgets.GroupBox: GroupBox with run fields.
        """
        self.run_button = QPushButton("Start Server")
        self.stop_button = QPushButton("Stop Server")

        self.existing_button = QPushButton("Use Existing Server")


        advanced_button = QRadioButton("Advanced")

        run_fields = QGroupBox("Run SuRVoS:")
        run_layout = QGridLayout()

        workspaces = os.listdir(CHROOT)
        self.workspaces_list = ComboBox()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)

        run_layout.addWidget(QLabel("Workspace Name:"), 0, 0)
        self.ws_name_linedt_2 = QLineEdit(self.workspace_config["workspace_name"])
        self.ws_name_linedt_2.setAlignment(Qt.AlignLeft)
        self.workspaces_list.setLineEdit(self.ws_name_linedt_2)

        # run_layout.addWidget(self.ws_name_linedt_2, 0, 1)

        run_layout.addWidget(self.workspaces_list, 0, 1)
        run_layout.addWidget(advanced_button, 1, 0)
        run_layout.addWidget(self.adv_run_fields, 2, 1)
        run_layout.addWidget(self.run_button, 3, 0, 1, 3)
        run_layout.addWidget(self.stop_button, 4, 0, 1, 3)
        run_layout.addWidget(self.existing_button, 5, 0, 1, 3)
        run_fields.setLayout(run_layout)

        advanced_button.toggled.connect(self.toggle_advanced)
        self.run_button.clicked.connect(self.run_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.existing_button.clicked.connect(self.existing_clicked)

        return run_fields

    def get_login_username(self):
        try:
            user = getpass.getuser()
        except Exception:
            user = ""
        return user


    def refresh_chroot(self):
        workspaces = os.listdir(DataModel.g.CHROOT)
        self.workspaces_list.clear()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)

    @pyqtSlot()
    def set_chroot(self):
        CHROOT = self.given_chroot_linedt.text()
        Config.update({"model": {"chroot": CHROOT}})
        logger.debug(f"Setting CHROOT to {CHROOT}")
        DataModel.g.CHROOT = CHROOT
        self.refresh_chroot()

    @pyqtSlot()
    def launch_data_loader(self):
        """Load the dialog box widget to select data with data preview window and ROI selection."""
        path = None
        int_h5_pth = None
        dialog = LoadDataDialog(self)
        result = dialog.exec_()
        self.roi_limits = None
        if result == QDialog.Accepted:
            path = dialog.winput.path.text()
            int_h5_pth = dialog.int_h5_pth.text()
            down_factor = dialog.downsample_spinner.value()
        if path and int_h5_pth:
            self.data_filepth_linedt.setText(path)
            self.h5_intpth_linedt.setText(int_h5_pth)
            self.downsample_spinner.setValue(down_factor)
            if dialog.roi_changed:
                self.roi_limits = tuple(map(str, dialog.get_roi_limits()))
                self.roi_fields.show()
                self.update_roi_fields_from_dialog()
            else:
                self.roi_fields.hide()

    def update_roi_fields_from_dialog(self):
        """Updates the ROI fields in the main window."""
        x_start, x_end, y_start, y_end, z_start, z_end = self.roi_limits
        self.xstart_roi_val.setText(x_start)
        self.xend_roi_val.setText(x_end)
        self.ystart_roi_val.setText(y_start)
        self.yend_roi_val.setText(y_end)
        self.zstart_roi_val.setText(z_start)
        self.zend_roi_val.setText(z_end)

    @pyqtSlot()
    def toggle_advanced(self):
        """Controls displaying/hiding the advanced run fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_run_fields.show()
        else:
            self.adv_run_fields.hide()

    @pyqtSlot()
    def toggle_ssh(self):
        """Controls displaying/hiding the SSH fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_ssh_fields.show()
        else:
            self.adv_ssh_fields.hide()

    @pyqtSlot()
    def create_workspace_clicked(self):
        """Performs checks and coordinates workspace creation on button press."""
        logger.debug("Creating workspace: ")
        # Set the path to the data file
        vol_path = Path(self.data_filepth_linedt.text())
        if not vol_path.is_file():
            err_str = f"No data file exists at {vol_path}!"
            logger.error(err_str)
            self.button_feedback_response(
                err_str, self.create_workspace_button, "maroon"
            )
        else:
            self.workspace_config["datasets_dir"] = str(vol_path.parent)
            self.workspace_config["vol_fname"] = str(vol_path.name)
            dataset_name = self.h5_intpth_linedt.text()
            self.workspace_config["dataset_name"] = str(dataset_name).strip("/")
            # Set the workspace name
            ws_name = self.ws_name_linedt_1.text()
            self.workspace_config["workspace_name"] = ws_name
            # Set the downsample factor
            ds_factor = self.downsample_spinner.value()
            self.workspace_config["downsample_by"] = ds_factor
            # Set the ROI limits if they exist
            if self.roi_limits:
                self.workspace_config["roi_limits"] = self.roi_limits
            try:
                response = init_ws(self.workspace_config)
                _, error = response
                if not error:
                    self.button_feedback_response(
                        "Workspace created sucessfully",
                        self.create_workspace_button,
                        "green",
                    )
                    # Update the workspace name in the 'Run' section
                    self.ws_name_linedt_2.setText(self.ws_name_linedt_1.text())
            except WorkspaceException as e:
                logger.exception(e)
                self.button_feedback_response(
                    str(e), self.create_workspace_button, "maroon"
                )
            self.refresh_chroot()

    def button_feedback_response(self, message, button, colour_str, timeout=2):
        """Changes button colour and displays feedback message for a limited time period.

        Args:
            message (str): Message to display in button.
            button (PyQt5.QWidgets.QBushButton): The button to manipulate.
            colour_str (str): The standard CSS colour string or hex code describing the colour to change the button to.
        """
        timeout *= 1000
        msg_old = button.text()
        col_old = button.palette().button().color
        txt_col_old = button.palette().buttonText().color
        button.setText(message)
        button.setStyleSheet(f"background-color: {colour_str}; color: white")
        timer = QTimer()
        timer.singleShot(
            timeout, lambda: self.reset_button(button, msg_old, col_old, txt_col_old)
        )

    @pyqtSlot()
    def reset_button(self, button, msg_old, col_old, txt_col_old):
        """Sets a button back to its original display settings.

        Args:
            button (PyQt5.QWidgets.QBushButton): The button to manipulate.
            msg_old (str): Message to display in button.
            col_old (str): The standard CSS colour string or hex code describing the colour to change the button to.
            txt_col_old (str): The standard CSS colour string or hex code describing the colour to change the button text to.
        """
        button.setStyleSheet(f"background-color: {col_old().name()}")
        button.setStyleSheet(f"color: {txt_col_old().name()}")
        button.setText(msg_old)
        button.update()

    @pyqtSlot()
    def output_config_clicked(self):
        """Outputs pipeline config YAML file on button click."""
        out_fname = "pipeline_cfg.yml"
        logger.debug(f"Outputting pipeline config: {out_fname}")
        with open(out_fname, "w") as outfile:
            yaml.dump(
                self.pipeline_config, outfile, default_flow_style=False, sort_keys=False
            )

    def get_ssh_params(self):
        ssh_host = self.ssh_host_linedt.text()
        ssh_user = self.ssh_username_linedt.text()
        ssh_port = int(self.ssh_port_linedt.text())
        return ssh_host, ssh_user, ssh_port

    def start_server_over_ssh(self):
        params = self.get_ssh_params()
        if not all(params):
            logger.error("Not all SSH parameters given! Not connecting to SSH.")
            pass
        ssh_host, ssh_user, ssh_port = params
        # Pop up dialog to ask for password
        text, ok = QInputDialog.getText(
            None, "Login", f"Password for {ssh_user}@{ssh_host}", QLineEdit.Password
        )
        if ok and text:
            self.ssh_worker = SSHWorker(params, text, self.run_config)
            self.ssh_thread = QThread(self)
            self.ssh_worker.moveToThread(self.ssh_thread)
            self.ssh_worker.button_message_signal.connect(self.send_msg_to_run_button)
            self.ssh_worker.error_signal.connect(self.on_ssh_error)
            self.ssh_worker.finished.connect(self.start_client)
            self.ssh_worker.update_ip_linedt_signal.connect(self.update_ip_linedt)
            self.ssh_thread.started.connect(self.ssh_worker.start_server_over_ssh)
            self.ssh_thread.start()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Quit",
            "Are you sure you want to quit? " "The server will be stopped.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    @pyqtSlot()
    def on_ssh_error(self):
        self.ssh_error = True

    @pyqtSlot(str)
    def update_ip_linedt(self, ip):
        self.server_ip_linedt.setText(ip)

    @pyqtSlot(list)
    def send_msg_to_run_button(self, param_list):
        self.button_feedback_response(
            param_list[0], self.run_button, param_list[1], param_list[2]
        )

    @pyqtSlot()
    def stop_clicked(self):
        logger.debug("Stopping server")
        if self.server_process is not None:
            self.server_process.kill()

    @pyqtSlot()
    def run_clicked(self):
        """Starts SuRVoS2 server and client as subprocesses when 'Run' button pressed.

        Raises:
            Exception: If survos.py not found.
        """
        with progress(total=3) as pbar:
            pbar.set_description("Starting server...")
            pbar.update(1)

        self.ssh_error = (
            False  # Flag which will be set to True if there is an SSH error
        )
        command_dir = os.path.abspath(os.path.dirname(__file__))  # os.getcwd()

        # Set current dir to survos root
        from pathlib import Path

        command_dir = Path(command_dir).absolute().parent.parent.parent.resolve()
        os.chdir(command_dir)

        self.script_fullname = os.path.join(command_dir, "survos.py")
        if not os.path.isfile(self.script_fullname):
            raise Exception("{}: Script not found".format(self.script_fullname))
        # Retrieve the parameters from the fields TODO: Put some error checking in
        self.run_config["workspace_name"] = self.ws_name_linedt_2.text()
        self.run_config["server_port"] = self.server_port_linedt.text()
        # Temporary measure to check whether the workspace exists or not
        full_ws_path = os.path.join(
            Config["model.chroot"], self.run_config["workspace_name"]
        )
        if not os.path.isdir(full_ws_path):
            logger.error(
                f"No workspace can be found at {full_ws_path}, Not starting SuRVoS."
            )
            self.button_feedback_response(
                f"Workspace {self.run_config['workspace_name']} does not appear to exist!",
                self.run_button,
                "maroon",
            )
            return
        pbar.update(1)
        # Try some fancy SSH stuff here
        if self.ssh_button.isChecked():
            self.start_server_over_ssh()
        else:
            self.server_process = subprocess.Popen(
                [
                    "python",
                    self.script_fullname,
                    "start_server",
                    self.run_config["workspace_name"],
                    self.run_config["server_port"],
                    DataModel.g.CHROOT,
                ]
            )
            try:
                outs, errs = self.server_process.communicate(timeout=10)
                print(f"OUTS: {outs, errs}")
            except subprocess.TimeoutExpired:
                pass

            # self.start_client()
            logger.info(f"setting remote: {self.server_port_linedt.text()}")
            remote_ip_port = "127.0.0.1:" + self.server_port_linedt.text()
            logger.info(f"setting remote: {remote_ip_port}")
            resp = Launcher.g.set_remote(remote_ip_port)
            logger.info(f"Response from server to setting remote: {resp}")

            cfg.ppw.clientEvent.emit(
                {
                    "source": "server_tab",
                    "data": "set_workspace",
                    "workspace": self.ws_name_linedt_2.text(),
                }
            )
            cfg.ppw.clientEvent.emit(
                {"source": "panel_gui", "data": "refresh", "value": None}
            )
            #cfg.ppw.clientEvent.emit({'data' : 'view_feature', 'feature_id' : '001_raw'})
        pbar.update(1)
    @pyqtSlot()
    def existing_clicked(self):
        ssh_ip = self.server_ip_linedt.text()
        remote_ip_port = ssh_ip + ":" + self.server_port_linedt.text()
        logger.info(f"setting remote: {remote_ip_port}")
        resp = Launcher.g.set_remote(remote_ip_port)
        logger.info(f"Response from server to setting remote: {resp}")

        cfg.ppw.clientEvent.emit(
            {
                "source": "server_tab",
                "data": "set_workspace",
                "workspace": self.ws_name_linedt_2.text(),
            }
        )
        cfg.ppw.clientEvent.emit(
            {"source": "panel_gui", "data": "refresh", "value": None}
        )

    def start_client(self):
        if not self.ssh_error:
            self.button_feedback_response(
                "Starting Client.", self.run_button, "green", 7
            )
            self.run_config["server_ip"] = self.server_ip_linedt.text()
            self.client_process = subprocess.Popen(
                [
                    "python",
                    self.script_fullname,
                    "nu_gui",
                    self.run_config["workspace_name"],
                    str(self.run_config["server_ip"])
                    + ":"
                    + str(self.run_config["server_port"]),
                ]
            )
