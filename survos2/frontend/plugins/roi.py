import re
import h5py as h5
import mrcfile
from skimage import io
import numpy as np
from loguru import logger
from qtpy.QtWidgets import QFileDialog, QGridLayout, QGroupBox, QLabel
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
from survos2.api.objects import get_entities
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import (
    ComboBox,
    LazyComboBox,
    Plugin,
    VBox,
    register_plugin,
)
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg

from survos2.frontend.components.base import *
from survos2.frontend.components.base import HWidgets, Slider
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.annotation_tool import AnnotationComboBox
from survos2.frontend.plugins.base import ComboBox
#from survos2.frontend.plugins.features import *
#from survos2.frontend.plugins.superregions import *
from survos2.frontend.utils import FileWidget
from survos2.server.state import cfg
from survos2.model.model import DataModel
from survos2.frontend.utils import ComboDialog, FileWidget, MplCanvas
from survos2.utils import decode_numpy
from survos2.frontend.plugins.objects import ObjectComboBox

FILE_TYPES = ["HDF5", "MRC", "TIFF"]
HDF_EXT = ".h5"
MRC_EXT = ".rec"
TIFF_EXT = ".tiff"


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
            "  background-color: #4e4e4e; "
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

        roi_fields = self.setup_roi_fields()
        rvbox.addWidget(QWidget(), 1)
        rvbox.addWidget(roi_fields)

        # Save | Cancel
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        rvbox.addWidget(self.buttonBox)
        #self.winput.path_updated.connect(self.load_data)
        self.slider.sliderReleased.connect(self.update_image)
        self.slider.valueChanged.connect(self.update_slider_z_label)

    def set_data(self, img_arr):
        self.data = img_arr

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
        self.canvas.redraw()

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

    def load_data(self):
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
        self.canvas.redraw()

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
        est_data_size /= self.downsample_spinner.value() ** 3
        self.data_size_label.setText(f"{est_data_size:.2f}")



@register_plugin
class ROIPlugin(Plugin):
    __icon__ = "fa.qrcode"
    __pname__ = "roi"
    __views__ = ["slice_viewer"]
    __tab__ = "roi"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vbox = VBox(self, spacing=10)
        hbox_layout3 = QtWidgets.QHBoxLayout()
        self.roi_layout = QtWidgets.QVBoxLayout()
        
        self._add_feature_source()
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)
        widget = HWidgets("Annotation to copy:", self.annotations_source, stretch=1)
        self.vbox.addWidget(widget)
        button_selectroi = QPushButton("Select ROI", self)
        self.vbox.addWidget(button_selectroi)
        button_selectroi.clicked.connect(self._launch_data_loader)
        self._add_boxes_source()

        #self.vbox.addLayout(self.roi_layout)
        self.vbox.addLayout(hbox_layout3)
        self.existing_roi = {}
        self.roi_layout = VBox(margin=0, spacing=5)
        self.vbox.addLayout(self.roi_layout)

    def _add_boxes_source(self):
        self.boxes_source = ObjectComboBox(full=True, filter=["boxes"])
        self.boxes_source.fill()
        self.boxes_source.setMaximumWidth(250)
        button_add_boxes_as_roi = QPushButton("Add Boxes as ROI", self)
        button_add_boxes_as_roi.clicked.connect(self.add_rois)
        widget = HWidgets("Select Boxes:", self.boxes_source,  button_add_boxes_as_roi, stretch=1)
        self.vbox.addWidget(widget)

    def _add_feature_source(self, label="Feature:"):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source,  stretch=1)
        self.vbox.addWidget(widget)

    def _select_feature(self, feature_id):
        features_src = DataModel.g.dataset_uri(feature_id, group="features")
        params = dict(
            workpace=True,
            src=features_src,
        )

        result = Launcher.g.run("features", "get_volume", **params)
        if result:
            feature_arr = decode_numpy(result)
            return feature_arr

    def _launch_data_loader(self):
        """Load the dialog box widget for ROI selection."""
        path = None
        int_h5_pth = None
        feature_id = self.feature_source.value()
        feature_arr = self._select_feature(feature_id)
        dialog = LoadDataDialog()
        dialog.set_data(feature_arr)
        dialog.load_data()
        result = dialog.exec_()
        self.roi_limits = None
        if result == QDialog.Accepted:
            y_st, y_end, x_st, x_end,  z_st, z_end = dialog.get_roi_limits()
            roi = [z_st, z_end, x_st, x_end, y_st, y_end]
            original_workspace = DataModel.g.current_workspace 
            roi_name = (
                    DataModel.g.current_workspace
                    + "_roi_"
                    + str(z_st)
                    + "_"
                    + str(z_end)
                    + "_"
                    + str(x_st)
                    + "_"
                    + str(x_end)
                    + "_"
                    + str(y_st)
                    + "_"
                    + str(y_end)
            )
            
            cfg.ppw.clientEvent.emit(
            {"source": "panel_gui", "data": "make_roi_ws", "roi": roi, "feature_id":feature_id}
            )
            self.add_roi(roi_name, original_workspace, roi)

    def add_rois(self, rois):
        """Adds ROIs to the ROI list.
        Args:
            rois (list): List of ROIs.
        """
        # Load objects 
        objects_id = str(self.boxes_source.value().rsplit("/", 1)[-1])
        #str(self.boxes_source.value())
        
        logger.debug(f"Get objects {objects_id}")
        objects_src = DataModel.g.dataset_uri(objects_id, group="objects")
        params = dict(workpace=True,src=objects_src, entity_type="boxes")
        
        result = Launcher.g.run("objects", "get_entities", **params)
        
        if result:
            entities_arr = decode_numpy(result)
        
        rois = entities_arr[:,4:]
        print(rois)

        # Iterate through ROIs and add them to the ROI list
        original_workspace = DataModel.g.current_workspace 

        for roi in rois:
            roi_list = list(roi)
            roi_list = [int(el) for el in roi_list]    
            # reorder to z_st, z_end, x_st, x_end, y_st, y_end
            roi_list = [roi_list[0], roi_list[3], roi_list[1], roi_list[4], roi_list[2], roi_list[5]]

            roi_name = (
                    DataModel.g.current_workspace
                    + "_roi_"
                    + str(roi[0])
                    + "_"
                    + str(roi[3])
                    + "_"
                    + str(roi[1])
                    + "_"
                    + str(roi[4])
                    + "_"
                    + str(roi[2])
                    + "_"
                    + str(roi[5])
            )
            cfg.ppw.clientEvent.emit(
            {"source": "panel_gui", "data": "make_roi_ws", "roi": roi_list}
            )   
            self.add_roi(roi_name, original_workspace, roi_list)


    def setup(self):
        result = Launcher.g.run("roi", "existing")
        logger.debug(f"roi result {result}")
        if result:
            for k,v in result.items():
                self._add_roi_widget(k,v)

    def _add_roi_widget(self, rid, rname, expand=False):
        widget = ROICard(rid, rname)
        widget.showContent(expand)
        self.roi_layout.addWidget(widget)
        self.existing_roi[rid] = widget
        return widget

    def add_roi(self, roi_fname, original_workspace, roi):
        """Adds a new ROI to the server-side ROI list.
        Checks to see if an annotation is provided and passes that if so.
        Refreshes the GUI after the command is complete.

        Parameters
        ----------
        roi_fname : String
            Name of the ROI in format name_[coords]
        original_workspace : String
            workspace name
        roi : List
            List of coordinates of the ROI in z_st, x_st, y_st, z_end, x_end, y_end format.
        """
        if self.annotations_source.value():
            original_level = str(self.annotations_source.value().rsplit("/", 1)[-1])
        else:
            original_level = None
        params = dict(workspace=original_workspace, roi_fname=roi_fname, roi=roi, original_workspace=original_workspace, original_level=original_level)
        result = Launcher.g.run("roi", "create", **params)
        if result:
            rid = result["id"]
            rname = result["name"]
            self._add_roi_widget(rid, rname, True)
        
        cfg.ppw.clientEvent.emit(
                {"source": "panel_gui", "data": "refresh", "value": None}
        )

    def clear(self):
        for region in list(self.existing_roi.keys()):
            self.existing_roi.pop(region).setParent(None)
        self.existing_roi = {}

    def button_setroi_clicked(self):
        """GUI handler for adding a new ROI.
        Grabs the coordinates of the ROI from the GUI and calls the add_roi method.
        """
        original_workspace = DataModel.g.current_workspace 
        roi_start = self.roi_start.value()
        roi_end = self.roi_end.value()
        roi = [
            roi_start[0],
            roi_start[1],
            roi_start[2],
            roi_end[0],
            roi_end[1],
            roi_end[2],
        ]

        roi_name = (
                DataModel.g.current_workspace
                + "_roi_"
                + str(roi[0])
                + "_"
                + str(roi[3])
                + "_"
                + str(roi[1])
                + "_"
                + str(roi[4])
                + "_"
                + str(roi[2])
                + "_"
                + str(roi[5])
        )
        
        cfg.ppw.clientEvent.emit(
           {"source": "panel_gui", "data": "make_roi_ws", "roi": roi}
        )
        self.add_roi(roi_name, original_workspace, roi)



class ROICard(Card):
    def __init__(self, rid, rname, parent=None):
        super().__init__(
            title=rname, collapsible=True, removable=True, editable=True, parent=parent
        )
        self.rname = rname
        self.rid = rid
        self.annotation_source = LevelComboBox(workspace=self.rname)
        self.annotation_target = LevelComboBox()
        self.pull_btn = PushButton("Pull into workspace")
        self.pull_btn.clicked.connect(self.pull_anno)
        self.add_row(HWidgets(None, self.annotation_source, self.annotation_target, self.pull_btn))

    def card_deleted(self):
        """Removes an ROI from the server-side ROI list.
        """
        logger.debug(f"Deleted ROI {self.rname}")
        params = dict(roi_fname=self.rname,workspace=True)
        result = Launcher.g.run("roi", "remove", **params)
        if result["done"]:
            self.setParent(None)

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited ROI title {newtitle}")
        
    def pull_anno(self):
        """Gui handler for the pull_anno command that grabs the annotation from the ROI's workspace
        and copies it into the current workspace.
        """
        anno_id = self.annotation_source.value().rsplit("/", 1)[-1]
        target_id = self.annotation_target.value().rsplit("/", 1)[-1]
        logger.debug(f"Pulling annotation into current workspace from workspace: {self.rname}, level: {anno_id}")
        all_params = dict(modal=True, roi_fname=self.rname, workspace=True, anno_id=anno_id, target_anno_id=target_id)
        result = Launcher.g.run("roi", "pull_anno", **all_params)




