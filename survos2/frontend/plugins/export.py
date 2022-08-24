import re

import h5py as h5
import mrcfile
from skimage import io
import numpy as np
from loguru import logger
from qtpy.QtWidgets import QFileDialog, QGridLayout, QGroupBox, QLabel

from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotations import LevelComboBox as AnnoComboBox
from survos2.frontend.plugins.base import (
    Plugin,
    register_plugin,
)
from survos2.frontend.components.base import ComboBox, LazyComboBox, VBox
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg

FILE_TYPES = ["HDF5", "MRC", "TIFF"]
HDF_EXT = ".h5"
MRC_EXT = ".rec"
TIFF_EXT = ".tiff"


class SuperRegionSegmentComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run("pipelines", "existing", **params)
        logger.debug(f"Result of pipelines existing: {result}")
        if result:
            self.addCategory("Segmentations")
            for fid in result:
                if result[fid]["kind"] == "superregion_segment":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class ExportPlugin(Plugin):

    __icon__ = "fa.qrcode"
    __pname__ = "export"
    __views__ = ["slice_viewer"]
    __tab__ = "export"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vbox = VBox(self, spacing=10)
        feature_widgets = self.setup_feature_widgets()
        anno_widgets = self.setup_annotation_widgets()
        pipe_widgets = self.setup_pipeline_widgets()
        self.existing_supervoxels = {}
        self.vbox.addWidget(feature_widgets)
        self.vbox.addWidget(anno_widgets)
        self.vbox.addWidget(pipe_widgets)

    def setup_feature_widgets(self):
        feat_group_box = QGroupBox("Features:")
        feat_box_layout = QGridLayout()
        # Labels
        feat_box_layout.addWidget(QLabel("Feature"), 0, 0, 1, 2)
        feat_box_layout.addWidget(QLabel("File type"), 0, 2)
        # Features combo
        self.feat_source = FeatureComboBox()
        feat_box_layout.addWidget(self.feat_source, 1, 0, 1, 2)
        # File type combo
        self.feat_ftype_combo = ComboBox()
        self.add_filetypes_to_combo(self.feat_ftype_combo)
        feat_box_layout.addWidget(self.feat_ftype_combo, 1, 2)
        # Button
        self.feat_export_btn = IconButton("fa.save", "Export data", accent=True)
        self.feat_export_btn.clicked.connect(self.save_feature)
        feat_box_layout.addWidget(self.feat_export_btn, 1, 3)

        feat_group_box.setLayout(feat_box_layout)
        return feat_group_box

    def setup_annotation_widgets(self):
        anno_group_box = QGroupBox("Annotations:")
        anno_box_layout = QGridLayout()
        # Labels
        anno_box_layout.addWidget(QLabel("Annotation"), 0, 0, 1, 2)
        anno_box_layout.addWidget(QLabel("File type"), 0, 2)
        # Annotations combo
        self.anno_source = AnnoComboBox()
        anno_box_layout.addWidget(self.anno_source, 1, 0, 1, 2)
        # File type combo
        self.anno_ftype_combo = ComboBox()
        self.add_filetypes_to_combo(self.anno_ftype_combo)
        anno_box_layout.addWidget(self.anno_ftype_combo, 1, 2)
        # Button
        self.anno_export_btn = IconButton("fa.save", "Export data", accent=True)
        self.anno_export_btn.clicked.connect(self.save_anno)
        anno_box_layout.addWidget(self.anno_export_btn, 1, 3)

        anno_group_box.setLayout(anno_box_layout)
        return anno_group_box

    def setup_pipeline_widgets(self):
        pipe_group_box = QGroupBox("Pipeline output:")
        pipe_box_layout = QGridLayout()
        # Labels
        pipe_box_layout.addWidget(QLabel("Pipeline"), 0, 0, 1, 2)
        pipe_box_layout.addWidget(QLabel("File type"), 0, 2)
        # Pipeline combo
        self.pipe_source = SuperRegionSegmentComboBox()
        pipe_box_layout.addWidget(self.pipe_source, 1, 0, 1, 2)
        # File type combo
        self.pipe_ftype_combo = ComboBox()
        self.add_filetypes_to_combo(self.pipe_ftype_combo)
        pipe_box_layout.addWidget(self.pipe_ftype_combo, 1, 2)
        # Button
        self.pipe_export_btn = IconButton("fa.save", "Export data", accent=True)
        self.pipe_export_btn.clicked.connect(self.save_pipe)
        pipe_box_layout.addWidget(self.pipe_export_btn, 1, 3)

        pipe_group_box.setLayout(pipe_box_layout)
        return pipe_group_box

    def add_filetypes_to_combo(self, combo):
        for file_type in FILE_TYPES:
            combo.addItem(file_type)

    def save_feature(self):
        result = re.search(r"(\d+) (.*)", self.feat_source.currentText())
        if result:
            fid = result.group(1) + "_" + result.group(2).lower().replace(" ", "_")
            logger.info(f"Feature ID: {fid}")
            fname_filter, ext = self.get_data_filetype(self.feat_ftype_combo)
            filename = result.group(2).replace(" ", "") + ext
            path, _ = QFileDialog.getSaveFileName(self, "Save Feature", filename, fname_filter)
            if path is not None and len(path) > 0:
                feat_data = self.get_arr_data(fid, "features")
                self.save_data(feat_data, path, ext)
        else:
            logger.info("No feature selected")

    def save_anno(self):
        result = re.search(r"(\d+) (.*)", self.anno_source.currentText())
        if result:
            lid = result.group(1) + "_" + result.group(2).lower().replace(" ", "_")
            logger.info(f"Label ID: {lid}")
            fname_filter, ext = self.get_data_filetype(self.anno_ftype_combo)
            filename = lid + "Annotations" + ext
            path, _ = QFileDialog.getSaveFileName(self, "Save Annotation", filename, fname_filter)
            if path is not None and len(path) > 0:
                anno_data = self.get_arr_data(lid, "annotations")
                anno_data = anno_data & 15
                self.save_data(anno_data.astype(np.uint8), path, ext)
        else:
            logger.info("No annotation selected")

    def save_pipe(self):
        result = re.search(r"(\d+) (\S+) (\S+)", self.pipe_source.currentText())
        if result:
            pid = result.group(1) + "_" + result.group(2).lower() + "_" + result.group(3).lower()
            logger.info(f"Pipeline ID: {pid}")
            fname_filter, ext = self.get_data_filetype(self.pipe_ftype_combo)
            filename = pid + "Output" + ext
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Pipeline Output", filename, fname_filter
            )
            if path is not None and len(path) > 0:
                pipe_data = self.get_arr_data(pid, "pipelines")
                self.save_data(pipe_data.astype(np.uint8), path, ext)
        else:
            logger.info("No pipeline selected")

    def get_data_filetype(self, combo_box):
        ftype_map = {
            "HDF5": ("HDF5 (*.h5 *.hdf5)", HDF_EXT),
            "MRC": ("MRC (*.mrc *.rec *.st)", MRC_EXT),
            "TIFF": ("TIFF (*.tif *.tiff)", TIFF_EXT),
        }
        return ftype_map.get(combo_box.currentText())

    def get_arr_data(self, item_id, item_type):
        src = DataModel.g.dataset_uri(item_id, group=item_type)
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
        return src_arr

    def save_data(self, data, path, ext):
        if ext == HDF_EXT:
            logger.info(f"Saving data to {path} in HDF5 format")
            with h5.File(path, "w") as f:
                f["/data"] = data
        elif ext == MRC_EXT:
            logger.info(f"Saving data to {path} in MRC format")
            with mrcfile.new(path, overwrite=True) as mrc:
                mrc.set_data(data)
        elif ext == TIFF_EXT:
            logger.info(f"Saving data to {path} in TIFF format")
            io.imsave(path, data)
