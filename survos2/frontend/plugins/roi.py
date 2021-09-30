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

from survos2.frontend.components.base import *
from survos2.frontend.components.base import HWidgets, Slider
from survos2.frontend.plugins.annotations import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import ComboBox
from survos2.frontend.plugins.features import *
from survos2.frontend.plugins.superregions import *
from survos2.frontend.utils import FileWidget
from survos2.server.state import cfg
from survos2.model.model import DataModel


FILE_TYPES = ["HDF5", "MRC", "TIFF"]
HDF_EXT = ".h5"
MRC_EXT = ".rec"
TIFF_EXT = ".tiff"



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
        self.roi_start = LineEdit3D(default=0, parse=int)
        self.roi_end = LineEdit3D(default=0, parse=int)
        button_setroi = QPushButton("Save ROI as workspace", self)
        button_setroi.clicked.connect(self.button_setroi_clicked)
        self.roi_start_row = HWidgets("ROI Start:", self.roi_start)
        self.roi_end_row = HWidgets("Roi End: ", self.roi_end)
        self.roi_layout = QtWidgets.QVBoxLayout()
        self.roi_layout.addWidget(self.roi_start_row)
        self.roi_layout.addWidget(self.roi_end_row)
        self.roi_layout.addWidget(button_setroi)
        self.vbox.addLayout(self.roi_layout)
        self.vbox.addLayout(hbox_layout3)
        self.existing_roi = {}
        self.roi_layout = VBox(margin=0, spacing=5)
        self.vbox.addLayout(self.roi_layout)

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

    def add_roi(self, roi_fname, original_workspace):
        params = dict(workspace=original_workspace, roi_fname=roi_fname)
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
        #self.refresh_workspaces()
        cfg.ppw.clientEvent.emit(
            {"source": "panel_gui", "data": "make_roi_ws", "roi": roi}
        )
        self.add_roi(roi_name, original_workspace)



class ROICard(Card):
    def __init__(self, rid, rname, parent=None):
        super().__init__(
            title=rname, collapsible=True, removable=True, editable=True, parent=parent
        )
        self.rname = rname
        self.rid = rid
        self.pull_btn = PushButton("Pull into workspace")
        self.pull_btn.clicked.connect(self.pull_anno)
        self.add_row(HWidgets(None, self.pull_btn))

    def card_deleted(self):
        logger.debug(f"Deleted ROI {self.title}")
    
    def card_title_edited(self, newtitle):
        logger.debug(f"Edited ROI title {newtitle}")
        
    def pull_anno(self):
        all_params = dict(modal=True, roi_fname=self.rname, workspace=True)
        result = Launcher.g.run("roi", "pull_anno", **all_params)

