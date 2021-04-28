import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import yaml
from loguru import logger
from numba import jit
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QCheckBox, QPushButton, QRadioButton
from scipy import ndimage
from skimage import img_as_float, img_as_ubyte, io
from vispy import scene
from vispy.color import Colormap

from survos2.frontend.components.base import *
from survos2.frontend.components.base import HWidgets, Slider, LineEdit3D
from survos2.frontend.plugins.annotations import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import ComboBox
from survos2.frontend.plugins.features import *
from survos2.frontend.plugins.regions import *
from survos2.frontend.utils import FileWidget
from survos2.server.state import cfg


class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.slice_mode = False

        self.slider = Slider(value=0, vmax=cfg.slice_max - 1)
        self.slider.setMinimumWidth(150)
        self.slider.sliderReleased.connect(self._params_updated)

        button1 = QPushButton("Refresh", self)
        button1.clicked.connect(self.button1_clicked)
        self.button2 = QPushButton("Slice mode", self)
        self.button2.clicked.connect(self.button2_clicked)

        self.filewidget = FileWidget(extensions="*.yaml", save=False)
        self.filewidget.path_updated.connect(self.load_workflow)

        button3 = QPushButton("Run workflow", self)
        button3.clicked.connect(self.button3_clicked)

        # self.session_list = ComboBox()
        # for s in cfg.sessions:
        #    self.session_list.addItem(key=s)
        # session_widget = HWidgets("Sessions:", self.session_list, Spacing(35), stretch=0)
        # self.session_list.activated[str].connect(self.sessions_selected)

        self.hbox_layout0 = QtWidgets.QHBoxLayout()
        hbox_layout1 = QtWidgets.QHBoxLayout()
        hbox_layout2 = QtWidgets.QHBoxLayout()
        hbox_layout3 = QtWidgets.QHBoxLayout()

        self.hbox_layout0.addWidget(self.slider)
        self.slider.hide()

        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=2)

        hbox_layout1.addWidget(button1)
        hbox_layout1.addWidget(self.button2)
        hbox_layout2.addWidget(self.filewidget)
        hbox_layout2.addWidget(button3)
        # hbox_layout3.addWidget(session_widget)

        vbox.addLayout(self.hbox_layout0)
        vbox.addLayout(hbox_layout1)
        vbox.addLayout(hbox_layout2)

        self.roi_start = LineEdit3D(default=0, parse=int)
        self.roi_end = LineEdit3D(default=0, parse=int)
        button_setroi= QPushButton("Set ROI", self)
        button_setroi.clicked.connect(self.button_setroi_clicked)
        self.roi_start_row = HWidgets("ROI Start:", self.roi_start)
        self.roi_end_row = HWidgets("Roi End: ", self.roi_end)

        self.roi_layout = QtWidgets.QVBoxLayout()
        self.roi_layout.addWidget(self.roi_start_row)
        self.roi_layout.addWidget(self.roi_end_row)
        self.roi_layout.addWidget(button_setroi)

        vbox.addLayout(self.roi_layout)

    def load_workflow(self, path):
        self.workflow_fullname = path
        print(f"Setting workflow fullname: {self.workflow_fullname}")

    def _params_updated(self):
        self.clientEvent.emit(
            {"source": "slider", "data": "jump_to_slice", "frame": self.slider.value()}
        )

    def sessions_selected(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "workspace_gui",
                "data": "set_session",
                "session": self.session_list.value(),
            }
        )

    def button_setroi_clicked(self):
        roi_start = self.roi_start.value()
        roi_end = self.roi_end.value()
        roi = [roi_start[0],roi_start[1], roi_start[2], roi_end[0], roi_end[1], roi_end[2]]
        cfg.ppw.clientEvent.emit(
            {"source": "workspace_gui", "data": "goto_roi", "roi": roi }
        )


    def button1_clicked(self):
        cfg.ppw.clientEvent.emit(
            {"source": "workspace_gui", "data": "refresh", "value": None}
        )

    def button2_clicked(self):
        if self.slice_mode:
            self.slider.hide()
            self.button2.setText("Slice Mode")
        else:
            self.slider.vmax = cfg.slice_max - 1
            self.slider.setRange(0, cfg.slice_max - 1)
            self.slider.show()
            self.button2.setText("Volume Mode")
        self.slice_mode = not self.slice_mode

        self.clientEvent.emit(
            {"source": "button2", "data": "slice_mode",}
        )

    def button3_clicked(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "workspace_gui",
                "data": "run_workflow",
                "workflow_file": self.workflow_fullname,
            }
        )


class PluginPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.pluginContainer = PluginContainer()

        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=5)
        vbox.addWidget(self.pluginContainer)
        self.setLayout(vbox)

        for plugin_name in list_plugins():
            plugin = get_plugin(plugin_name)
            name = plugin["name"]
            title = plugin["title"]
            plugin_cls = plugin["cls"]  # full classname
            tab = plugin["tab"]

            logger.debug(f"Plugin loaded: {name}, {title}, {plugin_cls}")
            self.pluginContainer.load_plugin(name, title, plugin_cls)
            self.pluginContainer.show_plugin(name, tab)

        logger.debug(f"Plugins loaded: {list_plugins()}")

    def setup(self):
        for plugin_name in list_plugins():
            plugin = get_plugin(plugin_name)
            name = plugin["name"]
            title = plugin["title"]
            plugin_cls = plugin["cls"]  # full classname
            tab = plugin["tab"]
            self.pluginContainer.show_plugin(name, tab)


class QtPlotWidget(QtWidgets.QWidget):
    def __init__(self):

        super().__init__()

        self.canvas = scene.SceneCanvas(bgcolor="k", keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.canvas.native)

        _ = scene.visuals.Line(
            pos=np.array([[0, 0], [700, 500]]), color="w", parent=self.canvas.scene,
        )
