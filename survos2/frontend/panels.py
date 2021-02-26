import numpy as np
import pandas as pd
from numba import jit
import scipy
import yaml
from loguru import logger


from scipy import ndimage
from skimage import img_as_ubyte, img_as_float
from skimage import io
import seaborn as sns

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QCheckBox
from vispy import scene
from vispy.color import Colormap
from survos2.frontend.components.base import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.regions import *
from survos2.frontend.plugins.features import *
from survos2.frontend.plugins.annotations import *
from survos2.frontend.plugins.base import ComboBox
from survos2.frontend.components.base import Slider, HWidgets
from survos2.server.state import cfg


class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.slice_mode = False

        self.slider = Slider(value=0, vmax=cfg.slice_max - 1)
        self.slider.setMinimumWidth(150)
        self.slider.sliderReleased.connect(self._params_updated)

        #button1 = QPushButton("Reload", self)
        #button1.clicked.connect(self.button1_clicked)
        self.button2 = QPushButton("Slice mode", self)
        self.button2.clicked.connect(self.button2_clicked)
        
        #button3 = QPushButton("Run workflow", self)
        #button3.clicked.connect(self.button3_clicked)

        #self.session_list = ComboBox()
        #for s in cfg.sessions:
        #    self.session_list.addItem(key=s)
        #session_widget = HWidgets("Sessions:", self.session_list, Spacing(35), stretch=0)
        #self.session_list.activated[str].connect(self.sessions_selected)    

        self.hbox_layout0 = QtWidgets.QHBoxLayout()
        hbox_layout1 = QtWidgets.QHBoxLayout()
        hbox_layout2 = QtWidgets.QHBoxLayout()
        hbox_layout3 = QtWidgets.QHBoxLayout()

        self.hbox_layout0.addWidget(self.slider)
        self.slider.hide()
        
        
        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=2)

        #hbox_layout1.addWidget(button1)
        hbox_layout1.addWidget(self.button2)
        #hbox_layout2.addWidget(button3)
        #hbox_layout3.addWidget(session_widget)
        
        vbox.addLayout(self.hbox_layout0)
        vbox.addLayout(hbox_layout1)
        vbox.addLayout(hbox_layout2)
        
        #vbox.addLayout(hbox_layout3)

    def _params_updated(self):
        self.clientEvent.emit(
            {"source": "slider", "data": "jump_to_slice", "frame": self.slider.value()}
        )


    def sessions_selected(self):
        cfg.ppw.clientEvent.emit(
            {"source": "workspace_gui", "data": "set_session", "session": self.session_list.value()}
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
            {
                "source": "button2",
                "data": "slice_mode",
            }
        )

    def button3_clicked(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "workspace_gui",
                "data": "run_workflow",
                "workflow_file": "tests/workflows/feature_set.yaml",
            }
        )

    def check1_checked(self):
        self._check1_checked = not self._check1_checked
        self.clientEvent.emit(
            {
                "source": "checkbox",
                "data": "flip_coords",
                "axis": "z",
                "value": self._check1_checked,
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
            pos=np.array([[0, 0], [700, 500]]),
            color="w",
            parent=self.canvas.scene,
        )
