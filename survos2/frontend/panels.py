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


from survos2.frontend.components.base import Slider

class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.slice_mode = False

        self.slider = Slider(value=100)
        self.slider.setMinimumWidth(150)
        self.slider.sliderReleased.connect(self._params_updated)
        
        #button1 = QPushButton("Jump", self)
        #button1.clicked.connect(self.button1_clicked)
        button2 = QPushButton("Slice mode", self)
        button2.clicked.connect(self.button2_clicked)
        self._selected_entity_idx = 0

        check1 = QCheckBox("Z", self)
        check2 = QCheckBox("X", self)
        check3 = QCheckBox("Y", self)

        check1.setText("Z")
        check2.setText("X")
        check3.setText("Y")

        self._check1_checked = False
        check1.clicked.connect(self.check1_checked)

        self.hbox_layout0 = QtWidgets.QHBoxLayout()
        hbox_layout1 = QtWidgets.QHBoxLayout()
        hbox_layout2 = QtWidgets.QHBoxLayout()
        
        self.hbox_layout0.addWidget(self.slider)
        self.slider.hide()
           
        #hbox_layout1.addWidget(button1)
        hbox_layout1.addWidget(button2)

        vbox = VBox(self, margin=(1, 0, 0, 0), spacing=5)

        vbox.addLayout(self.hbox_layout0)
        vbox.addLayout(hbox_layout1)
        #vbox.addLayout(hbox_layout2)

        #label_flip = QtWidgets.QLabel("Flip coords:")
        #hbox_layout2.addWidget(label_flip)
        #hbox_layout2.addWidget(check1, 0)
        #hbox_layout2.addWidget(check2, 1)
        #hbox_layout2.addWidget(check3, 2)

    def _params_updated(self):
        self.clientEvent.emit({"source": "slider", "data": "jump_around", "frame" : self.slider.value()})


    def button1_clicked(self):
        self.clientEvent.emit({"source": "button1", "data": "jump_around"})

    def button2_clicked(self):
        if self.slice_mode:
            self.slider.hide()
        else:
            self.slider.show()

        self.slice_mode = not self.slice_mode

        
        self.clientEvent.emit(
            {
                "source": "button2",
                "data": "slice_mode",
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
