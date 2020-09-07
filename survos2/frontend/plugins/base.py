"""
QT Plugins Base


"""
import numpy as np
import pandas as pd
from numba import jit
import scipy
import yaml
from vispy import scene
from vispy.color import Colormap
from loguru import logger
from scipy import ndimage
from skimage import img_as_ubyte, img_as_float
from skimage import io

from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize
from qtpy import QtWidgets, QtCore, QtGui

from vispy import scene
from vispy.color import Colormap

import pyqtgraph as pg
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from survos2.server.config import cfg
from survos2.helpers import prepare_3channel, simple_norm, norm1
from survos2.frontend.control import Launcher

from survos2.frontend.components.base import *
from survos2.frontend.components.base import QCSWidget

from collections import OrderedDict

# from plugins/base.py
__available_plugins__ = OrderedDict()


class Plugin(QCSWidget):

    change_view = QtCore.Signal(str, dict)

    __icon__ = 'square'
    __pname__ = 'plugin'
    __title__ = None
    __views__ = []

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._loaded_views = dict()

    def show_view(self, name, **kwargs):
        if name not in self.__views__:
            raise ValueError('View `{}` was not preloaded.'.format(name))
        self.change_view.emit(name, kwargs)

    def register_view(self, view, widget):
        if not view in self.__views__:
            return
        self._loaded_views[view] = widget

    def on_created(self):
        pass

    def __getitem__(self, name):
        return self._loaded_views[name]


def register_plugin(cls):
    name = cls.__pname__
    icon = cls.__icon__
    title = cls.__title__
    views = cls.__views__

    if name in __available_plugins__:
        raise ValueError('Plugin {} already registered.'.format(name))

    if title is None:
        title = name.capitalize()

    desc = dict(cls=cls, name=name, icon=icon, title=title, views=views)
    __available_plugins__[name] = desc
    return cls


def get_plugin(name):
    if name not in __available_plugins__:
        raise ValueError('Plugin {} not registered'.format(name))
    return __available_plugins__[name]


def list_plugins():
    return list(__available_plugins__.keys())



##
# from mainwindow.py


class PluginContainer(QCSWidget):

    view_requested = QtCore.Signal(str, dict)

    __sidebar_width__ = 440

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMinimumWidth(self.__sidebar_width__)
        #self.setMaximumWidth(self.__sidebar_width__)

        self.title = Header('Plugin')
        self.container = ScrollPane(parent=self)

        vbox = VBox(self, margin=(1, 1, 2, 0), spacing=2)
        #vbox.addWidget(self.title)
        vbox.addWidget(self.container, 1)

        self.plugins = {}
        self.selected_name = None
        self.selected = None

    def load_plugin(self, name, title, cls):
        if name in self.plugins:
            return
        widget = cls()
        widget.change_view.connect(self.view_requested)
        self.plugins[name] = dict(widget=widget, title=title)
        return widget

    def unload_plugin(self, name):
        self.plugins.pop(name, None)

    def show_plugin(self, name):
        if name in self.plugins: #and name != self.selected_name:
            print(f"show_plugin: {name}")
        
            #if self.selected is not None:
            #    self.selected['widget'].setParent(None)
            self.selected_name = name
            self.selected = self.plugins[name]
            self.title.setText(self.selected['title'])
            self.container.addWidget(self.selected['widget'], 1)
            if hasattr(self.selected['widget'], 'setup'):
                self.selected['widget'].setup()



