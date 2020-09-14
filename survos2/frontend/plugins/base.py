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

from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget, QRadioButton, QPushButton
from qtpy.QtCore import QSize
from qtpy import QtWidgets, QtCore, QtGui

from vispy import scene
from vispy.color import Colormap

import pyqtgraph as pg

import pyqtgraph.parametertree.parameterTypes as pTypes

from survos2.frontend.components.base import *
from survos2.frontend.components.base import QCSWidget

from collections import OrderedDict

__available_plugins__ = OrderedDict()


class Plugin(QCSWidget):

    change_view = QtCore.Signal(str, dict)

    __icon__ = "square"
    __pname__ = "plugin"
    __title__ = None
    __views__ = []
    __tab__ = "workspace"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._loaded_views = dict()

    def show_view(self, name, **kwargs):
        if name not in self.__views__:
            raise ValueError("View `{}` was not preloaded.".format(name))
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
    tab = cls.__tab__

    if name in __available_plugins__:
        raise ValueError("Plugin {} already registered.".format(name))

    if title is None:
        title = name.capitalize()

    desc = dict(cls=cls, name=name, icon=icon, title=title, views=views, tab=tab)
    __available_plugins__[name] = desc

    
    return cls


def get_plugin(name):
    if name not in __available_plugins__:
        raise ValueError("Plugin {} not registered".format(name))
    return __available_plugins__[name]


def list_plugins():
    return list(__available_plugins__.keys())


class PluginContainer(QCSWidget):

    view_requested = QtCore.Signal(str, dict)
    __sidebar_width__ = 440

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMinimumWidth(self.__sidebar_width__)
        
        self.tabwidget = QTabWidget()
        vbox = VBox(self, margin=(1, 1, 2, 0), spacing=2)
        vbox.addWidget(self.tabwidget, 1)

        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()

        self.tabwidget.addTab(tab1, "Workspace")
        self.tabwidget.addTab(tab2, "Segmentation")
        self.tabwidget.addTab(tab3, "Objects")
        self.tabwidget.addTab(tab4, "Analyze")

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)

        tab2.layout = QVBoxLayout()
        tab2.setLayout(tab2.layout)

        tab3.layout = QVBoxLayout()
        tab3.setLayout(tab3.layout)

        tab4.layout = QVBoxLayout()
        tab4.setLayout(tab4.layout)

        self.title = Header("Plugin")

        self.workspace_container = ScrollPane(parent=self)
        self.segmentation_container = ScrollPane(parent=self)
        self.entities_container = ScrollPane(parent=self)
        self.analyze_container = ScrollPane(parent=self)

        tab1.layout.addWidget(self.workspace_container)
        tab2.layout.addWidget(self.segmentation_container)
        tab3.layout.addWidget(self.entities_container)
        tab4.layout.addWidget(self.analyze_container)

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

    def show_plugin(self, name, tab):
        if name in self.plugins:  # and name != self.selected_name:
            print(f"show_plugin: {name}")

            # if self.selected is not None:
            #    self.selected['widget'].setParent(None)
            self.selected_name = name
            self.selected = self.plugins[name]
            self.title.setText(self.selected["title"])

            if tab=='workspace':
                self.workspace_container.addWidget(self.selected["widget"], 1)
            elif tab=='segmentation':
                self.segmentation_container.addWidget(self.selected["widget"], 1)
            elif tab=='entities':
                self.entities_container.addWidget(self.selected["widget"], 1)
                
            elif tab=='analyze':
                self.analyze_container.addWidget(self.selected["widget"], 1)                 
            if hasattr(self.selected["widget"], "setup"):
                self.selected["widget"].setup()
