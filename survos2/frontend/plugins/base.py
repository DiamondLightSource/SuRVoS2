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
#import seaborn as sns


from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize

from qtpy import QtWidgets, QtCore, QtGui

from vispy import scene
from vispy.color import Colormap

import pyqtgraph as pg
#from pyqtgraph.Qt import QtCore, QtGui


import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
#rom pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.opengl as gl
import pyqtgraph as pg

#import rx
#from rx.subject import Subject


from survos2.server.config import scfg
from survos2.helpers import prepare_3channel, simple_norm, norm1
from survos2.frontend.components import *
from survos2.frontend.control import Launcher

#############################################################
def _fill_features(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    
    result = Launcher.g.run('features', 'existing', **params)
    
    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]['name'])

    else:
        result = dict()
        params.setdefault('id',7)
        params.setdefault('name', 'feat0')
        params.setdefault('kind', 'unknown')
    
        result[0] = params
    
        
   

###########################
_FeatureNotifier = PluginNotifier()

class FeatureComboBox(LazyComboBox):

    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(header=(None, 'None'), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        
        _fill_features(self, full=self.full)


class SourceComboBox(LazyComboBox):

    def __init__(self, ignore_source=None, parent=None):
        self.ignore_source = ignore_source
        super().__init__(header=('__data__', 'Raw Data'), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, ignore=self.ignore_source)


class MultiSourceComboBox(LazyMultiComboBox):

    def __init__(self, parent=None):
        super().__init__(header=('__data__', 'Raw Data'), text='Select Source',
                         parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, full=True)

class Slider(QCSWidget):

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(self, value=None, vmax=100, vmin=0, step=1, tracking=True,
                 label=True, auto_accept=True, center=False, parent=None):
        super().__init__(parent=parent)
        if value is None:
            value = vmin
        self.setMinimumWidth(200)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(vmin)
        self.slider.setMaximum(vmax)
        self.slider.setValue(value)
        self.slider.setTickInterval(step)
        self.slider.setSingleStep(step)
        self.slider.setTracking(tracking)
        self.step = step

        hbox = HBox(self, spacing=5)
        if label:
            self.label = Label(str(value))
            self.label.setMinimumWidth(50)
            if center:
                hbox.addSpacing(50)
            hbox.addWidget(self.slider, 1)
            hbox.addWidget(self.label)
            self.valueChanged.connect(self.update_label)
        else:
            hbox.addWidget(self.slider, 1)

        self.slider.valueChanged.connect(self.value_changed)
        self.slider.wheelEvent = self.wheelEvent
        self.auto_accept = auto_accept
        self.locked_idx = None
        self.pending = None
        self.blockSignals = self.slider.blockSignals

    def value_changed(self, idx):
        if self.auto_accept:
            self.valueChanged.emit(idx)
        elif self.locked_idx is None:
            self.locked_idx = idx
            self.valueChanged.emit(idx)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(self.locked_idx)
            self.slider.blockSignals(False)
            self.pending = idx

    def accept(self):
        if self.pending is not None:
            val = self.pending
            self.pending = None
            self.slider.blockSignals(True)
            self.slider.setValue(val)
            self.slider.blockSignals(False)
            self.valueChanged.emit(val)
        self.locked_idx = None

    def update_label(self, idx):
        self.label.setText(str(idx))

    def wheelEvent(self, e):
        if e.angleDelta().y() > 0 and self.value() < self.maximum():
            self.setValue(self.value()+self.step)
        elif e.angleDelta().y() < 0 and self.value() > self.minimum():
            self.setValue(self.value()-self.step)

    def value(self):
        return self.pending or self.slider.value()

    def setValue(self, value):
        return self.slider.setValue(value)

    def __getattr__(self, key):
        return self.slider.__getattribute__(key)


class RealSlider(Slider):

    def __init__(self, value=0, vmax=100, vmin=0, n=1000, **kwargs):
        super().__init__(value=0, vmin=0, vmax=n, **kwargs)
        self._n = n
        self._vmin = vmin
        self._vmax = vmax
        self._update_linspace()
        self.blockSignals(True)
        self.setValue(value)
        self.update_label(self._mapvalue(value))
        self.blockSignals(False)

    def _mapvalue(self, val):
        return (np.abs(self._values - val)).argmin()

    def value(self):
        return self._values[self.slider.value()]

    def update_label(self, idx):
        idx = '{0:.3f}'.format(self._values[idx])
        super().update_label(idx)

    def _update_linspace(self):
        self._values = np.linspace(self._vmin, self._vmax,
                                   self._n + 1, endpoint=True)

    def setValue(self, val):
        idx = self._mapvalue(val)
        super().setValue(idx)

    def setMaximum(self, vmax):
        self._vmax = vmax
        self._update_linspace()

    def setMinimum(self, vmin):
        self._vmin = vmin
        self._update_linspace()

    def maximum(self):
        return self._vmax

    def minimum(self):
        return self._vmin

class Label(QtWidgets.QLabel):

    def __init__(self, *args):
        super().__init__(*args)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def value(self):
        return self.text()






###############################################################

from collections import OrderedDict

# from plugins/base.py
__available_plugins__ = OrderedDict()


class Plugin(QCSWidget):

    change_view = QtCore.pyqtSignal(str, dict)

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


class Tool(QCSWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._viewer = None
        self._current_idx = 0

    @property
    def viewer(self):
        return self._viewer

    @property
    def current_idx(self):
        return self._current_idx

    def setEnabled(self, flag):
        super().setEnabled(flag)
        if flag:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        if self.viewer:
            self.viewer.slice_updated.connect(self.slice_updated)

    def disconnect(self):
        if self.viewer:
            self.viewer.slice_updated.disconnect(self.slice_updated)

    def set_viewer(self, viewer):
        self.disconnect()
        self._viewer = viewer
        self.connect()

    def slice_updated(self, idx):
        self._current_idx = idx


class ViewerExtension(QtCore.QObject):

    def __init__(self, modifiers=None, enabled=True):
        super().__init__()
        self.fig = None
        self.axes = None

        self._connections = []
        self._enabled = enabled
        self._modifiers = modifiers or QtCore.Qt.NoModifier

    def isEnabled(self):
        return self._enabled

    def setEnabled(self, flag):
        self._enabled = bool(flag)

    def active(self):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        return self.isEnabled() and modifiers == self._modifiers

    def install(self, fig, axes):
        self.disconnect()
        self.fig = fig
        self.axes = axes

    def disable(self):
        self.disconnect()
        self.fig = None
        self.axes = None

    def connect(self, event, callback):
        if not self.fig:
            return
        func = lambda evt: self.active() and callback(evt)
        self._connections.append(self.fig.mpl_connect(event, func))

    def disconnect(self):
        if not self.fig:
            return
        for conn in self._connections:
            self.fig.mpl_disconnect(conn)
        self._connections.clear()

    def redraw(self):
        if self.fig:
            self.fig.redraw()



##
# from mainwindow.py


class PluginContainer(QCSWidget):

    view_requested = QtCore.pyqtSignal(str, dict)

    __sidebar_width__ = 400

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        #self.setMinimumWidth(self.__sidebar_width__)
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



