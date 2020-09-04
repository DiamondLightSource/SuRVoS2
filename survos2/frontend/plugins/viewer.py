from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize
from qtpy import QtWidgets, QtCore, QtGui
from survos2.frontend.components.base import QCSWidget


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

