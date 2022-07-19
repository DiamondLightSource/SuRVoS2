import numpy as np

from loguru import logger
from scipy import ndimage
from skimage import img_as_ubyte, img_as_float
from skimage import io
from qtpy.QtWidgets import QRadioButton, QPushButton

from qtpy import QtWidgets, QtCore, QtGui

from survos2.frontend.components.base import *
from survos2.frontend.control import Launcher


def _fill_features(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)

    result = Launcher.g.run("features", "existing", **params)

    if ignore==None:
        ignore=[]


    if result:
        for fid in result:
            if fid not in ignore:
                combo.addItem(fid, result[fid]["name"])

    else:
        result = dict()
        params.setdefault("id", 7)
        params.setdefault("name", "feat0")
        params.setdefault("kind", "unknown")

        result[0] = params


_FeatureNotifier = PluginNotifier()


class SourceComboBox(LazyComboBox):
    def __init__(self, ignore_source=None, parent=None):
        self.ignore_source = ignore_source
        super().__init__(header=("__data__", "Raw Data"), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, ignore=self.ignore_source)


class MultiSourceComboBox(LazyMultiComboBox):
    def __init__(self, parent=None):
        super().__init__(
            header=("__data__", "Raw Data"), text="Select Source", parent=parent
        )
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, full=True)


class Slider(QCSWidget):

    valueChanged = QtCore.Signal(int)

    def __init__(
        self,
        value=None,
        vmax=100,
        vmin=0,
        step=1,
        tracking=True,
        label=True,
        auto_accept=True,
        center=False,
        parent=None,
    ):
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
            self.setValue(self.value() + self.step)
        elif e.angleDelta().y() < 0 and self.value() > self.minimum():
            self.setValue(self.value() - self.step)

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
        idx = "{0:.3f}".format(self._values[idx])
        super().update_label(idx)

    def _update_linspace(self):
        self._values = np.linspace(self._vmin, self._vmax, self._n + 1, endpoint=True)

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


