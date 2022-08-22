from qtpy import QtWidgets, QtCore, QtGui
import numpy as np
from skimage.draw import line
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap
import numpy as np
from loguru import logger

from survos2.frontend.control.launcher import Launcher
from survos2.frontend.plugins.viewer import ViewerExtension, Tool
from survos2.frontend.components.base import (
    LazyComboBox,
    LazyMultiComboBox,
    HBox,
    FAIcon,
    PluginNotifier,
    Slider,
)
from survos2.frontend.plugins.superregions import RegionComboBox
from survos2.model.model import DataModel
from survos2.utils import decode_numpy

_AnnotationNotifier = PluginNotifier()


def _fill_level(combo, level):
    combo.addCategory(level["name"])
    if not "labels" in level:
        return
    for label in level["labels"].values():
        icon = FAIcon("fa.square", color=label["color"])
        lidx = "{}:{}".format(level["id"], label["idx"])
        data = dict(level=level["id"], idx=label["idx"], color=label["color"])
        combo.addItem(lidx, label["name"], icon=icon, data=data)


def _fill_annotations(combo, full=False, workspace=True):
    params = dict(workspace=workspace, full=full)
    levels = Launcher.g.run("annotations", "get_levels", **params)
    if levels:
        for level in levels:
            if combo.exclude_from_fill is not None:
                logger.debug(f"Excluding level {combo.exclude_from_fill} {level}")
                if level["id"] != combo.exclude_from_fill:
                    _fill_level(combo, level)
            else:
                _fill_level(combo, level)


class AnnotationComboBox(LazyComboBox):
    def __init__(
        self,
        full=False,
        parent=None,
        header=(None, "Select Label"),
        exclude_from_fill=None,
        workspace=True,
    ):
        self.full = full
        self.exclude_from_fill = exclude_from_fill
        self.workspace = workspace
        super().__init__(parent=parent, header=header)
        _AnnotationNotifier.listen(self.update)

    def fill(self):
        _fill_annotations(self, full=self.full, workspace=self.workspace)


class MultiAnnotationComboBox(LazyMultiComboBox):
    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(parent=parent, text="Select Label", groupby=("level", "idx"))
        _AnnotationNotifier.listen(self.update)

    def fill(self):
        _fill_annotations(self, full=self.full)
