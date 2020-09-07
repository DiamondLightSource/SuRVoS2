
import numpy as np
from skimage.draw import line
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from matplotlib.colors import ListedColormap


import numpy as np

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal


from survos2.frontend.components.base import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import register_plugin, Plugin
from survos2.frontend.plugins.viewer import Tool, ViewerExtension
from survos2.frontend.plugins.annotation_tool import AnnotationTool
from survos2.frontend.plugins.regions import RegionComboBox
from survos2.frontend.components.icon_buttons import DelIconButton, IconButton
from survos2.frontend.control.launcher import Launcher

from survos2.model import DataModel

from loguru import logger

from survos2.utils import decode_numpy
from survos2.server.config import cfg

_AnnotationNotifier = PluginNotifier()

class LevelComboBox(LazyComboBox):

    def __init__(self, full=False, header=(None, 'None'), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        
        #result = [{'kind': 'supervoxels'}, ] 
        result = Launcher.g.run('annotations', 'get_levels', **params)
        logger.debug(f"Result of regions existing: {result}")
        
        if result:
            self.addCategory('Annotations')
            for r in result:
                print(r)
                level_name = r['name']
                if r['kind'] == 'level':
                    self.addItem(r['id'], r['name'])





@register_plugin
class AnnotationPlugin(Plugin):

    __icon__ = 'fa.pencil'
    __pname__ = 'annotations'
    __views__ = ['slice_viewer']

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.btn_addlevel = IconButton('fa.plus', 'Add Level', accent=True)
        self.vbox = VBox(self, spacing=10)
        self.vbox.addWidget(self.btn_addlevel)
        self.levels = {}
        self.btn_addlevel.clicked.connect(self.add_level)
        self.annotation_tool = AnnotationTool()

    def on_created(self):
        self['slice_viewer'].add_tool('Annotations', self.__icon__,
                                      tool=self.annotation_tool)

    def add_level(self):
        level = Launcher.g.run('annotations', 'add_level', workspace=True)
        if level:
            self._add_level_widget(level)
            _AnnotationNotifier.notify()

    def _add_level_widget(self, level):
        widget = AnnotationLevel(level)
        widget.removed.connect(self.remove_level)
        self.vbox.addWidget(widget)
        self.levels[level['id']] = widget

    def uncheck_labels(self):
        for button in self.btn_group.buttons():
            selected = self.btn_group.checkedButton()
            button.setChecked(button is selected)

    def remove_level(self, level):
        if level in self.levels:
            self.levels.pop(level).setParent(None)

    def setup(self):
        result = Launcher.g.run('annotations', 'get_levels', workspace=True)
        if not result:
            return
        # Remove levels that no longer exist in the server
        rlevels = [r['id'] for r in result]
        for level in list(self.levels.keys()):
            if level not in rlevels:
                self.remove_level(level)
        # Populate with new levels if any
        for level in result:
            if level['id'] not in self.levels:
                self._add_level_widget(level)

    

class AnnotationLevel(Card):

    removed = QtCore.Signal(str)

    def __init__(self, level, parent=None):
        super().__init__(level['name'], editable=True, collapsible=True,
                         removable=True, addbtn=True, parent=parent)
        self.level_id = level['id']
        self.le_title = LineEdit(level['name'])
        self.le_title.setProperty('header', True)
        self.labels = {}
        self._populate_labels()
        self._add_view_btn()

    def card_title_edited(self, title):
        params = dict(level=self.level_id, name=title, workspace=True)
        return Launcher.g.run('annotations', 'rename_level', **params)

    def card_add_item(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'add_label', **params)
        if result:
            self._add_label_widget(result)
            _AnnotationNotifier.notify()

    def card_deleted(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'delete_level', **params)
        if result:
            self.removed.emit(self.level_id)
            _AnnotationNotifier.notify()

    def view_level(self):
        logger.debug(f"View feature_id {self.level_id}")
        cfg.ppw.clientEvent.emit({'source': 'annotations', 
                                            'data':'view_annotations', 
                                            'level_id': self.level_id})

    def remove_label(self, idx):
        if idx in self.labels:
            self.labels.pop(idx).setParent(None)
            self.update_height()

    def _add_label_widget(self, label):
        widget = AnnotationLabel(label, self.level_id)
        widget.removed.connect(self.remove_label)
        self.add_row(widget)
        self.labels[label['idx']] = widget
        self.expand()

    def _populate_labels(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run('annotations', 'get_labels', **params)
        if result:
            for k, label in result.items():
                if k not in self.labels:
                    self._add_label_widget(label)

    def _add_view_btn(self):
        btn_view = PushButton('View', accent=True)
        btn_view.clicked.connect(self.view_level)
        self.add_row(HWidgets(None, btn_view, Spacing(35)))


class AnnotationLabel(QCSWidget):

    __height__ = 30

    removed = QtCore.Signal(int)

    def __init__(self, label, level_dataset, parent=None):
        super().__init__(parent=parent)
        self.level_dataset = level_dataset
        self.label_idx = label['idx']
        self.label_color = label['color']
        self.label_name = label['name']
        self.label_visible = label['visible']

        self.btn_del = DelIconButton(secondary=True)
        self.txt_label_name = LineEdit(label['name'])
        self.btn_label_color = ColorButton(label['color'])
        self.btn_select = Spacing(35)
        self.setMinimumHeight(self.__height__)
        self.setFixedHeight(self.__height__)

        self.txt_label_name.editingFinished.connect(self.update_label)
        self.btn_label_color.colorChanged.connect(self.update_label)
        self.btn_del.clicked.connect(self.delete)

        hbox = HBox(self)
        hbox.addWidget(HWidgets(self.btn_del, self.btn_label_color,
                                self.txt_label_name, self.btn_select, stretch=2))

    def update_label(self):
        label = dict(idx=self.label_idx, name=self.txt_label_name.text(),
                     color=self.btn_label_color.color, visible=self.label_visible)
        params = dict(level=self.level_dataset, workspace=True)
        result = Launcher.g.run('annotations', 'update_label', **params, **label)
        if result:
            self.label_name = result['name']
            self.label_color = result['color']
            self.label_visible = result['visible']
            _AnnotationNotifier.notify()
        else:
            self.txt_label_name.setText(self.label_name)
            self.btn_label_color.setColor(self.label_color)

    def delete(self):
        params = dict(level=self.level_dataset, workspace=True, idx=self.label_idx)
        result = Launcher.g.run('annotations', 'delete_label', **params)
        if result:
            _AnnotationNotifier.notify()
            self.removed.emit(self.label_idx)
