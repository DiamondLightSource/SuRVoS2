import os
import numpy as np
import ntpath
from loguru import logger
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation
from skimage.draw import line
from skimage.morphology import disk
from qtpy import QtWidgets, QtCore
from qtpy.QtWidgets import QPushButton, QRadioButton, QGroupBox, QGridLayout
from survos2.frontend.components.base import (
    VBox,
    HWidgets,
    PushButton,
    QCSWidget,
    CheckBox,
    CardWithId,
    LineEdit,
    HBox,
    LazyComboBox,
    PluginNotifier,
    Slider,
    ColorButton,
    ParentButton,
)
from survos2.frontend.plugins.base import (
    Plugin,
    register_plugin,
)

from survos2.frontend.components.icon_buttons import DelIconButton, IconButton
from survos2.frontend.control.launcher import Launcher
from survos2.frontend.plugins.annotation_tool import (
    AnnotationComboBox,
    _AnnotationNotifier,
)


from survos2.frontend.plugins.superregions import RegionComboBox
from survos2.model import DataModel
from survos2.server.state import cfg
from napari.qt.progress import progress

_AnnotationNotifier = PluginNotifier()


class LevelComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None, workspace=True, ignore=None):
        self.full = full
        self.ignore = ignore
        self.workspace = workspace
        super().__init__(header=header, parent=parent)
        _AnnotationNotifier.listen(self.update)

    def fill(self):
        params = dict(workspace=self.workspace, full=self.full)
        result = Launcher.g.run("annotations", "get_levels", **params)
        if result:
            self.addCategory("Annotations")
            for r in result:
                level_name = r["name"]
                if level_name != self.ignore:
                    if r["kind"] == "level":
                        self.addItem(r["id"], r["name"])


def dilate_annotations(yy, xx, img_shape, line_width):
    try:
        data = np.zeros(img_shape)
        data[yy, xx] = True

        r = np.ceil(line_width / 2)
        ymin = int(max(0, yy.min() - r))
        ymax = int(min(data.shape[0], yy.max() + 1 + r))
        xmin = int(max(0, xx.min() - r))
        xmax = int(min(data.shape[1], xx.max() + 1 + r))
        mask = data[ymin:ymax, xmin:xmax]

        mask = binary_dilation(mask, disk(line_width / 2).astype(bool))
        yy, xx = np.where(mask)
        yy += ymin
        xx += xmin
    except Exception as err:
        logger.debug(f"Exception: {err}")
    return yy, xx


@register_plugin
class AnnotationPlugin(Plugin):

    __icon__ = "fa.pencil"
    __pname__ = "annotations"
    __views__ = ["slice_viewer"]
    __tab__ = "annotations"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.btn_addlevel = IconButton("fa.plus", "Add Level", accent=True)
        self.vbox = VBox(self, spacing=10)
        self.vbox2 = VBox(self, spacing=10)

        advanced_button = QRadioButton("Advanced")
        advanced_button.toggled.connect(self.toggle_advanced)
        self.vbox.addWidget(advanced_button)

        self.setup_adv_run_fields()
        self.vbox.addWidget(self.adv_run_fields)
        self.adv_run_fields.hide()

        self.levels = {}
        self.btn_addlevel.clicked.connect(self.add_level)
        self.timer_id = -1

        hbox = HBox(self, margin=1, spacing=3)
        self.label = AnnotationComboBox()
        self.region = RegionComboBox(header=(None, "Voxels"), full=True)
        self.label.currentIndexChanged.connect(self.set_sv)
        self.region.currentIndexChanged.connect(self.set_sv)
        self.btn_set = IconButton("fa.pencil", accent=True)
        self.btn_set.clicked.connect(self.set_sv)

        cfg.three_dim_checkbox = CheckBox(checked=False)

        hbox.addWidget(self.label)
        hbox.addWidget(self.region)
        self.width = Slider(value=10, vmin=2, vmax=100, step=2, auto_accept=True)
        hbox.addWidget(self.width)
        hbox.addWidget(self.btn_set)

        self.vbox.addLayout(hbox)
        self.vbox.addWidget(self.btn_addlevel)

    def toggle_advanced(self):
        """Controls displaying/hiding the advanced run fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_run_fields.show()
        else:
            self.adv_run_fields.hide()

    def setup_adv_run_fields(self):
        self.adv_run_fields = QGroupBox()
        self.button_pause_save = QPushButton("Local Annotation", self)
        self.button_pause_save.clicked.connect(self.button_pause_save_clicked)
        self.button_save_anno = QPushButton("Save to Server", self)
        self.button_save_anno.clicked.connect(self.button_save_anno_clicked)
        self.button_annotation_from_slice = QPushButton("Annotation from Slice", self)
        self.button_annotation_from_slice.clicked.connect(self.annotation_from_slice_clicked)
        self.hbox2 = HBox(self, margin=1, spacing=10)
        self.hbox2.addWidget(self.button_pause_save)
        self.hbox2.addWidget(self.button_save_anno)
        self.hbox2.addWidget(self.button_annotation_from_slice)
        self.adv_run_fields.setLayout(self.hbox2)

    def on_created(self):
        pass

    def add_level(self):
        cfg.anno_data = None
        level = Launcher.g.run("annotations", "add_level", workspace=True)
        if level:
            self._add_level_widget(level)
            _AnnotationNotifier.notify()

    def _add_level_widget(self, level):
        widget = AnnotationLevel(level, None, self.width)
        widget.removed.connect(self.remove_level)
        self.vbox.addWidget(widget)
        self.levels[level["id"]] = widget

    def uncheck_labels(self):
        for button in self.btn_group.buttons():
            selected = self.btn_group.checkedButton()
            button.setChecked(button is selected)

    def remove_level(self, level):
        if level in self.levels:
            self.levels.pop(level).setParent(None)
        _AnnotationNotifier.notify()

    def clear(self):
        for level in list(self.levels.keys()):
            self.remove_level(level)
        self.levels = {}

    def button_save_anno_clicked(self):
        if "anno_data" in cfg:
            cfg.anno_data = cfg.anno_data & 15
        cfg.ppw.clientEvent.emit(
            {
                "source": "save_annotation",
                "data": "save_annotation",
                "value": None,
            }
        )

    def annotation_from_slice_clicked(self):
        slice_num = cfg.viewer.cursor.position[0]
        params = dict(target_level=self.label.value()["level"],
                      source_level=self.label.value()["level"],
                      region=os.path.basename(cfg.current_supervoxels),
                      slice_num=slice_num,
                      modal=False,
                      workspace=True,
                      viewer_order=cfg.viewer_order,)
        
        result = Launcher.g.run("annotations", "annotate_from_slice", json_transport=True, **params)
        cfg.ppw.clientEvent.emit(
                {
                    "source": "annotations",
                    "data": "paint_annotations",
                    "level_id": self.label.value()["level"],
                }
            )

    def button_pause_save_clicked(self):
        cfg.remote_annotation = not cfg.remote_annotation
        if cfg.remote_annotation:
            self.button_pause_save.setText("Local Annotation")
        else:
            self.button_pause_save.setText("Remote Annotation")
        if "prev_arr" in cfg:
            cfg.prev_arr = np.zeros_like(cfg.prev_arr)

    def setup(self):
        params = dict(workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace)
        result = Launcher.g.run("annotations", "get_levels", **params)
        if not result:
            return

        # Remove levels that no longer exist in the server
        rlevels = [r["id"] for r in result]
        for level in list(self.levels.keys()):
            if level not in rlevels:
                self.remove_level(level)

        # Populate with new levels if any
        for level in result:
            if level["id"] not in self.levels:
                self._add_level_widget(level)

    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.set_sv()

    def set_sv(self):
        cfg.current_supervoxels = self.region.value()
        cfg.label_value = self.label.value()
        cfg.brush_size = self.width.value()
        logger.debug(f"set_sv {cfg.current_supervoxels}, {cfg.label_value}")
        # cfg.three_dim = cfg.three_dim_checkbox.value()

        if cfg.label_value is not None:
            # example 'label_value': {'level': '001_level', 'idx': 2, 'color': '#ff007f'}
            cfg.ppw.clientEvent.emit(
                {
                    "source": "annotations",
                    "data": "set_paint_params",
                    "paint_params": {
                        "current_supervoxels": self.region.value(),
                        "label_value": self.label.value(),
                        "brush_size": self.width.value(),
                        "level_id": self.label.value()["level"],
                    },
                }
            )
            cfg.ppw.clientEvent.emit(
                {
                    "source": "annotations",
                    "data": "paint_annotations",
                    "level_id": self.label.value()["level"],
                }
            )
            cfg.ppw.clientEvent.emit(
                {
                    "source": "annotations",
                    "data": "set_paint_params",
                    "paint_params": {
                        "current_supervoxels": self.region.value(),
                        "label_value": self.label.value(),
                        "brush_size": self.width.value(),
                        "level_id": self.label.value()["level"],
                    },
                }
            )


class AnnotationLevel(CardWithId):

    removed = QtCore.Signal(str)

    def __init__(self, level, parent=None, brush_slider=None):
        super().__init__(
            level["name"],
            level["id"],
            editable=True,
            collapsible=True,
            removable=True,
            addbtn=True,
            parent=parent,
        )
        self.brush_slider = brush_slider
        self.level_id = level["id"]
        self.le_title = LineEdit(level["name"])
        self.le_title.setProperty("header", True)
        self.labels = {}
        self._add_view_btn()
        self._populate_labels()

    def card_title_edited(self, title):
        params = dict(level=self.level_id, name=title, workspace=True)
        return Launcher.g.run("annotations", "rename_level", **params)

    def card_add_item(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run("annotations", "add_label", **params)
        self.view_level()
        if result:
            self._add_label_widget(result)
            _AnnotationNotifier.notify()

    def card_deleted(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "annotations",
                "data": "remove_layer",
                "layer_name": self.level_id,
            }
        )

        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run("annotations", "delete_level", **params)
        if result:
            self.removed.emit(self.level_id)
            # _AnnotationNotifier.notify()

        cfg.current_annotation_layer = None

    def remove_label(self, idx):
        if idx in self.labels:
            self.labels.pop(idx).setParent(None)
            self.update_height()

    def _add_label_widget(self, label):
        widget = AnnotationLabel(label, self.level_id, None, self.brush_slider)
        widget.removed.connect(self.remove_label)
        self.add_row(widget)
        self.labels[label["idx"]] = widget
        self.expand()

    def _populate_labels(self):
        params = dict(level=self.level_id, workspace=True)
        result = Launcher.g.run("annotations", "get_labels", **params)
        if result:
            for k, label in result.items():
                if k not in self.labels:
                    self._add_label_widget(label)

    def view_level(self):
        with progress(total=2) as pbar:
            pbar.set_description("Viewing feature")
            pbar.update(1)
            cfg.ppw.clientEvent.emit(
                {
                    "source": "annotations",
                    "data": "paint_annotations",
                    "level_id": self.level_id,
                }
            )
            pbar.update(1)

    def _add_view_btn(self):
        btn_view = PushButton("View", accent=True)
        btn_view.clicked.connect(self.view_level)
        self.add_row(HWidgets(None, btn_view))


class AnnotationLabel(QCSWidget):

    __height__ = 30
    removed = QtCore.Signal(int)

    def __init__(self, label, level_dataset, parent=None, brush_slider=None):
        super().__init__(parent=parent)
        logger.debug(f"Adding label: {label}")
        self.level_dataset = level_dataset
        self.label_idx = int(label["idx"])
        self.label_color = label["color"]
        self.label_name = label["name"]
        self.label_visible = label["visible"]
        self.level_number = int(level_dataset.split("_")[0])

        self.btn_del = DelIconButton(secondary=True)
        self.txt_label_name = LineEdit(label["name"])
        self.txt_idx = LineEdit(str(label["idx"]))
        self.btn_label_color = ColorButton(label["color"])
        self.btn_set = IconButton("fa.pencil", accent=True)

        self.brush_slider = brush_slider

        params = dict(workspace=True, level=level_dataset, label_idx=int(label["idx"]))
        result = Launcher.g.run("annotations", "get_label_parent", **params)

        if result[0] == -1:
            self.parent_level = -1
            self.parent_label = -1
            self.btn_label_parent = ParentButton(
                color=None,
                source_level=self.level_dataset,
                parent_level=self.parent_level,
                parent_label=self.parent_label,
            )
        else:
            self.parent_level = result[0]
            self.parent_label = result[1]
            self.btn_label_parent = ParentButton(
                color=result[2],
                source_level=self.level_dataset,
                parent_level=self.parent_level,
                parent_label=self.parent_label,
            )

        self.setMinimumHeight(self.__height__)
        self.setFixedHeight(self.__height__)
        self.txt_label_name.editingFinished.connect(self.update_label)
        self.btn_label_color.colorChanged.connect(self.update_label)
        self.btn_del.clicked.connect(self.delete)
        self.btn_set.clicked.connect(self.set_label)
        self.btn_label_parent.colorChanged.connect(self.set_parent)

        hbox = HBox(self)
        hbox.addWidget(
            HWidgets(
                self.btn_del,
                str(label["idx"] - 1),
                self.btn_label_color,
                self.btn_label_parent,
                self.txt_label_name,
                self.btn_set,
                stretch=4,
            )
        )

    def set_label(self):
        label_dict = {
            "level": self.level_dataset,
            "idx": self.label_idx,
            "color": self.label_color,
        }
        cfg.ppw.clientEvent.emit(
            {
                "source": "annotations",
                "data": "set_paint_params",
                "paint_params": {
                    "current_supervoxels": cfg.current_supervoxels,
                    "label_value": label_dict,
                    "brush_size": self.brush_slider.value(),
                    "level_id": self.level_dataset,
                },
            }
        )
        cfg.ppw.clientEvent.emit(
            {
                "source": "annotations",
                "data": "paint_annotations",
                "level_id": self.level_dataset,
            }
        )
        cfg.ppw.clientEvent.emit(
            {
                "source": "annotations",
                "data": "set_paint_params",
                "paint_params": {
                    "current_supervoxels": cfg.current_supervoxels,
                    "label_value": label_dict,
                    "brush_size": self.brush_slider.value(),
                    "level_id": self.level_dataset,
                },
            }
        )
        cfg.brush_size = self.brush_slider.value()

    def set_parent(self):
        try:
            logger.debug(
                f"Setting parent level to {self.btn_label_parent.parent_level}, with label {self.label_idx}"
            )

            if int(self.label_idx) != -1:
                params = dict(
                    workspace=True,
                    level=self.level_dataset,
                    label_idx=self.label_idx,
                    parent_level=self.btn_label_parent.parent_level,
                    parent_label_idx=self.btn_label_parent.parent_label,
                    parent_color=self.btn_label_parent.color,
                )
                result = Launcher.g.run("annotations", "set_label_parent", **params)
        except Exception as e:
            logger.debug(e)

        cfg.ppw.clientEvent.emit(
            {
                "source": "annotations",
                "data": "paint_annotations",
                "level_id": self.level_dataset,
            }
        )

    def update_label(self):
        label = dict(
            idx=self.label_idx,
            name=self.txt_label_name.text(),
            color=self.btn_label_color.color,
            visible=self.label_visible,
        )
        params = dict(level=self.level_dataset, workspace=True)
        result = Launcher.g.run("annotations", "update_label", **params, **label)

        if result:
            self.label_name = result["name"]
            self.label_color = result["color"]
            self.label_visible = result["visible"]
            _AnnotationNotifier.notify()
        else:
            self.txt_label_name.setText(self.label_name)
            self.btn_label_color.setColor(self.label_color)

    def delete(self):
        params = dict(level=self.level_dataset, workspace=True, idx=self.label_idx)
        result = Launcher.g.run("annotations", "delete_label", **params)
        if result:
            _AnnotationNotifier.notify()
            self.removed.emit(self.label_idx)
