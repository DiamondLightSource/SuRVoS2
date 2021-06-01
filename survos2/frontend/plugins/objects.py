import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton

from survos2.frontend.components.base import *
from survos2.frontend.components.entity import (
    SmallVolWidget,
    TableWidget,
    setup_entity_table,
)
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.utils import FileWidget
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg


class ObjectComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)

        result = Launcher.g.run("objects", "existing", **params)
        logger.debug(f"Result of objects existing: {result}")
        if result:
            # self.addCategory("Points")
            for fid in result:
                if result[fid]["kind"] == "points":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class ObjectsPlugin(Plugin):

    __icon__ = "fa.picture-o"
    __pname__ = "objects"
    __views__ = ["slice_viewer"]
    __tab__ = "objects"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vbox = VBox(self, spacing=10)
        self.objects_combo = ComboBox()
        self.vbox.addWidget(self.objects_combo)

        self.existing_objects = {}
        self.objects_layout = VBox(margin=0, spacing=5)
        self.objects_combo.currentIndexChanged.connect(self.add_objects)

        self.vbox.addLayout(self.objects_layout)
        self._populate_objects()

    def _populate_objects(self):
        self.objects_params = {}
        self.objects_combo.clear()
        self.objects_combo.addItem("Add objects")

        result = None
        result = Launcher.g.run("objects", "available", workspace=True)

        logger.debug(f"objects available: {result}")
        if result:
            all_categories = sorted(set(p["category"] for p in result))

            for i, category in enumerate(all_categories):
                self.objects_combo.addItem(category)
                self.objects_combo.model().item(
                    i + len(self.objects_params) + 1
                ).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.objects_params[f["name"]] = f["params"]
                    self.objects_combo.addItem(f["name"])

    def add_objects(self, idx):
        if idx == 0:
            return
        pipeline_type = self.objects_combo.itemText(idx)
        self.objects_combo.setCurrentIndex(0)

        params = dict(
            order=0, workspace=True, fullname="survos2/entity/blank_entities.csv",
        )
        result = Launcher.g.run("objects", "create", **params)

        if result:
            objectsid = result["id"]
            objectsname = result["name"]
            objectsfullname = result["fullname"]
            self._add_objects_widget(objectsid, objectsname, objectsfullname, True)

    def _add_objects_widget(
        self, objectsid, objectsname, objectsfullname, expand=False
    ):
        logger.debug(f"Add objects {objectsid} {objectsname} {objectsfullname}")
        widget = ObjectsCard(objectsid, objectsname, objectsfullname)
        widget.showContent(expand)
        self.objects_layout.addWidget(widget)

        src = DataModel.g.dataset_uri(objectsid, group="objects")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            src_dataset.set_metadata("entities_fullname", objectsfullname)
        self.existing_objects[objectsid] = widget

        return widget

    def setup(self):
        params = dict(workspace=True)
        result = Launcher.g.run("objects", "existing", **params)
        logger.debug(f"objects result {result}")

        if result:
            # Remove objects that no longer exist in the server
            for objects in list(self.existing_objects.keys()):
                if objects not in result:
                    self.existing_objects.pop(objects).setParent(None)

            # Populate with new entity if any
            for entity in sorted(result):
                if entity in self.existing_objects:
                    continue
                params = result[entity]
                objectsid = params.pop("id", entity)
                objectsname = params.pop("name", entity)
                objectsfullname = params.pop("fullname", entity)

                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_objects_widget(
                        objectsid, objectsname, objectsfullname
                    )
                    widget.update_params(params)
                    self.existing_objects[objectsid] = widget
                else:
                    logger.debug(
                        "+ Skipping loading entity: {}, {}".format(
                            objectsid, objectsname
                        )
                    )

    def setup2(self):
        params = dict(order=0, workspace=True)

        params["id"] = 0
        params["name"] = "points1"
        params["kind"] = "objects"
        params["fullname"] = "a.csv"

        result = {}
        result[0] = params

        result = Launcher.g.run("objects", "existing", **params)
        if result:
            # Remove objects that no longer exist in the server
            for entity in list(self.existing_objects.keys()):
                if entity not in result:
                    self.existing_objects.pop(entity).setParent(None)

            # Populate with new entity if any
            for entity in sorted(result):

                if entity in self.existing_objects:
                    continue
                params = result[entity]
                objectsid = params.pop("id", entity)
                objectsname = params.pop("name", entity)
                objectsfullname = params.pop("fullname", entity)

                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_objects_widget(
                        objectsid, objectsname, objectsfullname
                    )
                    widget.update_params(params)
                    self.existing_objects[objectsid] = widget
                else:
                    logger.debug(
                        "+ Skipping loading entity: {}, {}".format(
                            objectsid, objectsname
                        )
                    )


class ObjectsCard(Card):
    def __init__(self, objectsid, objectsname, objectsfullname, parent=None):
        super().__init__(
            title=objectsname,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.objectsid = objectsid
        self.objectsname = objectsname
        self.object_scale = 1.0

        self.objectsfullname = objectsfullname
        self.widgets = {}
        self.filewidget = FileWidget(extensions="*.csv", save=False)
        self.filewidget.path.setText(self.objectsfullname)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)
        self.get_btn = PushButton("Get", accent=True)

        self._add_param("scale", title="Scale: ", type="Float", default=1)
        self._add_param("offset", title="Offset: ", type="FloatOrVector", default=0)
        self._add_param("crop_start", title="Crop Start: ", type="FloatOrVector", default=0)
        self._add_param("crop_end", title="Crop End: ", type="FloatOrVector", default=9000)

        self.add_row(HWidgets(None, self.view_btn, self.get_btn, Spacing(35)))

        self.view_btn.clicked.connect(self.view_objects)
        self.get_btn.clicked.connect(self.get_objects)

        cfg.object_scale = self.widgets["scale"].value()
        cfg.object_offset = self.widgets["offset"].value()
        cfg.object_crop_start = self.widgets["crop_start"].value()
        cfg.object_crop_end = self.widgets["crop_end"].value()

        self.table_control = TableWidget()
        self.add_row(self.table_control.w, max_height=500)

        tabledata, _ = setup_entity_table(objectsfullname, 
                                          scale=cfg.object_scale, 
                                          offset=cfg.object_offset,
                                          crop_start=cfg.object_crop_start,
                                          crop_end=cfg.object_crop_end)
        cfg.tabledata = tabledata
        self.table_control.set_data(tabledata)
        cfg.entity_table = self.table_control
        
    def _add_param(self, name, title=None, type="String", default=None):
        if type == "Int":
            p = LineEdit(default=default, parse=int)
        elif type == "Float":
            p = LineEdit(default=default, parse=float)
        elif type == "FloatOrVector":
            p = LineEdit3D(default=default, parse=float)
        elif type == "IntOrVector":
            p = LineEdit3D(default=default, parse=int)
        else:
            p = None
        if title is None:
            title = name
        if p:
            self.widgets[name] = p
            self.add_row(HWidgets(None, title, p, Spacing(35)))

    def load_data(self, path):
        self.objectsfullname = path
        print(f"Setting objectsfullname: {self.objectsfullname}")

    def card_deleted(self):
        params = dict(objects_id=self.objectsid, workspace=True)
        result = Launcher.g.run("objects", "remove", **params)
        if result["done"]:
            self.setParent(None)
        self.table_control = None

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited entity title {newtitle}")
        params = dict(objects_id=self.objectsid, new_name=newtitle, workspace=True)
        result = Launcher.g.run("objects", "rename", **params)
        return result["done"]

    def view_objects(self):
        logger.debug(f"Transferring objects {self.objectsid} to viewer")
        cfg.ppw.clientEvent.emit(
            {"source": "objects", "data": "view_objects", "objects_id": self.objectsid}
        )

    def update_params(self, params):
        if "fullname" in params:
            self.objectsfullname = params["fullname"]

    def get_objects(self):
        cfg.object_scale = self.widgets["scale"].value()
        cfg.object_offset = self.widgets["offset"].value()
        cfg.object_crop_start = self.widgets["crop_start"].value()
        cfg.object_crop_end = self.widgets["crop_end"].value()

        dst = DataModel.g.dataset_uri(self.objectsid, group="objects")
        print(f"objectsfullname: {self.objectsfullname}")
        params = dict(dst=dst, fullname=self.objectsfullname, scale=cfg.object_scale, 
            offset=cfg.object_offset, crop_start=cfg.object_crop_start, crop_end=cfg.object_crop_end)
        logger.debug(f"Getting objects with params {params}")
        Launcher.g.run("objects", "points", **params)
        
        tabledata, _ = setup_entity_table(
            self.objectsfullname, entities_df=None, scale=cfg.object_scale, offset=cfg.object_offset, crop_start=cfg.object_crop_start,
            crop_end=cfg.object_crop_end
        )

        cfg.tabledata = tabledata
        print(f"Loaded tabledata {tabledata}")
        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()
