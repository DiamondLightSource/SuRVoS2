import numpy as np

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton

from survos2.frontend.plugins.base import *
from survos2.frontend.components.base import *
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.utils import FileWidget       
from survos2.improc.utils import DatasetManager
from survos2.server.state import cfg

from survos2.frontend.components.entity import (
    TableWidget,
    SmallVolWidget,
    setup_entity_table,
)


@register_plugin
class EntitysPlugin(Plugin):

    __icon__ = "fa.picture-o"
    __pname__ = "entitys"
    __views__ = ["slice_viewer"]
    __tab__ = "entitys"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)

        self(
            IconButton("fa.plus", "Add Objects", accent=True),
            connect=("clicked", self.add_entitys),
        )

        self.existing_entitys = {}
        self.entitys_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.entitys_layout)

    def add_entitys(self):
        params = dict(
            order=0,
            workspace=True,
            fullname="survos2/entity/blank_entities.csv",
        )
        result = Launcher.g.run("entitys", "create", **params)

        if result:
            entitysid = result["id"]
            entitysname = result["name"]
            entitysfullname = result["fullname"]
            self._add_entitys_widget(entitysid, entitysname, entitysfullname, True)

    def _add_entitys_widget(
        self, entitysid, entitysname, entitysfullname, expand=False
    ):
        widget = EntitysCard(entitysid, entitysname, entitysfullname)
        widget.showContent(expand)
        self.entitys_layout.addWidget(widget)

        src = DataModel.g.dataset_uri(entitysid, group="entitys")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            src_dataset.set_metadata("entities_fullname", entitysfullname)

        self.existing_entitys[entitysid] = widget
        return widget

    def setup(self):
        params = dict(order=1, workspace=True)

        params["id"] = 0
        params["name"] = "points1"
        params["kind"] = "entitys"
        params["fullname"] = "a.csv"
        result = {}
        result[0] = params

        result = Launcher.g.run("entitys", "existing", **params)
        if result:
            # Remove entitys that no longer exist in the server
            for entity in list(self.existing_entitys.keys()):
                if entity not in result:
                    self.existing_entitys.pop(entity).setParent(None)

            # Populate with new entity if any
            for entity in sorted(result):

                if entity in self.existing_entitys:
                    continue
                params = result[entity]
                entitysid = params.pop("id", entity)
                entitysname = params.pop("name", entity)
                entitysfullname = params.pop("fullname", entity)

                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_entitys_widget(
                        entitysid, entitysname, entitysfullname
                    )
                    widget.update_params(params)
                    self.existing_entitys[entitysid] = widget
                else:
                    logger.debug(
                        "+ Skipping loading entity: {}, {}".format(
                            entitysid, entitysname
                        )
                    )


class EntitysCard(Card):
    def __init__(self, entitysid, entitysname, entitysfullname, parent=None):
        super().__init__(
            title=entitysname,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.entitysid = entitysid
        self.entitysname = entitysname

        #self.entitysfullname = LineEdit(parse=str, default=50)
        #self.entitysfullname.setValue(entitysfullname)
        self.entitysfullname = entitysfullname

        self.filewidget = FileWidget(extensions="*.csv", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

        #self.entitysfullname.setMaximumWidth(250)
        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)
        self.get_btn = PushButton("Get", accent=True)

        #self.add_row(HWidgets("Source:", self.entitysfullname, stretch=1))
        self.add_row(HWidgets(None, self.view_btn, Spacing(35)))
        self.add_row(HWidgets(None, self.get_btn, Spacing(35)))

        self.view_btn.clicked.connect(self.view_entitys)
        self.get_btn.clicked.connect(self.get_entitys)

        self.table_control = TableWidget()
        tabledata, _ = setup_entity_table(entitysfullname, scale=0.25)
        cfg.tabledata = tabledata
        self.table_control.set_data(tabledata)
        cfg.entity_table = self.table_control
        self.add_row(self.table_control.w, max_height=500)
        

    def load_data(self, path):
        self.entitysfullname = path
        print(f"Setting entitiesfullname: {self.entitysfullname}")
    def card_deleted(self):
        params = dict(entitys_id=self.entitysid, workspace=True)
        result = Launcher.g.run("entitys", "remove", **params)
        if result["done"]:
            self.setParent(None)
        self.table_control = None

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited entity title {newtitle}")
        params = dict(entitys_id=self.entitysid, new_name=newtitle, workspace=True)
        result = Launcher.g.run("entitys", "rename", **params)
        return result["done"]

    def view_entitys(self):
        logger.debug(f"Transferring entitys {self.entitysid} to viewer")
        cfg.ppw.clientEvent.emit(
            {"source": "entitys", "data": "view_entitys", "entitys_id": self.entitysid}
        )

    def update_params(self, params):
        if "fullname" in params:
            self.entitysfullname = params["fullname"]
        
    def get_entitys(self):
        dst = DataModel.g.dataset_uri(self.entitysid, group="entitys")
        print(f"entitysfullname: {self.entitysfullname}")
        params = dict(dst=dst, fullname=self.entitysfullname)
        logger.debug(f"Getting entitys with params {params}")
        Launcher.g.run("entitys", "set_csv", **params)
        tabledata, _ = setup_entity_table(self.entitysfullname, scale=0.25)
        cfg.tabledata = tabledata
        print(f"Loaded tabledata {tabledata}")
        self.table_control.set_data(tabledata)
