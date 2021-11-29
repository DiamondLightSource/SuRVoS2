from survos2.config import Config
import numpy as np
from numpy.lib.function_base import flip
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton

from survos2.frontend.components.base import *
from survos2.frontend.components.entity import (
    SmallVolWidget,
    TableWidget,
    setup_entity_table,
    setup_bb_table,
)
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.utils import FileWidget
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.entity.patches import PatchWorkflow, organize_entities, make_patches



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
                elif result[fid]["kind"] == "boxes":
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

        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )
        result = Launcher.g.run("objects", "available", **params)

        print(result)
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
        logger.debug(f"Add objects with idx {idx}")
        if idx == 0 or idx == -1:
            return

        # self.objects_combo.setCurrentIndex(0)
        print(idx)
        order = idx - 2

        if order == 1:
            params = dict(
                order=order,
                workspace=DataModel.g.current_session
                + "@"
                + DataModel.g.current_workspace,
                fullname="survos2/entity/blank_boxes.csv",
            )
        else:
            params = dict(
                order=order,
                workspace=DataModel.g.current_session
                + "@"
                + DataModel.g.current_workspace,
                fullname="survos2/entity/blank_entities.csv",
            )
        result = Launcher.g.run("objects", "create", **params)

        if result:
            objectsid = result["id"]
            objectsname = result["name"]
            objectsfullname = result["fullname"]
            objectstype = result["kind"]
            self._add_objects_widget(
                objectsid, objectsname, objectsfullname, objectstype, True
            )

    def _add_objects_widget(
        self, objectsid, objectsname, objectsfullname, objectstype, expand=False
    ):
        logger.debug(
            f"Add objects {objectsid} {objectsname} {objectsfullname} {objectstype}"
        )
        widget = ObjectsCard(objectsid, objectsname, objectsfullname, objectstype)
        widget.showContent(expand)
        self.objects_layout.addWidget(widget)

        src = DataModel.g.dataset_uri(objectsid, group="objects")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            src_dataset.set_metadata("fullname", objectsfullname)
        self.existing_objects[objectsid] = widget

        return widget

    def clear(self):
        for objects in list(self.existing_objects.keys()):
            self.existing_objects.pop(objects).setParent(None)
        self.existing_objects = {}

    def setup(self):
        self._populate_objects()
        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )
        result = Launcher.g.run("objects", "existing", **params)

        logger.debug(f"objects result {result}")

        if result:
            # Remove objects that no longer exist in the server
            print(self.existing_objects.keys())
            for objects in list(self.existing_objects.keys()):
                if objects not in result:
                    self.existing_objects.pop(objects).setParent(None)

            # Populate with new entity if any
            for entity in sorted(result):
                if entity in self.existing_objects:
                    continue
                enitity_params = result[entity]
                objectsid = enitity_params.pop("id", entity)
                objectsname = enitity_params.pop("name", entity)
                objectsfullname = enitity_params.pop("fullname", entity)
                objectstype = enitity_params.pop("kind", entity)
                print(f"type: {objectstype}")
                if objectstype != "unknown":
                    widget = self._add_objects_widget(
                        objectsid, objectsname, objectsfullname, objectstype
                    )
                    widget.update_params(params)
                    self.existing_objects[objectsid] = widget
                else:
                    logger.debug(
                        "+ Skipping loading entity: {}, {}, {}".format(
                            objectsid, objectsname, objectstype
                        )
                    )


class ObjectsCard(Card):
    def __init__(
        self, objectsid, objectsname, objectsfullname, objectstype, parent=None
    ):
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
        self.objectstype = objectstype

        self.widgets = {}
        self.filewidget = FileWidget(extensions="*.csv", save=False)
        self.filewidget.path.setText(self.objectsfullname)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)
        self.get_btn = PushButton("Get", accent=True)

        # self._add_param("scale", title="Scale: ", type="Float", default=1)
        # self._add_param("offset", title="Offset: ", type="FloatOrVector", default=0)
        # self._add_param(
        #     "crop_start", title="Crop Start: ", type="FloatOrVector", default=0
        # )
        # self._add_param(
        #     "crop_end", title="Crop End: ", type="FloatOrVector", default=9000
        # )

        self.flipxy_checkbox = CheckBox(checked=True)
        self.add_row(HWidgets(None, self.flipxy_checkbox, Spacing(35)))
        self.add_row(HWidgets(None, self.view_btn, self.get_btn, Spacing(35)))

        self.view_btn.clicked.connect(self.view_objects)
        self.get_btn.clicked.connect(self.get_objects)

        #cfg.object_scale = self.widgets["scale"].value()
        #cfg.object_offset = self.widgets["offset"].value()
        #cfg.object_crop_start = self.widgets["crop_start"].value()
        #cfg.object_crop_end = self.widgets["crop_end"].value()

        cfg.object_scale = 1.0
        cfg.object_offset = (0,0,0)
        cfg.object_crop_start = (0,0,0)
        cfg.object_crop_end = (1e9,1e9,1e9)

        if self.objectstype == "patches":
            self._add_annotations_source()
            self.entity_mask_bvol_size = LineEdit3D(default=64, parse=int)
            self._add_feature_source()
            
            self.make_entity_mask_btn = PushButton("Make entity mask", accent=True)
            self.make_entity_mask_btn.clicked.connect(self.make_entity_mask)
            self.make_patches_btn = PushButton("Make patches", accent=True)
            self.make_patches_btn.clicked.connect(self.make_patches)
            self.train_fpn_btn = PushButton("Train FPN", accent=True)
            self.train_fpn_btn.clicked.connect(self.train_fpn)

            self.add_row(HWidgets(None, self.entity_mask_bvol_size, self.make_entity_mask_btn, Spacing(35)))
            self.add_row(HWidgets(None,  self.make_patches_btn, Spacing(35)))
            self.add_row(HWidgets(None, self.train_fpn_btn, Spacing(35)))


        self.table_control = TableWidget()
        self.add_row(self.table_control.w, max_height=500)
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

    def _add_annotations_source(self):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(
            "Annotation:", self.annotations_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited entity title {newtitle}")
        params = dict(objects_id=self.objectsid, new_name=newtitle, workspace=True)
        result = Launcher.g.run("objects", "rename", **params)
        return result["done"]

    def view_objects(self):
        logger.debug(f"Transferring objects {self.objectsid} to viewer")
        cfg.ppw.clientEvent.emit(
            {
                "source": "objects",
                "data": "view_objects",
                "objects_id": self.objectsid,
                "flipxy": self.flipxy_checkbox.value(),
            }
        )

    def update_params(self, params):
        if "fullname" in params:
            self.objectsfullname = params["fullname"]

    def _add_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def get_objects(self):
        #cfg.object_scale = self.widgets["scale"].value()
        #cfg.object_offset = self.widgets["offset"].value()
        #cfg.object_crop_start = self.widgets["crop_start"].value()
        #cfg.object_crop_end = self.widgets["crop_end"].value()

        dst = DataModel.g.dataset_uri(self.objectsid, group="objects")
        print(f"objectsfullname: {self.objectsfullname}")
        params = dict(
            dst=dst,
            fullname=self.objectsfullname,
            scale=cfg.object_scale,
            offset=cfg.object_offset,
            crop_start=cfg.object_crop_start,
            crop_end=cfg.object_crop_end,
        )
        logger.debug(f"Getting objects with params {params}")
        result = Launcher.g.run("objects", "update_metadata", workspace=True, **params)
          
        if self.objectstype == "points":
            
            tabledata, self.entities_df = setup_entity_table(
                 self.objectsfullname,
                 scale=cfg.object_scale,
                 offset=cfg.object_offset,
                 crop_start=cfg.object_crop_start,
                 crop_end=cfg.object_crop_end,
                 flipxy=self.flipxy_checkbox.value()
             )
            
        elif self.objectstype == "boxes":
            tabledata, self.entities_df = setup_bb_table(
                self.objectsfullname,
                scale=cfg.object_scale,
                offset=cfg.object_offset,
                crop_start=cfg.object_crop_start,
                crop_end=cfg.object_crop_end,
                flipxy=self.flipxy_checkbox.value()
            )
        elif self.objectstype == "patches":
            tabledata, self.entities_df = setup_entity_table(
                self.objectsfullname,
                scale=cfg.object_scale,
                offset=cfg.object_offset,
                crop_start=cfg.object_crop_start,
                crop_end=cfg.object_crop_end,
                flipxy=self.flipxy_checkbox.value()
            )      

        cfg.tabledata = tabledata
        self.table_control.set_data(tabledata)

        print(f"Loaded tabledata {tabledata}")
        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()

    def make_entity_mask(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_array = DM.sources[0][:]

        entity_arr = np.array(self.entities_df)

        bvol_dim = self.entity_mask_bvol_size.value()
        entity_arr[:, 0] -= bvol_dim[0]
        entity_arr[:, 1] -= bvol_dim[1]
        entity_arr[:, 2] -= bvol_dim[2]

        from survos2.entity.entities import make_entity_mask

        gold_mask = make_entity_mask(
            src_array, entity_arr, flipxy=True, bvol_dim=bvol_dim
        )[0]

        # create new raw feature
        params = dict(feature_type="raw", workspace=True)
        result = Launcher.g.run("features", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

            dst = DataModel.g.dataset_uri(fid, group="features")
            with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
                DM.out[:] = gold_mask

            cfg.ppw.clientEvent.emit(
                {"source": "objects_plugin", "data": "refresh", "value": None}
            )


    def make_patches(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_array = DM.sources[0][:]

        objects_scale = 1.0
        entity_meta = {
            "0": {
                "name": "class1",
                "size": np.array((15, 15, 15)) * objects_scale,
                "core_radius": np.array((7, 7, 7)) * objects_scale,
            },
        }
    
        entity_arr = np.array(self.entities_df)

        
        combined_clustered_pts, classwise_entities = organize_entities(
            src_array, entity_arr, entity_meta, plot_all=False
        )

        wparams = {}
        wparams["entities_offset"] = (0, 0, 0)
        wparams["entity_meta"] = entity_meta
        wparams["workflow_name"] = "Make_Patches"
        wparams["proj"] = DataModel.g.current_workspace
        wf = PatchWorkflow(
            [src_array], combined_clustered_pts, classwise_entities, src_array, wparams, combined_clustered_pts
        )

        src = DataModel.g.dataset_uri(self.annotations_source.value().rsplit("/", 1)[-1], group="annotations")
        with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            anno_level = src_dataset[:] & 15

        logger.debug(f"Obtained annotation level with labels {np.unique(anno_level)}")

        logger.debug(f"Making patches in path {src_dataset._path}")
        train_v_density = make_patches(wf, entity_arr, src_dataset._path, 
        proposal_vol=(anno_level > 0)* 1.0, 
        padding=(32,32,32), num_augs=0, max_vols=-1)

        self.patches = train_v_density

        cfg.ppw.clientEvent.emit(
            {"source": "panel_gui", "data": "view_patches", "patches_fullname": train_v_density}
        )


    def train_fpn(self):
        from survos2.entity.train import train_seg
        from survos2.entity.pipeline_ops import make_proposal
        
        wf_params = {}
        wf_params["torch_models_fullpath"] = "/experiments"
        model_file = train_seg(self.patches, wf_params, num_epochs=1)

        patch_size=(64,64,64)
        patch_overlap=(16, 16, 16)
        overlap_mode="crop"
        model_type="fpn3d"

        threshold_devs=1.5,
        invert=True,

        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_array = DM.sources[0][:]
        
        proposal = make_proposal(
        src_array,
        os.path.join(wf_params["torch_models_fullpath"],model_file),
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
        )

        # create new float image
        params = dict(feature_type="raw", workspace=True)
        result = Launcher.g.run("features", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

            dst = DataModel.g.dataset_uri(fid, group="features")
            with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
                DM.out[:] = proposal

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "refresh", "value": None}
            )

