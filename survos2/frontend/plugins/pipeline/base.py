import os
import ast
import logging
import numpy as np
from loguru import logger
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QSize, Signal


from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotation_tool import AnnotationComboBox
from survos2.frontend.plugins.annotations import LevelComboBox

from survos2.frontend.components.base import (
    VBox,
    ComboBox,
    LazyComboBox,
    HWidgets,
    PushButton,
    CheckBox,
    Card,
    LineEdit3D,
    LineEdit,
)

from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox, RealSlider
from survos2.frontend.plugins.superregions import RegionComboBox
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.utils import (
    get_array_from_dataset,
    get_color_mapping,
    hex_string_to_rgba,
)
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.utils import FileWidget


from napari.qt.progress import progress


class PipelinesComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run("pipelines", "existing", **params)
        if result:
            self.addCategory("Segmentations")
            for fid in result:
                self.addItem(fid, result[fid]["name"])


class PipelineCardBase(Card):
    def __init__(self, fid, ftype, fname, fparams, parent=None, pipeline_notifier=None):
        super().__init__(fname, removable=True, editable=True, collapsible=True, parent=parent)
        self.pipeline_id = fid
        self.pipeline_type = ftype
        self.pipeline_name = fname
        self.annotations_source = None
        self.params = fparams
        self.widgets = dict()

        self._PipelineNotifier = pipeline_notifier

        self.setup()

        for pname, params in fparams.items():
            if pname not in ["src", "dst"]:
                self._add_param(pname, **params)

        self._add_btns()
        self.dst = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")

    def _compute_pipeline(self):
        with progress(total=2) as pbar:
            pbar.set_description("Computing pipeline...")
            pbar.update(1)
            all_params = self.compute_pipeline()
            all_params.update({k: v.value() for k, v in self.widgets.items()})
            try:
                pbar.update(1)
                result = Launcher.g.run("pipelines", self.pipeline_type, **all_params)
                logger.debug(result)
            except Exception as err:
                logger.debug(err)
            if result is not None:
                pbar.update(2)

    def setup(self):
        pass

    def compute_pipeline(self):
        pass

    def _add_patch_params(self):
        self.patch_size = LineEdit3D(default=64, parse=int)
        self.add_row(HWidgets("Patch Size:", self.patch_size, stretch=1))

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def load_data(self, path):
        self.model_fullname = path
        logger.debug(f"Setting model fullname: {self.model_fullname}")

    def _add_refine_choice(self):
        self.refine_checkbox = CheckBox(checked=True)
        self.add_row(HWidgets("MRF Refinement:", self.refine_checkbox, stretch=0))

    def _add_confidence_choice(self):
        self.confidence_checkbox = CheckBox(checked=False)
        self.add_row(HWidgets("Confidence Map as Feature:", self.confidence_checkbox, stretch=0))

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)

        widget = HWidgets("Objects:", self.objects_source, stretch=1)
        self.add_row(widget)

    def _add_classifier_choice(self):
        self.classifier_type = ComboBox()
        self.classifier_type.addItem(key="Ensemble")
        self.classifier_type.addItem(key="SVM")
        widget = HWidgets("Classifier:", self.classifier_type, stretch=0)

        self.classifier_type.currentIndexChanged.connect(self._on_classifier_changed)

        self.clf_container = QtWidgets.QWidget()
        clf_vbox = VBox(self, spacing=4)
        clf_vbox.setContentsMargins(0, 0, 0, 0)
        self.clf_container.setLayout(clf_vbox)

        self.add_row(widget)
        self.add_row(self.clf_container, max_height=300)
        self.clf_container.layout().addWidget(self.ensembles)

    def _on_classifier_changed(self, idx):
        if idx == 0:
            self.clf_container.layout().addWidget(self.ensembles)
            self.svm.setParent(None)
        elif idx == 1:
            self.clf_container.layout().addWidget(self.svm)
            self.ensembles.setParent(None)

    def _add_fcn_choice(self):
        self.fcn_type = ComboBox()
        self.fcn_type.addItem(key="fpn3d")
        self.fcn_type.addItem(key="unet3d")
        self.fcn_type.addItem(key="vnet")
        widget = HWidgets("FCN Type:", self.fcn_type, stretch=0)
        self.add_row(widget)

    def _add_projection_choice(self):
        self.projection_type = ComboBox()
        self.projection_type.addItem(key="None")
        self.projection_type.addItem(key="pca")
        self.projection_type.addItem(key="rbp")
        self.projection_type.addItem(key="rproj")
        self.projection_type.addItem(key="std")
        widget = HWidgets("Projection:", self.projection_type, stretch=0)
        self.add_row(widget)

    def _add_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source, stretch=1)
        self.add_row(widget)

    def _add_feature_source2(self):
        self.feature_source2 = FeatureComboBox()
        self.feature_source2.fill()
        self.feature_source2.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source2, stretch=1)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)
        cfg.pipelines_features_source = self.features_source
        widget = HWidgets("Features:", self.features_source, stretch=1)
        self.add_row(widget)

    def _add_constrain_source(self):
        logger.debug(self.annotations_source.value())
        self.constrain_mask_source = AnnotationComboBox(header=(None, "None"), full=True)
        self.constrain_mask_source.fill()
        self.constrain_mask_source.setMaximumWidth(250)

        widget = HWidgets("Constrain mask:", self.constrain_mask_source, stretch=1)
        self.add_row(widget)

    def _add_annotations_source(self, label="Annotation"):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source, stretch=1)

        self.add_row(widget)

    def _add_annotations_source2(self, label="Annotation 2"):
        self.annotations_source2 = LevelComboBox(full=True)
        self.annotations_source2.fill()
        self.annotations_source2.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source2, stretch=1)
        self.add_row(widget)

    def _add_pipelines_source(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets("Segmentation:", self.pipelines_source, stretch=1)
        self.add_row(widget)

    def _add_regions_source(self):
        self.regions_source = RegionComboBox(full=True)  # SourceComboBox()
        self.regions_source.fill()
        self.regions_source.setMaximumWidth(250)

        widget = HWidgets("Superregions:", self.regions_source, stretch=1)
        cfg.pipelines_regions_source = self.regions_source
        self.add_row(widget)

    def _add_overlap_choice(self):
        self.overlap_type = ComboBox()
        self.overlap_type.addItem(key="crop")
        self.overlap_type.addItem(key="average")
        widget = HWidgets("Overlap:", self.overlap_type, stretch=0)
        self.add_row(widget)

    def _add_param(self, name, title=None, type="String", default=None):
        if type == "Int":
            p = LineEdit(default=0, parse=int)
        elif type == "FloatSlider":
            p = RealSlider(value=0.0, vmax=1, vmin=0)
            title = "MRF Refinement Amount:"
        elif type == "Float":
            p = LineEdit(default=0.0, parse=float)
            title = title
        elif type == "FloatOrVector":
            p = LineEdit3D(default=0, parse=float)
        elif type == "IntOrVector":
            p = LineEdit3D(default=0, parse=int)
        elif type == "SmartBoolean":
            p = CheckBox(checked=True)
        else:
            p = None

        if title is None:
            title = name

        if p:
            self.widgets[name] = p
            self.add_row(HWidgets(None, title, p))

    def _add_btns(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_pipeline)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as image", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)
        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self._compute_pipeline)
        self.add_row(
            HWidgets(
                None,
                load_as_float_btn,
                load_as_annotation_btn,
                compute_btn,
                view_btn,
            )
        )

    def _update_params(self, params):
        logger.debug(f"Pipeline update params {params}")
        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)
        if "anno_id" in params:
            if params["anno_id"] is not None:
                if isinstance(params["anno_id"], list):
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"][0][0])
                    )
                else:
                    self.annotations_source.select(os.path.join("annotations/", params["anno_id"]))

        if "object_id" in params:
            if params["object_id"] is not None:
                self.objects_source.select(os.path.join("objects/", params["object_id"]))
        if "feature_id" in params:
            # self.feature_source.select(os.path.join("features/", params["feature_id"]))
            self.feature_source.select(params["feature_id"])
        if "feature_ids" in params:
            for source in params["feature_ids"]:
                self.features_source.select(os.path.join("features/", source))

        if "feature_A" in params:
            self.feature_source.select(params["feature_A"])

        if "feature_B" in params:
            self.feature_source2.select(params["feature_B"])

        if "level_over" in params:
            logger.debug("level over found")
            self.annotations_source.select(os.path.join("annotations/", params["level_over"]))

        if "level_base" in params:
            logger.debug("level_base found")
            self.annotations_source2.select(os.path.join("annotations/", params["level_base"]))

        if "region_id" in params:
            if params["region_id"] is not None:
                self.regions_source.select(os.path.join("superregions/", params["region_id"]))

        if "constrain_mask" in params:
            if params["constrain_mask"] is not None and params["constrain_mask"] != "None":
                import ast

                constrain_mask_dict = ast.literal_eval(params["constrain_mask"])
                logger.debug(constrain_mask_dict)

                constrain_mask_source = (
                    constrain_mask_dict["level"] + ":" + str(constrain_mask_dict["idx"])
                )
                logger.debug(f"Constrain mask source {constrain_mask_source}")
                self.constrain_mask_source.select(constrain_mask_source)
        if "multi_ax_train_params" in params:
            if params["anno_id"]:
                table_dict = dict(
                    Labels=params["anno_id"],
                    Data=params["feature_id"],
                    Workspaces=params["workspace"],
                )
                self._update_data_table_from_dict(table_dict)
                self._update_multi_axis_cnn_train_params(
                    params["multi_ax_train_params"]["cyc_frozen"],
                    params["multi_ax_train_params"]["cyc_unfrozen"],
                )

    def card_deleted(self):
        params = dict(pipeline_id=self.pipeline_id, workspace=True)
        result = Launcher.g.run("pipelines", "remove", **params)
        if result["done"]:
            self.setParent(None)
            # self._PipelineNotifier.notify()

        cfg.ppw.clientEvent.emit(
            {
                "source": "pipelines",
                "data": "remove_layer",
                "layer_name": self.pipeline_id,
            }
        )

    def card_title_edited(self, newtitle):
        params = dict(pipeline_id=self.pipeline_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("pipelines", "rename", **params)

        if result["done"]:
            self._PipelineNotifier.notify()

        return result["done"]

    def view_pipeline(self):
        logger.debug(f"View pipeline_id {self.pipeline_id}")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing feature")
            pbar.update(1)
            if self.annotations_source:
                logger.debug(f"anno source value {self.annotations_source.value()}")

                if self.annotations_source.value():
                    level_id = str(self.annotations_source.value().rsplit("/", 1)[-1])
                    # To cover situtations where the level_id might not exist in ws:
                    anno_result = Launcher.g.run("annotations", "get_levels", workspace=True)
                    if not any([x["id"] == level_id for x in anno_result]):
                        logger.debug(f"level_id {level_id} does not exist changing:")
                        level_id = anno_result[-1]["id"]
                else:
                    level_id = "001_level"

                logger.debug(f"Assigning annotation level {level_id}")

                cfg.ppw.clientEvent.emit(
                    {
                        "source": "pipelines",
                        "data": "view_pipeline",
                        "pipeline_id": self.pipeline_id,
                        "level_id": level_id,
                    }
                )
            else:
                cfg.ppw.clientEvent.emit(
                    {
                        "source": "pipelines",
                        "data": "view_pipeline",
                        "pipeline_id": self.pipeline_id,
                        "level_id": "001_level",
                    }
                )

            pbar.update(1)

    def get_model_path(self):
        workspace_path = os.path.join(DataModel.g.CHROOT, DataModel.g.current_workspace)
        self.model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, ("Select model"), workspace_path, ("Model files (*.pytorch)")
        )
        self.model_file_line_edit.setValue(self.model_path)

    def _update_annotations_from_ws(self, workspace):
        self.annotations_source.clear()
        params = {"workspace": workspace}
        anno_result = Launcher.g.run("annotations", "get_levels", **params)
        logger.debug(f"anno_result: {anno_result}")
        if anno_result:
            for r in anno_result:
                if r["kind"] == "level":
                    self.annotations_source.addItem(r["id"], r["name"])

    def refresh_multi_ax_data(self):
        sender = self.sender()
        cfg.ppw.clientEvent.emit({"source": "workspace_gui", "data": "refresh", "value": None})
        if sender == self.multi_ax_pred_refresh_btn:
            search_str = "U-Net prediction"
            self.annotations_source.fill()
        elif sender == self.multi_ax_train_refresh_btn:
            search_str = "U-Net Training"
            self._update_annotations_from_ws(DataModel.g.current_workspace)
        index = self.annotations_source.findText(search_str, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.annotations_source.setCurrentText(search_str)
            logger.debug(f"Setting index to {index}")
        else:
            num_annotations = self.annotations_source.count()
            self.annotations_source.setCurrentIndex(num_annotations - 1)
        logger.debug(f"Annotations source selected: {self.annotations_source.value()}")

    def load_as_float(self):
        logger.debug(f"Loading prediction {self.pipeline_id} as float image.")

        # get pipeline output
        src = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
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
                DM.out[:] = src_arr

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "refresh_plugin", "plugin_name": "features"}
            )

    def load_as_annotation(self):
        logger.debug(f"Loading prediction {self.pipeline_id} as annotation.")

        # get pipeline output
        src = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
        label_values = np.unique(src_arr)

        # create new level
        params = dict(level=self.pipeline_id, workspace=True)
        result = Launcher.g.run("annotations", "add_level", workspace=True)

        # create a blank label for each unique value in the pipeline output array
        if result:
            level_id = result["id"]
            for v in label_values:
                params = dict(
                    level=level_id,
                    idx=int(v),
                    name=str(v),
                    color="#11FF11",
                    workspace=True,
                )
                label_result = Launcher.g.run("annotations", "add_label", **params)

            params = dict(
                level=str(self.annotations_source.value().rsplit("/", 1)[-1]),
                workspace=True,
            )
            print(self.annotations_source.value().rsplit("/", 1))
            print(params)
            anno_result = Launcher.g.run("annotations", "get_single_level", **params)

            print(anno_result)
            params = dict(level=str(level_id), workspace=True)
            level_result = Launcher.g.run("annotations", "get_single_level", **params)
            print(level_result)
            print(level_id)
            print(params)
            try:
                # set the new level color mapping to the mapping from the pipeline
                for v in level_result["labels"].keys():
                    if v in anno_result["labels"]:
                        label_hex = anno_result["labels"][v]["color"]
                        label = dict(
                            idx=int(v),
                            name=str(v),
                            color=label_hex,
                        )
                        params = dict(level=result["id"], workspace=True)
                        label_result = Launcher.g.run(
                            "annotations", "update_label", **params, **label
                        )
            except Exception as err:
                logger.debug(f"Exception {err}")

            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

            # set levels array to pipeline output array
            dst = DataModel.g.dataset_uri(fid, group="annotations")
            with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                DM.out[:] = src_arr

            cfg.ppw.clientEvent.emit(
                {
                    "source": "workspace_gui",
                    "data": "faster_refresh_plugin",
                    "plugin_name": "annotations",
                }
            )
