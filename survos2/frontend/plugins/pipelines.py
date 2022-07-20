import ast
import logging
import numpy as np
from loguru import logger
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal

from survos2.frontend.components.base import *
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotation_tool import AnnotationComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import ComboBox, LazyComboBox, LazyMultiComboBox,  DataTableWidgetItem
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

_PipelineNotifier = PluginNotifier()


def _fill_pipelines(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    result = Launcher.g.run("pipelines", "existing", **params)

    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


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


@register_plugin
class PipelinesPlugin(Plugin):
    __icon__ = "fa.picture-o"
    __pname__ = "pipelines"
    __views__ = ["slice_viewer"]
    __tab__ = "pipelines"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.pipeline_combo = ComboBox()
        self.vbox = VBox(self, spacing=4)
        self.vbox.addWidget(self.pipeline_combo)
        self.pipeline_combo.currentIndexChanged.connect(self.add_pipeline)
        self.existing_pipelines = dict()
        self._populate_pipelines()

    def _populate_pipelines(self):
        self.pipeline_params = {}
        self.pipeline_combo.clear()
        self.pipeline_combo.addItem("Add segmentation")

        result = Launcher.g.run("pipelines", "available", workspace=True)

        if not result:
            params = {}
            params["category"] = "superregion"
            params["name"] = "s0"
            params["type"] = "superregion_segment"
            result = {}
            result[0] = params
            self.pipeline_params["superregion_segment"] = {
                "sr_params": {
                    "type": "sr2",
                }
            }
        else:
            all_categories = sorted(set(p["category"] for p in result))

            for i, category in enumerate(all_categories):
                self.pipeline_combo.addItem(category)
                self.pipeline_combo.model().item(
                    i + len(self.pipeline_params) + 1
                ).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.pipeline_params[f["name"]] = f["params"]
                    self.pipeline_combo.addItem(f["name"])

    def add_pipeline(self, idx):
        if idx <= 0:
            return
        if self.pipeline_combo.itemText(idx) == "":
            return

        logger.debug(f"Adding pipeline {self.pipeline_combo.itemText(idx)}")

        pipeline_type = self.pipeline_combo.itemText(idx)
        self.pipeline_combo.setCurrentIndex(0)

        params = dict(pipeline_type=pipeline_type, workspace=True)
        result = Launcher.g.run("pipelines", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            self._add_pipeline_widget(fid, ftype, fname, True)
            _PipelineNotifier.notify()

    def _add_pipeline_widget(self, fid, ftype, fname, expand=False):
        widget = PipelineCard(fid, ftype, fname, self.pipeline_params[ftype])
        widget.showContent(expand)
        self.vbox.addWidget(widget)
        self.existing_pipelines[fid] = widget
        return widget

    def clear(self):
        for pipeline in list(self.existing_pipelines.keys()):
            self.existing_pipelines.pop(pipeline).setParent(None)
        self.existing_pipelines = {}

    def setup(self):
        self._populate_pipelines()
        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )
        result = Launcher.g.run("pipelines", "existing", **params)
        logger.debug(f"Pipeline result {result}")

        if result:
            # Remove pipelines that no longer exist in the server
            for pipeline in list(self.existing_pipelines.keys()):
                if pipeline not in result:
                    self.existing_pipelines.pop(pipeline).setParent(None)

            # Populate with new pipelines if any
            for pipeline in sorted(result):
                if pipeline in self.existing_pipelines:
                    continue
                params = result[pipeline]
                logger.debug(f"Pipeline params {params}")
                fid = params.pop("id", pipeline)
                ftype = params.pop("kind")
                fname = params.pop("name", pipeline)
                widget = self._add_pipeline_widget(fid, ftype, fname)
                widget.update_params(params)
                self.existing_pipelines[fid] = widget


class SVMWidget(QtWidgets.QWidget):
    predict = Signal(dict)

    def __init__(self, parent=None):
        super(SVMWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = ComboBox()
        self.type_combo.addCategory("Kernel Type:")
        self.type_combo.addItem("linear")
        self.type_combo.addItem("poly")
        self.type_combo.addItem("rbf")
        self.type_combo.addItem("sigmoid")
        vbox.addWidget(self.type_combo)

        self.penaltyc = LineEdit(default=1.0, parse=float)
        self.gamma = LineEdit(default=1.0, parse=float)

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("Penalty C:"),
                self.penaltyc,
                QtWidgets.QLabel("Gamma:"),
                self.gamma,
                stretch=[0, 1, 0, 1],
            )
        )

    def on_predict_clicked(self):
        params = {
            "clf": "svm",
            "kernel": self.type_combo.currentText(),
            "C": self.penaltyc.value(),
            "gamma": self.gamma.value(),
        }

        self.predict.emit(params)

    def get_params(self):
        params = {
            "clf": "svm",
            "kernel": self.type_combo.currentText(),
            "C": self.penaltyc.value(),
            "gamma": self.gamma.value(),
            "type": self.type_combo.value(),
        }

        return params


class EnsembleWidget(QtWidgets.QWidget):
    train_predict = Signal(dict)

    def __init__(self, parent=None):
        super(EnsembleWidget, self).__init__(parent=parent)

        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)

        self.type_combo = ComboBox()
        self.type_combo.addCategory("Ensemble Type:")
        self.type_combo.addItem("Random Forest")
        self.type_combo.addItem("ExtraRandom Forest")
        self.type_combo.addItem("AdaBoost")
        self.type_combo.addItem("GradientBoosting")
        #self.type_combo.addItem("XGBoost")

        self.type_combo.currentIndexChanged.connect(self.on_ensemble_changed)
        vbox.addWidget(self.type_combo)

        self.ntrees = LineEdit(default=100, parse=int)
        self.depth = LineEdit(default=15, parse=int)
        self.lrate = LineEdit(default=1.0, parse=float)
        self.subsample = LineEdit(default=1.0, parse=float)

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("# Trees:"),
                self.ntrees,
                QtWidgets.QLabel("Max Depth:"),
                self.depth,
                stretch=[0, 1, 0, 1],
            )
        )

        vbox.addWidget(
            HWidgets(
                QtWidgets.QLabel("Learn Rate:"),
                self.lrate,
                QtWidgets.QLabel("Subsample:"),
                self.subsample,
                stretch=[0, 1, 0, 1],
            )
        )

        # self.btn_train_predict = PushButton('Train & Predict')
        # self.btn_train_predict.clicked.connect(self.on_train_predict_clicked)
        self.n_jobs = LineEdit(default=10, parse=int)
        vbox.addWidget(HWidgets("Num Jobs", self.n_jobs))

    def on_ensemble_changed(self, idx):
        if idx == 2:
            self.ntrees.setDefault(50)
        else:
            self.ntrees.setDefault(100)

        if idx == 3:
            self.lrate.setDefault(0.1)
            self.depth.setDefault(3)
        else:
            self.lrate.setDefault(1.0)
            self.depth.setDefault(15)

    def on_train_predict_clicked(self):
        ttype = ["rf", "erf", "ada", "gbf", "xgb"]
        params = {
            "clf": "ensemble",
            "type": ttype[self.type_combo.currentIndex()],
            "n_estimators": self.ntrees.value(),
            "max_depth": self.depth.value(),
            "learning_rate": self.lrate.value(),
            "subsample": self.subsample.value(),
            "n_jobs": self.n_jobs.value(),
        }
        self.train_predict.emit(params)

    def get_params(self):
        ttype = ["rf", "erf", "ada", "gbf", "xgb"]
        if self.type_combo.currentIndex() - 1 == 0:
            current_index = 0
        else:
            current_index = self.type_combo.currentIndex() - 1
        logger.debug(f"Ensemble type_combo index: {current_index}")
        params = {
            "clf": "ensemble",
            "type": ttype[current_index],
            "n_estimators": self.ntrees.value(),
            "max_depth": self.depth.value(),
            "learning_rate": self.lrate.value(),
            "subsample": self.subsample.value(),
            "n_jobs": self.n_jobs.value(),
        }
        return params


class PipelineCard(Card):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(
            fname, removable=True, editable=True, collapsible=True, parent=parent
        )
        self.pipeline_id = fid
        self.pipeline_type = ftype
        self.pipeline_name = fname
        self.annotations_source = None
        self.params = fparams
        self.widgets = dict()

        if self.pipeline_type == "superregion_segment":
            logger.debug("Adding a superregion_segment pipeline")
            self._add_features_source()
            self._add_annotations_source()
            self._add_constrain_source()
            self._add_regions_source()

            self.ensembles = EnsembleWidget()
            self.ensembles.train_predict.connect(self.compute_pipeline)
            self.svm = SVMWidget()
            self.svm.predict.connect(self.compute_pipeline)

            self._add_classifier_choice()
            self._add_projection_choice()
            self._add_param("lam", type="FloatSlider", default=0.15)
            self._add_confidence_choice()

        elif self.pipeline_type == "rasterize_points":
            self._add_annotations_source()
            self._add_feature_source()
            self._add_objects_source()

        elif self.pipeline_type == "watershed":
            self._add_annotations_source()
            self._add_feature_source()

        elif self.pipeline_type == "predict_segmentation_fcn":
            self._add_annotations_source()
            self._add_feature_source()
            self._add_model_file()
            self._add_model_type()
            # self._add_patch_params()
        elif self.pipeline_type == "label_postprocess":
            self._add_annotations_source(label="Layer Over: ")
            self._add_annotations_source2(label="Layer Base: ")
            self.label_index = LineEdit(default=-1, parse=int)
            #widget = HWidgets("Selected label:", self.label_index, Spacing(35), stretch=1)
            #self.add_row(widget)
            #self.offset = LineEdit(default=-1, parse=int)
            #widget2 = HWidgets("Offset:", self.offset, Spacing(35), stretch=1)
            #self.add_row(widget2)
        elif self.pipeline_type == "feature_postprocess":
            self._add_feature_source()
            self._add_feature_source2()
            self.label_index = LineEdit(default=-1, parse=int)
        elif self.pipeline_type == "per_object_cleaning":
            self._add_feature_source()
            self._add_objects_source()
        elif self.pipeline_type == "cleaning":
            self._add_feature_source()
            self._add_annotations_source()
        elif self.pipeline_type == "train_3d_fcn":
            self._add_annotations_source()
            self._add_feature_source()
            self._add_objects_source()
            self._add_3d_fcn_training_params()
            self._add_fcn_choice()
        elif self.pipeline_type == "predict_3d_fcn":
            self._add_annotations_source()
            self._add_feature_source()
            self._add_model_file()
            self._add_model_type()
            self._add_overlap_choice()
            # self._add_patch_params()
        elif self.pipeline_type == "train_2d_unet":
            self._add_2dunet_training_ws_widget()
            self._add_2dunet_annotations_features_from_ws(DataModel.g.current_workspace)
            self._add_2dunet_data_table()
            self._add_unet_2d_training_params()
        elif self.pipeline_type == "predict_2d_unet":
            self.annotations_source = LevelComboBox()
            self.annotations_source.hide()
            self._add_feature_source()
            self._add_unet_2d_prediction_params()

        else:
            logger.debug(f"Unsupported pipeline type {self.pipeline_type}.")

        for pname, params in fparams.items():
            if pname not in ["src", "dst"]:
                self._add_param(pname, **params)

        self._add_compute_btn()
        self._add_view_btn()


    def _add_2dunet_data_table(self):
        columns = ["Workspace", "Data", "Labels", ""]
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(columns)
        table_fields = QtWidgets.QGroupBox("Training Datasets")
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.addWidget(QLabel('Press "Compute" when list is complete.'))
        table_layout.addWidget(self.table)
        table_fields.setLayout(table_layout)
        self.add_row(table_fields, max_height=200)

    def table_delete_clicked(self):
        button = self.sender()
        if button:
            row = self.table.indexAt(button.pos()).row()
            self.table.removeRow(row)

    def _add_item_to_data_table(self, item):
        ws = DataTableWidgetItem(self.workspaces_list.currentText())
        ws.hidden_field = self.workspaces_list.value()
        data = DataTableWidgetItem(self.feature_source.currentText())
        data.hidden_field = self.feature_source.value()
        labels = DataTableWidgetItem(self.annotations_source.currentText())
        labels.hidden_field = self.annotations_source.value()
        self._add_data_row(ws, data, labels)

    def _add_data_row(self, ws, data, labels):
        row_pos = self.table.rowCount()
        self.table.insertRow(row_pos)
        self.table.setItem(row_pos, 0, ws)
        self.table.setItem(row_pos, 1, data)
        self.table.setItem(row_pos, 2, labels)
        delete_button = QtWidgets.QPushButton("Delete")
        delete_button.clicked.connect(self.table_delete_clicked)
        self.table.setCellWidget(row_pos, 3, delete_button)

    def _update_data_table_from_dict(self, data_dict):
        for ws, ds, lbl in zip(data_dict["Workspaces"], data_dict["Data"], data_dict["Labels"]):
            ws = ast.literal_eval(ws)
            ws_item = DataTableWidgetItem(ws[1])
            ws_item.hidden_field = ws[0]
            ds = ast.literal_eval(ds)
            ds_item = DataTableWidgetItem(ds[1])
            ds_item.hidden_field = ds[0]
            lbl = ast.literal_eval(lbl)
            lbl_item = DataTableWidgetItem(lbl[1])
            lbl_item.hidden_field = lbl[0]
            self._add_data_row(ws_item, ds_item, lbl_item)

    def _update_2d_unet_train_params(self, frozen, unfrozen):
        self.cycles_frozen.setText(str(frozen))
        self.cycles_unfrozen.setText(str(unfrozen))

    def _add_2dunet_training_ws_widget(self):
        data_label = QtWidgets.QLabel("Select training data:")
        self.add_row(data_label)
        self.workspaces_list = self._get_workspaces_list()
        self.workspaces_list.currentTextChanged.connect(self.on_ws_combobox_changed)
        ws_widget = HWidgets("Workspace:", self.workspaces_list, Spacing(35), stretch=1)
        self.add_row(ws_widget)

    def _add_2dunet_annotations_source(self):
        self.annotations_source = ComboBox()
        self.annotations_source.setMaximumWidth(250)
        anno_widget = HWidgets("Annotation (Labels):", self.annotations_source, Spacing(35), stretch=1)
        self.add_row(anno_widget)

    def _add_2dunet_feature_source(self):
        self.feature_source = ComboBox()
        self.feature_source.setMaximumWidth(250)
        feature_widget = HWidgets("Feature (Data):", self.feature_source, Spacing(35), stretch=1)
        self.add_row(feature_widget)

    def _update_annotations_from_ws(self, workspace):
        self.annotations_source.clear()
        params = {"workspace" : workspace}
        anno_result = Launcher.g.run("annotations", "get_levels", **params)
        logger.debug(f"anno_result: {anno_result}")
        if anno_result:
            for r in anno_result:
                if r["kind"] == "level":
                    self.annotations_source.addItem(r["id"], r["name"])
    
    def _update_features_from_ws(self, workspace):
        self.feature_source.clear()
        workspace = "default@" + workspace
        params = {"workspace" : workspace}
        logger.debug(f"Filling features from session: {params}")
        result = Launcher.g.run("features", "existing", **params)
        if result:
            for fid in result:
                self.feature_source.addItem(fid, result[fid]["name"])

    def _add_2dunet_annotations_features_from_ws(self, workspace):
        self._add_2dunet_feature_source()
        self._update_features_from_ws(workspace)
        self._add_2dunet_annotations_source()
        self._update_annotations_from_ws(workspace)
        train_data_btn = PushButton("Add to list", accent=True)
        train_data_btn.clicked.connect(self._add_item_to_data_table)
        widget = HWidgets(None, train_data_btn, Spacing(35), stretch=0)
        self.add_row(widget)

    def _get_workspaces_list(self):
        workspaces = [d for d in next(os.walk(DataModel.g.CHROOT))[1]]
        workspaces_list = ComboBox()
        workspaces_list.setMaximumWidth(250)
        for s in workspaces:
            workspaces_list.addItem(key=s)
        workspaces_list.setCurrentText(DataModel.g.current_workspace)
        return workspaces_list

    def on_ws_combobox_changed(self, workspace):
        self._update_annotations_from_ws(workspace)
        self._update_features_from_ws(workspace)

    def _add_model_type(self):
        self.model_type = ComboBox()
        self.model_type.addItem(key="unet3d")
        self.model_type.addItem(key="fpn3d")
        widget = HWidgets("Model type:", self.model_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_patch_params(self):
        self.patch_size = LineEdit3D(default=64, parse=int)
        self.add_row(HWidgets("Patch Size:", self.patch_size, Spacing(35), stretch=1))

    def _add_unet_2d_training_params(self):
        self.add_row(HWidgets("Training Parameters:", Spacing(35), stretch=1))
        self.cycles_frozen = LineEdit(default=8, parse=int)
        self.cycles_unfrozen = LineEdit(default=5, parse=int)
        refresh_label = Label('Please: 1. "Compute", 2. "Refresh Data", 3. Reopen dialog and "View".')
        self.unet_train_refresh_btn = PushButton("Refresh Data", accent=True)
        self.unet_train_refresh_btn.clicked.connect(self.refresh_unet_data)
        self.unet_pred_refresh_btn = None
        self.add_row(HWidgets("No. Cycles Frozen:", self.cycles_frozen,
                              "No. Cycles Unfrozen", self.cycles_unfrozen,
                              stretch=1))
        self.add_row(HWidgets(refresh_label, stretch=1))
        self.add_row(HWidgets(self.unet_train_refresh_btn, stretch=1))
        
    def _add_unet_2d_prediction_params(self):
        self.model_file_line_edit = LineEdit(default="Filepath", parse=str)
        model_input_btn = PushButton("Select Model", accent=True)
        model_input_btn.clicked.connect(self.get_model_path)
        self.radio_group = QtWidgets.QButtonGroup()
        self.radio_group.setExclusive(True)
        single_pp_rb = QRadioButton("Single plane")
        single_pp_rb.setChecked(True)
        self.radio_group.addButton(single_pp_rb, 1)
        triple_pp_rb = QRadioButton("Three plane")
        self.radio_group.addButton(triple_pp_rb, 3)
        refresh_label = Label('Please: 1. "Compute", 2. "Refresh Data", 3. Reopen dialog and "View".')
        self.unet_pred_refresh_btn = PushButton("Refresh Data", accent=True)
        self.unet_pred_refresh_btn.clicked.connect(self.refresh_unet_data)
        self.unet_train_refresh_btn = None
        self.add_row(HWidgets(self.model_file_line_edit, model_input_btn, Spacing(35)))
        self.add_row(HWidgets("Prediction Parameters:", Spacing(35), stretch=1))
        self.add_row(HWidgets(single_pp_rb, triple_pp_rb, stretch=1))
        self.add_row(HWidgets(refresh_label, stretch=1))
        self.add_row(HWidgets(self.unet_pred_refresh_btn, stretch=1))
    def _add_3d_fcn_training_params(self):
        pass
    def _add_3d_fcn_prediction_params(self):
        pass

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def load_data(self, path):
        self.model_fullname = path
        print(f"Setting model fullname: {self.model_fullname}")

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_pipeline)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as image", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)
        self.add_row(
            HWidgets(
                None, load_as_float_btn, load_as_annotation_btn, view_btn, Spacing(35)
            )
        )



    def _add_refine_choice(self):
        self.refine_checkbox = CheckBox(checked=True)
        self.add_row(
            HWidgets("MRF Refinement:", self.refine_checkbox, Spacing(35), stretch=0)
        )

    def _add_confidence_choice(self):
        self.confidence_checkbox = CheckBox(checked=False)
        self.add_row(
            HWidgets("Confidence Map as Feature:", self.confidence_checkbox, Spacing(35), stretch=0)
        )

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)

        widget = HWidgets("Objects:", self.objects_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_classifier_choice(self):
        self.classifier_type = ComboBox()
        self.classifier_type.addItem(key="Ensemble")
        self.classifier_type.addItem(key="SVM")
        widget = HWidgets("Classifier:", self.classifier_type, Spacing(35), stretch=0)

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
        widget = HWidgets("FCN Type:", self.fcn_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_projection_choice(self):
        self.projection_type = ComboBox()
        self.projection_type.addItem(key="None")
        self.projection_type.addItem(key="pca")
        self.projection_type.addItem(key="rbp")
        self.projection_type.addItem(key="rproj")
        self.projection_type.addItem(key="std")
        widget = HWidgets("Projection:", self.projection_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_feature_source2(self):
        self.feature_source2 = FeatureComboBox()
        self.feature_source2.fill()
        self.feature_source2.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source2, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)
        cfg.pipelines_features_source = self.features_source
        widget = HWidgets("Features:", self.features_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_constrain_source(self):
        print(self.annotations_source.value())
        self.constrain_mask_source = AnnotationComboBox(
            header=(None, "None"), full=True
        )
        self.constrain_mask_source.fill()
        self.constrain_mask_source.setMaximumWidth(250)

        widget = HWidgets(
            "Constrain mask:", self.constrain_mask_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

    def _add_annotations_source(self, label="Annotation"):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source, Spacing(35), stretch=1)

        self.add_row(widget)

    def _add_annotations_source2(self, label="Annotation 2"):
        self.annotations_source2 = LevelComboBox(full=True)
        self.annotations_source2.fill()
        self.annotations_source2.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source2, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_pipelines_source(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets(
            "Segmentation:", self.pipelines_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

    def _add_regions_source(self):
        self.regions_source = RegionComboBox(full=True)  # SourceComboBox()
        self.regions_source.fill()
        self.regions_source.setMaximumWidth(250)

        widget = HWidgets("Superregions:", self.regions_source, Spacing(35), stretch=1)
        cfg.pipelines_regions_source = self.regions_source
        self.add_row(widget)
    def _add_overlap_choice(self):
        self.overlap_type = ComboBox()
        self.overlap_type.addItem(key="crop")
        self.overlap_type.addItem(key="average")
        widget = HWidgets("Overlap:", self.overlap_type, Spacing(35), stretch=0)
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
            self.add_row(HWidgets(None, title, p, Spacing(35)))

    def _add_compute_btn(self):
        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self.compute_pipeline)
        self.add_row(HWidgets(None, compute_btn, Spacing(35)))

    def update_params(self, params):
        logger.debug(f"Pipeline update params {params}")
        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)
        if "anno_id" in params:
            if params["anno_id"] is not None:
                if isinstance(params["anno_id"], list):
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"][0])
                    )
                else:
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"])
                    )
        
        if "object_id" in params:
            if params["object_id"] is not None:
                self.objects_source.select(
                    os.path.join("objects/", params["object_id"])
                )
        if "feature_id" in params:
            #self.feature_source.select(os.path.join("features/", params["feature_id"]))
            self.feature_source.select(params["feature_id"])
        if "feature_ids" in params:
            for source in params["feature_ids"]:
                self.features_source.select(os.path.join("features/", source))
        
        if "feature_A" in params:
            self.feature_source.select(params["feature_A"])

        if "feature_B" in params:
            self.feature_source2.select(params["feature_B"])

        if "level_over" in params:
            print("level over found")
            self.annotations_source.select(os.path.join("annotations/", params["level_over"]))

        if "level_base" in params:
            print("level_base found")
            self.annotations_source2.select(os.path.join("annotations/", params["level_base"]))

        if "region_id" in params:
            if params["region_id"] is not None:
                self.regions_source.select(
                    os.path.join("superregions/", params["region_id"])
                )
            
        if "constrain_mask" in params:
            if (
                params["constrain_mask"] is not None
                and params["constrain_mask"] != "None"
            ):
                import ast

                constrain_mask_dict = ast.literal_eval(params["constrain_mask"])
                print(constrain_mask_dict)

                constrain_mask_source = (
                    constrain_mask_dict["level"] + ":" + str(constrain_mask_dict["idx"])
                )
                print(f"Constrain mask source {constrain_mask_source}")
                self.constrain_mask_source.select(constrain_mask_source)
        if "unet_train_params" in params:
            if params["anno_id"]:
                table_dict = dict(Labels=params["anno_id"],
                                Data=params["feature_id"],
                                Workspaces=params["workspace"])
                self._update_data_table_from_dict(table_dict)
                self._update_2d_unet_train_params(
                    params["unet_train_params"]["cyc_frozen"],
                    params["unet_train_params"]["cyc_unfrozen"]
                    )

    def card_deleted(self):
        params = dict(pipeline_id=self.pipeline_id, workspace=True)
        result = Launcher.g.run("pipelines", "remove", **params)
        if result["done"]:
            self.setParent(None)
            _PipelineNotifier.notify()

        cfg.ppw.clientEvent.emit(
            {
                "source": "pipelines",
                "data": "remove_layer",
                "layer_name": self.pipeline_id,
            }
        )

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
                    level_id = '001_level'
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
                        "level_id": '001_level',
                    }
                )

            pbar.update(1)
          
    def get_model_path(self):
        workspace_path = os.path.join(DataModel.g.CHROOT, DataModel.g.current_workspace)
        self.model_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
        ("Select model"), workspace_path, ("Model files (*.zip)"))
        self.model_file_line_edit.setValue(self.model_path)

    def refresh_unet_data(self):
        sender = self.sender()
        cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "refresh", "value": None}
            )
        if sender == self.unet_pred_refresh_btn:
            search_str = "U-Net prediction"
            self.annotations_source.fill()
        elif sender == self.unet_train_refresh_btn:
            search_str = "U-Net Training"
            self._update_annotations_from_ws(DataModel.g.current_workspace)
        index = self.annotations_source.findText(search_str, QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.annotations_source.setCurrentText(search_str)
            print(f"Setting index to {index}")
        else:
            num_annotations = self.annotations_source.count()
            self.annotations_source.setCurrentIndex(num_annotations -1)
        print(f"Annotations source selected: {self.annotations_source.value()}")

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
                {"source": "workspace_gui", "data": "refresh_plugin", "plugin_name" : "features"}
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
            anno_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            params = dict(level=str(level_id), workspace=True)
            level_result = Launcher.g.run("annotations", "get_levels", **params)[0]

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
                {"source": "workspace_gui", "data": "faster_refresh_plugin", "plugin_name": "annotations"}
            )

    def setup_params_superregion_segment(self, dst):
        feature_names_list = [
            n.rsplit("/", 1)[-1] for n in self.features_source.value()
        ]
        src_grp = None if self.annotations_source.currentIndex() == 0 else "pipelines"
        src = DataModel.g.dataset_uri(
            self.annotations_source.value().rsplit("/", 1)[-1],
            group="annotations",
        )
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace

        logger.info(f"Setting src to {self.annotations_source.value()} ")
        all_params["region_id"] = str(self.regions_source.value().rsplit("/", 1)[-1])
        all_params["feature_ids"] = feature_names_list
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        if self.constrain_mask_source.value() != None:
            all_params[
                "constrain_mask"
            ] = self.constrain_mask_source.value()  # .rsplit("/", 1)[-1]
        else:
            all_params["constrain_mask"] = "None"

        all_params["dst"] = dst
        all_params["refine"] = self.widgets["refine"].value()  
        all_params["lam"] = self.widgets["lam"].value()
        all_params["classifier_type"] = self.classifier_type.value()
        all_params["projection_type"] = self.projection_type.value()
        all_params["confidence"] = self.confidence_checkbox.value()

        if self.classifier_type.value() == "Ensemble":
            all_params["classifier_params"] = self.ensembles.get_params()
        else:
            all_params["classifier_params"] = self.svm.get_params()
        return all_params

    def setup_params_rasterize_points(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        # all_params["anno_id"] = str(
        #    self.annotations_source.value().rsplit("/", 1)[-1]
        # )
        all_params["feature_id"] = self.feature_source.value()
        all_params["object_id"] = str(self.objects_source.value())
        all_params["acwe"] = self.widgets["acwe"].value()
        # all_params["object_scale"] = self.widgets["object_scale"].value()
        # all_params["object_offset"] = self.widgets["object_offset"].value()
        all_params["dst"] = dst
        return all_params

    def setup_params_watershed(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["dst"] = self.pipeline_id
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        return all_params


    def setup_params_feature_postprocess(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_A"] = str(self.feature_source.value())
        all_params["feature_B"] = str(self.feature_source2.value())
        all_params["dst"] = dst
        return all_params

    def setup_params_label_postprocess(self, dst):
        src = DataModel.g.dataset_uri(self.annotations_source.value().rsplit("/", 1)[-1], group="annotations")
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        print(self.annotations_source.value())

        if(self.annotations_source.value()):
            all_params["level_over"] = str(
                self.annotations_source.value().rsplit("/", 1)[-1]
            )
        else:
            all_params["level_over"] = "None"
        all_params["level_base"] = str(
            self.annotations_source2.value().rsplit("/", 1)[-1]
        )
        all_params["dst"] = dst
        all_params["selected_label"] = int(self.widgets["selected_label"].value())
        all_params["offset"] = int(self.widgets["offset"].value())
        return all_params

    def setup_params_cleaning(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        return all_params

    def setup_params_per_object_cleaning(self, dst):
        all_params = dict(dst=dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["patch_size"] = self.widgets["patch_size"].value()
        return all_params

    def setup_params_train_3d_fcn(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        if self.objects_source.value() != None:
            all_params["objects_id"] = str(self.objects_source.value().rsplit("/", 1)[-1])
            #all_params["objects_id"] = str(self.objects_source.value()
            print(all_params["objects_id"])
        else:
            #     all_params["objects_id"] = str(self.objects_source.value())
            all_params["objects_id"] = "None"
        all_params["fpn_train_params"] = {}
        all_params["fcn_type"] = self.fcn_type.value()
        
        return all_params

    def setup_params_predict_3d_fcn(self, dst):
        src = DataModel.g.dataset_uri(
            self.feature_source.value(), group="features"
        )
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        all_params["feature_id"] = self.feature_source.value()
        all_params["model_fullname"] = self.model_fullname
        all_params["model_type"] = self.model_type.value()
        all_params["dst"] = dst
        all_params["overlap_mode"] = self.overlap_type.value()
        
        return all_params
    
    def setup_params_train_2d_unet(self, dst):
        # Retrieve params from table
        num_rows = self.table.rowCount()
        if num_rows == 0:
            logging.error("No data selected for training!")
            return
        else:
            workspace_list = []
            data_list = []
            label_list = []
            for i in range(num_rows):
                workspace_list.append((self.table.item(i, 0).get_hidden_field(),
                self.table.item(i, 0).text()))
                data_list.append((self.table.item(i, 1).get_hidden_field(),
                self.table.item(i, 1).text()))
                label_list.append((self.table.item(i, 2).get_hidden_field(),
                self.table.item(i, 2).text()))
        # Can the src parameter be removed?        
        src = DataModel.g.dataset_uri("001_raw", group="features")
        all_params = dict(src=src, dst=dst, modal=True)
        all_params["workspace"] = workspace_list
        all_params["feature_id"] = data_list
        all_params["anno_id"] = label_list
        all_params["unet_train_params"] = dict(cyc_frozen=self.cycles_frozen.value(),
                                               cyc_unfrozen=self.cycles_unfrozen.value())
        return all_params

    def setup_params_predict_2d_unet(self, dst):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["model_path"] = str(self.model_file_line_edit.value())
        all_params["no_of_planes"] = self.radio_group.checkedId()
        return all_params
        


    def compute_pipeline(self):
        dst = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")
        
        with progress(total=3) as pbar:
            pbar.set_description("Calculating pipeline")
            pbar.update(1)
            try:
                if self.pipeline_type == "superregion_segment":
                    all_params = self.setup_params_superregion_segment(dst)
                elif self.pipeline_type == "rasterize_points":
                    all_params = self.setup_params_rasterize_points(dst)
                elif self.pipeline_type == "watershed":
                    all_params = self.setup_params_watershed(dst)
                elif self.pipeline_type == "label_postprocess":
                    all_params = self.setup_params_label_postprocess(dst)
                elif self.pipeline_type == "feature_postprocess":
                    all_params = self.setup_params_feature_postprocess(dst)
                elif self.pipeline_type == "cleaning":
                    all_params = self.setup_params_cleaning(dst)
                elif self.pipeline_type == "per_object_cleaning":
                    all_params = self.setup_params_per_object_cleaning(dst)
                elif self.pipeline_type == "train_2d_unet":
                    all_params = self.setup_params_train_2d_unet(dst)
                elif self.pipeline_type == "predict_2d_unet":
                    all_params = self.setup_params_predict_2d_unet(dst)
                elif self.pipeline_type == "train_3d_fcn":
                    all_params = self.setup_params_train_3d_fcn(dst)
                elif self.pipeline_type == "predict_3d_fcn":
                    all_params = self.setup_params_predict_3d_fcn(dst)                
                else:
                    logger.warning(f"No action exists for pipeline: {self.pipeline_type}")

                all_params.update({k: v.value() for k, v in self.widgets.items()})

                logger.info(f"Computing pipelines {self.pipeline_type} {all_params}")
                try:
                    pbar.update(1)
                    result = Launcher.g.run("pipelines", self.pipeline_type, **all_params)
                    print(result)
                except Exception as err:
                    print(err)
                if result is not None:
                    pbar.update(1)

            except Exception as e:
                print(e)

    def card_title_edited(self, newtitle):
        params = dict(pipeline_id=self.pipeline_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("pipelines", "rename", **params)

        if result["done"]:
            _PipelineNotifier.notify()

        return result["done"]



