import numpy as np
from loguru import logger
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal

from survos2.frontend.components.base import *
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotation_tool import AnnotationComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import ComboBox, LazyComboBox, LazyMultiComboBox
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox, RealSlider
from survos2.frontend.plugins.regions import RegionComboBox
from survos2.frontend.utils import (
    get_array_from_dataset,
    get_color_mapping,
    hex_string_to_rgba,
)
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.utils import FileWidget

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
        logger.debug(f"Result of pipelines existing: {result}")
        if result:
            self.addCategory("Segmentations")
            for fid in result:
                if result[fid]["kind"] == "pipelines":
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

        result = None
        result = Launcher.g.run("pipelines", "available", workspace=True)

        if not result:
            params = {}
            params["category"] = "superregion"
            params["name"] = "s0"
            params["type"] = "superregion_segment"
            result = {}
            result[0] = params
            self.pipeline_params["superregion_segment"] = {
                "sr_params": {"type": "sr2",}
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
        if idx == 0:
            return
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

    def setup(self):
        params = dict(workspace=True)
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
        self.type_combo.addCategory('Kernel Type:')
        self.type_combo.addItem('linear')
        self.type_combo.addItem('poly')
        self.type_combo.addItem('rbf')
        self.type_combo.addItem('sigmoid')
        vbox.addWidget(self.type_combo)

        self.penaltyc = LineEdit(default=1.0, parse=float)
        self.gamma = LineEdit(default=1.0, parse=float)

        vbox.addWidget(HWidgets(QtWidgets.QLabel('Penalty C:'),
                                self.penaltyc,
                                QtWidgets.QLabel('Gamma:'),
                                self.gamma,
                                stretch=[0, 1, 0, 1]))

        #self.btn_predict = PushButton('Predict')
        #self.btn_predict.clicked.connect(self.on_predict_clicked)
        #vbox.addWidget(HWidgets(None, self.btn_predict, stretch=[1, 0]))

    def on_predict_clicked(self):
        params = {
            'clf': 'svm',
            'kernel': self.type_combo.currentText(),
            'C': self.penaltyc.value(),
            'gamma': self.gamma.value()
        }

        self.predict.emit(params)


    def get_params(self):
        params = {
            'clf': 'svm',
            'kernel': self.type_combo.currentText(),
            'C': self.penaltyc.value(),
            'gamma': self.gamma.value()
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
        self.type_combo.addCategory('Ensemble Type:') 
        self.type_combo.addItem('Random Forest')
        self.type_combo.addItem('ExtraRandom Forest')
        self.type_combo.addItem('AdaBoost')
        self.type_combo.addItem('GradientBoosting')
    
        self.type_combo.currentIndexChanged.connect(self.on_ensemble_changed)
        vbox.addWidget(self.type_combo)

        self.ntrees = LineEdit(default=100, parse=int)
        self.depth = LineEdit(default=15, parse=int)
        self.lrate = LineEdit(default=1., parse=float)
        self.subsample = LineEdit(default=1., parse=float)

        vbox.addWidget(HWidgets(QtWidgets.QLabel('# Trees:'),
                                self.ntrees,
                                QtWidgets.QLabel('Max Depth:'),
                                self.depth,
                                stretch=[0, 1, 0, 1]))

        vbox.addWidget(HWidgets(QtWidgets.QLabel('Learn Rate:'),
                                self.lrate,
                                QtWidgets.QLabel('Subsample:'),
                                self.subsample,
                                stretch=[0, 1, 0, 1]))

        #self.btn_train_predict = PushButton('Train & Predict')
        #self.btn_train_predict.clicked.connect(self.on_train_predict_clicked)
        self.n_jobs = LineEdit(default=1, parse=int)
        vbox.addWidget(HWidgets('Num Jobs', self.n_jobs))
        
    def on_ensemble_changed(self, idx):
        if idx == 2:
            self.ntrees.setDefault(50)
        else:
            self.ntrees.setDefault(100)

        if idx == 3:
            self.lrate.setDefault(0.1)
            self.depth.setDefault(3)
        else:
            self.lrate.setDefault(1.)
            self.depth.setDefault(15)

    def on_train_predict_clicked(self):
        ttype = ['rf', 'erf', 'ada', 'gbf']
        params = {
            'clf': 'ensemble',
            'type': ttype[self.type_combo.currentIndex()],
            'n_estimators': self.ntrees.value(),
            'max_depth': self.depth.value(),
            'learning_rate': self.lrate.value(),
            'subsample': self.subsample.value(),
            'n_jobs': self.n_jobs.value()
        }
        self.train_predict.emit(params)

    def get_params(self):
        ttype = ['rf', 'erf', 'ada', 'gbf']
        if self.type_combo.currentIndex()-1 == 0:
            current_index = 0
        else:
            current_index = self.type_combo.currentIndex()-1
        logger.debug(f"Ensemble type_combo index: {current_index}")
        params = {
            'clf': 'ensemble',
            'type': ttype[current_index],
            'n_estimators': self.ntrees.value(),
            'max_depth': self.depth.value(),
            'learning_rate': self.lrate.value(),
            'subsample': self.subsample.value(),
            'n_jobs': self.n_jobs.value()
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

        from qtpy.QtWidgets import QProgressBar

        self.pbar = QProgressBar(self)
        self.add_row(self.pbar)

        self.params = fparams
        print(fparams)
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
            #self._add_refine_choice()
            self._add_param("lam", type="FloatSlider", default=0.15)
            
        elif self.pipeline_type == "make_annotation":
            self._add_annotations_source()
            self._add_features_source()
            self._add_objects_source()

        elif self.pipeline_type == "predict_segmentation_fcn":
            self._add_annotations_source()
            self._add_features_source()
            self._add_objects_source()
            self._add_workflow_file()
            self._add_model_type()
            # self._add_patch_params()

        else:
            logger.debug(f"Unsupported pipeline type {self.pipeline_type}.")

        for pname, params in fparams.items():
            if pname not in ["src", "dst"]:
                self._add_param(pname, **params)

        self._add_compute_btn()
        self._add_view_btn()

    def _add_model_type(self):
        self.model_type = ComboBox()
        self.model_type.addItem(key="unet3d")
        self.model_type.addItem(key="fpn3d")
        widget = HWidgets("Model type:", self.model_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_patch_params(self):
        self.patch_size = LineEdit3D(default=64, parse=int)
        self.add_row(HWidgets("Patch Size:", self.patch_size, Spacing(35), stretch=1))

    def _add_workflow_file(self):
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
        self.add_row(self.clf_container,  max_height=500)
        self.clf_container.layout().addWidget(self.ensembles)
        
    def _on_classifier_changed(self, idx):
        if idx==0:
            self.clf_container.layout().addWidget(self.ensembles)
            self.svm.setParent(None)

        elif idx==1:
            self.clf_container.layout().addWidget(self.svm)        
            self.ensembles.setParent(None)

    def _add_projection_choice(self):
        self.projection_type = ComboBox()
        self.projection_type.addItem(key="None")
        self.projection_type.addItem(key="pca")
        self.projection_type.addItem(key="rbp")
        self.projection_type.addItem(key="rproj")
        self.projection_type.addItem(key="std")
        widget = HWidgets("Projection:", self.projection_type, Spacing(35), stretch=0)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_constrain_source(self):
        print(self.annotations_source.value())
        self.constrain_mask_source = AnnotationComboBox(full=True)
        self.constrain_mask_source.fill()
        self.constrain_mask_source.setMaximumWidth(250)

        widget = HWidgets(
            "Constrain mask:", self.constrain_mask_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

    def _add_annotations_source(self):
        self.annotations_source = LevelComboBox(full=True)  
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(
            "Annotation:", self.annotations_source, Spacing(35), stretch=1
        )

        self.add_row(widget)

    def _add_regions_source(self):
        self.regions_source = RegionComboBox(full=True)  # SourceComboBox()
        self.regions_source.fill()
        self.regions_source.setMaximumWidth(250)

        widget = HWidgets("Superregions:", self.regions_source, Spacing(35), stretch=1)

        self.add_row(widget)

    def _add_param(self, name, title=None, type="String", default=None):
        if type == "Int":
            p = LineEdit(default=default, parse=int)
        elif type == "FloatSlider":
            p = RealSlider(value=default, vmax=1, vmin=0)
            title = "MRF Refinement Amount:"
        elif type == "Float":
            p = LineEdit(default=0.0, parse=float)
            title = title
        elif type == "FloatOrVector":
            p = LineEdit3D(default=0, parse=float)
        elif type == "IntOrVector":
            p = LineEdit3D(default=default, parse=int)
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
        if "source" in params:
            for source in params["source"]:
                self.features_source.select(source)

        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)

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
        if self.annotations_source.value() is not None:
            level_id = str(self.annotations_source.value().rsplit("/", 1)[-1])
            logger.debug(f"Assigning annotation level {level_id} ti")
            str(self.annotations_source.value().rsplit("/", 1)[-1])
            cfg.ppw.clientEvent.emit(
                {
                    "source": "pipelines",
                    "data": "view_pipeline",
                    "pipeline_id": self.pipeline_id,
                    "level_id": level_id,
                }
            )

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
                        label = dict(idx=int(v), name=str(v), color=label_hex,)
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
                {"source": "workspace_gui", "data": "refresh", "value": None}
            )

    def compute_pipeline(self):
        dst = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")
        feature_names_list = [
            n.rsplit("/", 1)[-1] for n in self.features_source.value()
        ]

        if self.pipeline_type == "superregion_segment":
            src_grp = (
                None if self.annotations_source.currentIndex() == 0 else "pipelines"
            )
            src = DataModel.g.dataset_uri(
                self.annotations_source.value().rsplit("/", 1)[-1], group="annotations"
            )
            all_params = dict(src=src, modal=True)
            all_params["workspace"] = DataModel.g.current_workspace

            logger.info(f"Setting src to {self.annotations_source.value()} ")
            all_params["region_id"] = str(
                self.regions_source.value().rsplit("/", 1)[-1]
            )
            all_params["feature_ids"] = feature_names_list
            all_params["anno_id"] = str(
                self.annotations_source.value().rsplit("/", 1)[-1]
            )
            print(self.constrain_mask_source.value())
            if self.constrain_mask_source.value() != None:
                all_params["constrain_mask"] = self.constrain_mask_source.value() #.rsplit("/", 1)[-1]
            else:
                all_params["constrain_mask"] = "None"
            all_params["dst"] = dst
            all_params["refine"] = self.widgets["refine"].value() #self.refine_checkbox.value()
            all_params["lam"] = self.widgets["lam"].value()
            all_params["classifier_type"] = self.classifier_type.value()
            all_params["projection_type"] = self.projection_type.value()
        
            if self.classifier_type.value() == 'Ensemble':    
                all_params["classifier_params"] = self.ensembles.get_params()
            else:
                all_params["classifier_params"] = self.svm.get_params()
            

        elif self.pipeline_type == "make_annotation":
            src = DataModel.g.dataset_uri(
                self.features_source.value()[0], group="pipelines"
            )
            all_params = dict(src=src, dst=dst, modal=True)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["anno_id"] = str(
                self.annotations_source.value().rsplit("/", 1)[-1]
            )
            all_params["feature_ids"] = feature_names_list
            all_params["object_id"] = str(self.objects_source.value())
            all_params["acwe"] = self.widgets["acwe"].value()
            #all_params["object_scale"] = self.widgets["object_scale"].value()
            #all_params["object_offset"] = self.widgets["object_offset"].value()
            all_params["dst"] = self.pipeline_id

        elif self.pipeline_type == "predict_segmentation_fcn":
            src = DataModel.g.dataset_uri(
                self.features_source.value()[0], group="pipelines"
            )
            all_params = dict(src=src, dst=dst, modal=True)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["anno_id"] = str(
                self.annotations_source.value().rsplit("/", 1)[-1]
            )
            all_params["feature_ids"] = feature_names_list
            all_params["model_fullname"] = self.model_fullname
            all_params["model_type"] = self.model_type.value()
            all_params["dst"] = self.pipeline_id

        print(self.widgets.items())
        #all_params.update({k: v.value() for k, v in self.widgets.items()})

        logger.info(f"Computing pipelines {self.pipeline_type} {all_params}")

        self.pbar.setValue(20)
        result = Launcher.g.run("pipelines", self.pipeline_type, **all_params)

        if result is not None:
            self.pbar.setValue(100)

    def card_title_edited(self, newtitle):
        params = dict(pipeline_id=self.pipeline_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("pipelines", "rename", **params)

        if result["done"]:
            _PipelineNotifier.notify()

        return result["done"]
