import numpy as np
from loguru import logger
import os
from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton,QFileDialog
from qtpy.QtCore import QSize, Signal

from survos2.frontend.components.base import *
from survos2.frontend.plugins.base import *
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.plugins_components import SourceComboBox
from survos2.server.state import cfg
from survos2.frontend.utils import FileWidget
from napari.qt.progress import progress
from survos2.improc.utils import DatasetManager
import yaml
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
_FeatureNotifier = PluginNotifier()


class FeatureComboBox(LazyComboBox):
    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(header=(None, "None"), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, full=self.full)


def _fill_features(combo, full=False, filter=True, ignore='001 Raw'):
    params = dict(
        workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        full=full,
        filter=filter,
    )
    logger.debug(f"Filling features from session: {params}")
    result = Launcher.g.run("features", "existing", **params)

    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, suptitle="Feature"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout(pad=60)
        self.axes = self.fig.add_subplot(111)
        self.suptitle = suptitle
        super(MplCanvas, self).__init__(self.fig)
    def set_suptitle(self,suptitle):
        self.suptitle = suptitle
        self.fig.suptitle(suptitle)



@register_plugin
class FeaturesPlugin(Plugin):
    __icon__ = "fa.picture-o"
    __pname__ = "features"
    __views__ = ["slice_viewer"]
    __tab__ = "features"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.feature_combo = ComboBox()
        self.vbox = VBox(self, spacing=4)
        self.vbox2 = VBox(self, spacing=4)

        self.vbox.addWidget(self.feature_combo)
        self.feature_combo.currentIndexChanged.connect(self.add_feature)
        self.existing_features = dict()
        
        self._populate_features()

        self.vbox.addLayout(self.vbox2)
        self.workflow_button = PushButton("Save workflow", accent=True)
        self.workflow_button.clicked.connect(self.save_workflow)
       
        self.filewidget = FileWidget(extensions="*.yaml", save=False)
        self.filewidget_open = FileWidget(extensions="*.yaml", save=False)
        self.filewidget_open.path_updated.connect(self.load_workflow)

        button_runworkflow = QPushButton("Run workflow", self)
        button_runworkflow.clicked.connect(self.button_runworkflow_clicked)
    
        hbox_layout2 = QtWidgets.QHBoxLayout()
        hbox_layout2.addWidget(self.workflow_button)
        hbox_layout2.addWidget(self.filewidget_open)
        hbox_layout2.addWidget(button_runworkflow)
        self.vbox.addLayout(hbox_layout2)
        
        

    def load_workflow(self, path):
        self.workflow_fullname = path
        print(f"Setting workflow fullname: {self.workflow_fullname}")

    def button_runworkflow_clicked(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "panel_gui",
                "data": "run_workflow",
                "workflow_file": self.workflow_fullname,
            }
        )

    def _populate_features(self):
        self.feature_params = {}
        self.feature_combo.clear()
        self.feature_combo.addItem("Add feature")
        result = Launcher.g.run("features", "available", workspace=True)

        if result:
            all_categories = sorted(set(p["category"] for p in result))
            for i, category in enumerate(all_categories):
                self.feature_combo.addItem(category)
                self.feature_combo.model().item(
                    i + len(self.feature_params) + 1
                ).setEnabled(False)
                for f in [p for p in result if p["category"] == category]:
                    self.feature_params[f["name"]] = f["params"]
                    self.feature_combo.addItem(f["name"])

    def add_feature(self, idx):
        if idx <= 0:
            return
        logger.debug(f"Adding feature {self.feature_combo.itemText(idx)}")
        if self.feature_combo.itemText(idx) == "":
            return

        feature_type = self.feature_combo.itemText(idx)
        self.feature_combo.setCurrentIndex(0)
        params = dict(feature_type=feature_type, workspace=True)
        result = Launcher.g.run("features", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            self._add_feature_widget(fid, ftype, fname, True)
            _FeatureNotifier.notify()

    def _add_feature_widget(self, fid, ftype, fname, expand=False):
        if ftype in self.feature_params:
            widget = FeatureCard(fid, ftype, fname, self.feature_params[ftype])
            widget.showContent(expand)
            self.vbox2.addWidget(widget)
            self.existing_features[fid] = widget
            return widget

    def clear(self):
        for feature in list(self.existing_features.keys()):
            self.existing_features.pop(feature).setParent(None)
        self.existing_features = {}

    def save_workflow(self):
        fname_filter, ext = "YAML (*.yaml)", ".yaml"
        filename = "workflow" + ext
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Workflow", filename, fname_filter
        )
        if path is not None and len(path) > 0:
            workflow = {}
            for i,(k,v) in enumerate(self.existing_features.items()):
                for x,y in v.widgets.items():
                    print(x,y.value())
                    workflow["f"+str(i)] = {}
                    workflow["f"+str(i)]["action"] = "features." + str(v.feature_type)
                    workflow["f"+str(i)]["src"] = "001_raw"
                    workflow["f"+str(i)]["dst"] = "00" + str(i) + "_" + v.feature_type
                    workflow["f"+str(i)]["params"] = {}
                    param_value = y.value()
                    if isinstance(param_value, tuple):
                        param_value = list(param_value)
                    workflow["f"+str(i)]["params"][x] = param_value 
            
            workflow_yaml = path #os.path.join(os.path.dirname(__file__), "../../..", "workflow.yaml")
            with open(workflow_yaml, "w") as outfile:
                yaml.dump(workflow, outfile, default_flow_style=False)




    def setup(self):
        self._populate_features()
        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )
        result = Launcher.g.run("features", "existing", **params)
        logger.debug(f"Feature result {result}")

        if result:
            # Remove features that no longer exist in the server
            for feature in list(self.existing_features.keys()):
                if feature not in result:
                    self.existing_features.pop(feature).setParent(None)
            # Populate with new features if any
            for feature in sorted(result):
                if feature in self.existing_features:
                    continue
                params = result[feature]
                logger.debug(f"Feature params {params}")
                fid = params.pop("id", feature)
                ftype = params.pop("kind")
                fname = params.pop("name", feature)

                logger.debug(f"fid {fid} ftype {ftype} fname {fname}")
                if params.pop("kind", "unknown") != "unknown":

                    widget = self._add_feature_widget(fid, ftype, fname)
                    widget.update_params(params)
                    self.existing_features[fid] = widget

                else:
                    logger.debug(
                        "+ Skipping loading feature: {}, {}".format(fid, fname)
                    )
                    if ftype:
                        widget = self._add_feature_widget(fid, ftype, fname)
                        if widget:
                            widget.update_params(params)
                            self.existing_features[fid] = widget


class FeatureCard(CardWithId):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        self.feature_id = fid
        self.feature_type = ftype
        self.feature_name = fname

        super().__init__(
            fname, fid, removable=True, editable=True, collapsible=True, parent=parent
        )

        self.params = fparams
        self.widgets = dict()
        

        if self.feature_type == "wavelet":
            self._add_source()
            self.wavelet_type = ComboBox()
            self.wavelet_type.addItem(key="sym2")
            self.wavelet_type.addItem(key="sym3")
            self.wavelet_type.addItem(key="sym4")
            self.wavelet_type.addItem(key="sym6")
            self.wavelet_type.addItem(key="sym7")
            self.wavelet_type.addItem(key="sym8")
            self.wavelet_type.addItem(key="sym9")
            self.wavelet_type.addItem(key="haar")
            self.wavelet_type.addItem(key="db3")
            self.wavelet_type.addItem(key="db4")
            self.wavelet_type.addItem(key="db5")
            self.wavelet_type.addItem(key="db6")
            self.wavelet_type.addItem(key="db7")
            self.wavelet_type.addItem(key="db35")
            self.wavelet_type.addItem(key="coif1")
            self.wavelet_type.addItem(key="coif3")
            self.wavelet_type.addItem(key="coif7")
            self.wavelet_type.addItem(key="bior1.1")
            self.wavelet_type.addItem(key="bior2.2")
            self.wavelet_type.addItem(key="bior3.5")

            widget = HWidgets(
                "Wavelet type:", self.wavelet_type,  stretch=0
            )
            self.add_row(widget)

            self.wavelet_threshold = RealSlider(value=0.0, vmax=128, vmin=0, n=2000)
            widget = HWidgets(
                "Threshold:", self.wavelet_threshold, stretch=0, 
            )
            self.add_row(widget)
            self._add_btns()
        elif self.feature_type=="feature_composite":
            self._add_feature_source()
            self._add_feature_source2()
            self.label_index = LineEdit(default=-1, parse=int)
            self.op_type = ComboBox()
            self.op_type.addItem(key="*")
            self.op_type.addItem(key="+")
            widget = HWidgets("Operation:", self.op_type, stretch=0)
            self.add_row(widget)
            self._add_btns()
        elif self.feature_type=="raw":
            self._add_view_and_load_btns()

        else:
            self._add_source()
            self._add_btns()

        for pname, params in fparams.items():
            if pname not in ["src", "dst", "threshold"]:
                self._add_param(pname, **params)

        
        


    def _add_source(self):
        chk_clamp = CheckBox("Clamp")
        self.cmb_source = SourceComboBox([self.feature_id,'001 Raw'])
        self.cmb_source.fill()
        widget = HWidgets(self.cmb_source,  stretch=1)
        self.add_row(widget)

    def _add_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source,  stretch=1)
        self.add_row(widget)

    def _add_feature_source2(self):
        self.feature_source2 = FeatureComboBox()
        self.feature_source2.fill()
        self.feature_source2.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source2,  stretch=1)
        self.add_row(widget)
    
    def _add_param(self, name, type="String", default=None):
        if type == "Int":
            feature = LineEdit(default=default, parse=int)
        elif type == "Float":
            feature = LineEdit(default=default, parse=float)
        elif type == "FloatOrVector":
            feature = LineEdit3D(default=default, parse=float)
        elif type == "IntOrVector":
            feature = LineEdit3D(default=default, parse=int)
        elif type == "SmartBoolean":
            feature = CheckBox(checked=True)
        else:
            feature = None

        if feature:
            self.widgets[name] = feature
            self.add_row(HWidgets(None, name, feature))

    def _add_btns(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_feature)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)


        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self.compute_feature)
        self.add_row(
            HWidgets(
                None,
                load_as_annotation_btn, 
                compute_btn,
                view_btn
            )
        )

    def _add_view_and_load_btns(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_feature)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)

        self.add_row(
            HWidgets(
                None,
                load_as_annotation_btn, 
                view_btn
            )
        )



    def update_params(self, params):
        src = params.pop("source", None)
        print(params)
        if src is not None:
            if self.feature_type != "feature_composite" and self.feature_type != "raw":
                self.cmb_source.select(src)
        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)
        if "threshold" in params:
            if params["threshold"] is not None:
                self.wavelet_threshold.setValue(float(params["threshold"]))

    def card_deleted(self):
        params = dict(feature_id=self.feature_id, workspace=True)
        result = Launcher.g.run("features", "remove", **params)
        if result["done"]:
            self.setParent(None)
            _FeatureNotifier.notify()

        cfg.ppw.clientEvent.emit(
            {
                "source": "features",
                "data": "remove_layer",
                "layer_name": self.feature_id,
            }
        )

    def view_feature(self):
        logger.debug(f"View feature_id {self.feature_id}")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing feature")
            pbar.update(1)
            cfg.ppw.clientEvent.emit(
                {
                    "source": "features",
                    "data": "view_feature",
                    "feature_id": self.feature_id,
                }
            )
            pbar.update(1)    
        

    def load_as_annotation(self):
        logger.debug(f"Loading feature {self.feature_id} as annotation.")



        # get feature output
        src = DataModel.g.dataset_uri(self.feature_id, group="features")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]

        src_arr = (src_arr > 0) * 1
        label_values = np.unique(src_arr)

        # create new level
        params = dict(level=self.feature_id, workspace=True)
        result = Launcher.g.run("annotations", "add_level", workspace=True)

        # create a blank label for each unique value in the feature output array
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
                level=str('001_level'),
                workspace=True,
            )
            anno_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            params = dict(level=str(level_id), workspace=True)
            level_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            try:
                # set the new level color mapping to the mapping from the feature
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

            # set levels array to feature output array
            dst = DataModel.g.dataset_uri(fid, group="annotations")
            with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                DM.out[:] = src_arr

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "faster_refresh_plugin", "plugin_name": "annotations"}
            )

    def compute_feature(self):
        with progress(total=3) as pbar:
            pbar.set_description("Calculating feature")
            # self.pbar.setValue(25)
            pbar.update(1)

            if self.feature_type != "feature_composite":
                src_grp = None if self.cmb_source.currentIndex() == 0 else "features"
                src = DataModel.g.dataset_uri(self.cmb_source.value(), group=src_grp)
            else:
                src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")

            dst = DataModel.g.dataset_uri(self.feature_id, group="features")
            

            logger.info(f"Setting dst: {self.feature_id}")
            logger.info(f"widgets.items() {self.widgets.items()}")
            pbar.update(1)

            all_params = dict(src=src, dst=dst, modal=True)

            if self.feature_type == "wavelet":
                all_params["wavelet"] = str(self.wavelet_type.value())
                all_params["threshold"] = self.wavelet_threshold.value()

            elif self.feature_type == "feature_composite":
                all_params["workspace"] = DataModel.g.current_workspace
                all_params["feature_A"] = str(self.feature_source.value())
                all_params["feature_B"] = str(self.feature_source2.value())
                all_params["op"] = str(self.op_type.value())


            all_params.update({k: v.value() for k, v in self.widgets.items()})

            logger.info(f"Computing features: {self.feature_type} {all_params}")
                        
            Launcher.g.run("features", self.feature_type, **all_params)
            pbar.update(1)
            
                
            
    def card_title_edited(self, newtitle):
        params = dict(feature_id=self.feature_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("features", "rename", **params)

        if result["done"]:
            _FeatureNotifier.notify()

        return result["done"]




