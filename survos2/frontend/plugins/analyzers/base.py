import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.qt.progress import progress
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton
from survos2.entity.entities import make_entity_df

from survos2.frontend.components.entity import TableWidget
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.components.base import (
    VBox,
    LazyComboBox,
    HWidgets,
    PushButton,
    SimpleComboBox,
    Card,
    LineEdit,
    MultiComboBox,
)
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.pipelines import PipelinesComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.utils import FileWidget
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.plugins.analyzers.constants import feature_names


class AnalyzersComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run("analyzer", "existing", **params)
        logger.debug(f"Result of analyzer existing: {result}")
        if result:
            self.addCategory("analyzer")
            for fid in result:
                self.addItem(fid, result[fid]["name"])


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, suptitle="Feature"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.suptitle = suptitle
        super(MplCanvas, self).__init__(self.fig)

    def set_suptitle(self, suptitle):
        self.suptitle = suptitle
        self.fig.suptitle(suptitle)


class AnalyzerCardBase(Card):
    def __init__(
        self,
        analyzer_id,
        analyzer_name,
        analyzer_type,
        title=None,
        collapsible=True,
        removable=True,
        editable=True,
        parent=None,
    ):
        super().__init__(
            title=title,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.analyzer_id = analyzer_id
        self.analyzer_name = analyzer_name
        self.analyzer_type = analyzer_type
        self.annotations_source = "001_level"  # default annotation level to use for labels
        self.additional_buttons = []
        self.annotations_selected = False
        self.op_cards = []
        self.model_fullname = "None"
        self.setup()

        self.calc_btn = PushButton("Compute")

        if len(self.additional_buttons) > 0:
            self.add_row(HWidgets(None, *self.additional_buttons, self.calc_btn))
        else:
            self.add_row(HWidgets(None, self.calc_btn))

        self.calc_btn.clicked.connect(self.calculate)
        self.table_control = None
        self.plots = []
        self.object_analyzer_plots = []
        self.object_analyzer_controls = []

    def _calculate(self):
        pass

    def setup(self):
        pass

    def calculate(self):
        with progress(total=2) as pbar:
            pbar.set_description("Calculating pipeline")
            pbar.update(1)

            self._calculate()
            pbar.update(2)

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)
        widget = HWidgets("Objects:", self.objects_source, stretch=1)
        self.add_row(widget)

    def _add_objects_source2(self, title="Objects"):
        self.objects_source2 = ObjectComboBox(full=True)
        self.objects_source2.fill()
        self.objects_source2.setMaximumWidth(250)
        widget = HWidgets(title, self.objects_source2, stretch=1)
        self.add_row(widget)

    def _add_object_detection_stats_source(self):
        self.gold_objects_source = ObjectComboBox(full=True)
        self.gold_objects_source.fill()
        self.gold_objects_source.setMaximumWidth(250)
        widget = HWidgets("Gold Objects:", self.gold_objects_source, stretch=1)
        self.add_row(widget)

        self.predicted_objects_source = ObjectComboBox(full=True)
        self.predicted_objects_source.fill()
        self.predicted_objects_source.setMaximumWidth(250)
        widget = HWidgets("Predicted Objects:", self.predicted_objects_source, stretch=1)
        self.add_row(widget)

    def _add_annotations_source(self, label="Annotation"):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source, stretch=1)

        self.add_row(widget)

    def _add_pipelines_source(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets("Segmentation:", self.pipelines_source, stretch=1)
        self.add_row(widget)

    def _add_pipelines_source2(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets("Segmentation:", self.pipelines_source, stretch=1)
        self.add_row(widget)
        load_as_objects = PushButton("Load as Objects")
        return load_as_objects

    def _add_analyzers_source(self):
        self.analyzers_source = AnalyzersComboBox()
        self.analyzers_source.fill()
        self.analyzers_source.setMaximumWidth(250)
        widget = HWidgets("Analyzers:", self.analyzers_source, stretch=1)
        self.add_row(widget)

    def _add_feature_source(self, label="Feature:"):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source, stretch=1)
        self.add_row(widget)

    def _add_feature_source3(self, label="Feature:"):
        self.feature_source3 = FeatureComboBox()
        self.feature_source3.fill()
        self.feature_source3.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source3, stretch=1)
        self.add_row(widget)

    def _add_feature_source2(self, label="Feature:"):
        self.feature_source2 = FeatureComboBox()
        self.feature_source2.fill()
        self.feature_source2.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source2, stretch=1)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source, stretch=1)
        self.add_row(widget)

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_analyzer)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as feature", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)
        self.add_row(
            HWidgets(
                None,
                load_as_float_btn,
                load_as_annotation_btn,
                view_btn,
            )
        )

    def _add_labelsplitter_view_btns(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_analyzer)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as feature", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)

        self.export_csv_btn = PushButton("Export CSV")
        self.load_as_objects_btn = PushButton("Load as Objects")
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        self.export_csv_btn.clicked.connect(self.export_csv)

        self.add_row(
            HWidgets(
                None,
                load_as_annotation_btn,
                load_as_float_btn,
                self.load_as_objects_btn,
                self.export_csv_btn,
                view_btn,
            )
        )

    def _add_view_entities(self):
        view_btn = PushButton("View Entities", accent=True)
        view_btn.clicked.connect(self.view_entities)
        self.add_row(
            HWidgets(
                None,
                view_btn,
            )
        )

    def load_data(self, path):
        self.model_fullname = path
        print(f"Setting model fullname: {self.model_fullname}")

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def view_analyzer(self):
        logger.debug(f"View analyzer_id {self.analyzer_id}")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing analyzer")
            pbar.update(1)

            if self.annotations_source:
                if self.annotations_selected:
                    level_id = self.annotations_source.value().rsplit("/", 1)[-1]
                else:
                    level_id = "001_level"
                logger.debug(f"Assigning annotation level {level_id}")

                cfg.ppw.clientEvent.emit(
                    {
                        "source": "analyzer",
                        "data": "view_pipeline",
                        "pipeline_id": self.analyzer_id,
                        "level_id": level_id,
                    }
                )

            pbar.update(1)

    def load_as_float(self):
        logger.debug(f"Loading analyzer result {self.analyzer_id} as float image.")

        # get analyzer output
        src = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
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
        logger.debug(f"Loading analyzer result {self.analyzer_id} as annotation.")

        # get analyzer output
        src = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
        label_values = np.unique(src_arr)

        # create new level
        params = dict(level=self.analyzer_id, workspace=True)
        result = Launcher.g.run("annotations", "add_level", workspace=True)

        # create a blank label for each unique value in the analyzer output array
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

            # derive label colours from given annotation
            params = dict(
                level=str("001_level"),
                workspace=True,
            )
            anno_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            params = dict(level=str(level_id), workspace=True)
            level_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            try:
                # set the new level color mapping to the mapping from the given annotation
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

    def card_deleted(self):
        params = dict(analyzer_id=self.analyzer_id, workspace=True)
        result = Launcher.g.run("analyzer", "remove", **params)
        if result["done"]:
            self.setParent(None)

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited analyzer title {newtitle}")
        params = dict(analyzer_id=self.analyzer_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("analyzer", "rename", **params)
        return result["done"]

    def update_params(self, params):
        logger.debug(f"Analyzer update params {params}")
        # for k, v in params.items():
        #     if k in self.widgets:
        #         self.widgets[k].setValue(v)

        if "anno_id" in params:
            if params["anno_id"] is not None:
                if isinstance(params["anno_id"], list):
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"][0])
                    )
                else:
                    self.annotations_source.select(os.path.join("annotations/", params["anno_id"]))

        if "object_id" in params:
            if params["object_id"] is not None:
                self.objects_source.select(os.path.join("objects/", params["object_id"]))
        if "feature_id" in params:
            self.feature_source.select(params["feature_id"])
        if "feature_ids" in params:
            for source in params["feature_ids"]:
                self.features_source.select(os.path.join("features/", source))
        if "region_id" in params:
            if params["region_id"] is not None:
                self.regions_source.select(os.path.join("regions/", params["region_id"]))

    def display_splitter_results(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
                result[i][4],
                result[i][5],
                result[i][6],
                result[i][7],
                result[i][8],
                result[i][9],
                result[i][10],
                result[i][11],
                result[i][12],
                result[i][13],
                result[i][14],
                result[i][15],
                result[i][16],
                result[i][17],
                result[i][18],
                result[i][19],
            )
            tabledata.append(entry)

            entity = (result[i][0], result[i][2], result[i][1], 0)
            entities.append(entity)

        tabledata = np.array(
            tabledata,
            dtype=[
                ("index", int),
                ("z", int),
                ("x", int),
                ("y", int),
                ("Sum", float),
                ("Mean", float),
                ("Std", float),
                ("Var", float),
                ("BB Vol", float),
                ("BB Vol Log10", float),
                ("BB Depth", float),
                ("BB Height", float),
                ("BB Width", float),
                ("OrientBB Vol", float),
                ("OrientBB Vol Log10", float),
                ("OrientBB Depth", float),
                ("OrientBB Height", float),
                ("OrientBB Width", float),
                ("Seg Surface Area", float),
                ("Seg Volume", float),
                ("Seg Sphericity", float),
            ],
        )

        if self.table_control is None:
            self.table_control = TableWidget()
            max_height = 500
            self.table_control.w.setProperty("header", False)
            self.table_control.w.setMaximumHeight(max_height)
            self.vbox.addWidget(self.table_control.w)
            self.total_height += 500 + self.spacing
            self.setMinimumHeight(self.total_height)

        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()
        self.tabledata = tabledata
        self.entities_arr = np.array(entities)

    def display_component_results(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
            )
            tabledata.append(entry)

            entity = (result[i][0], result[i][1], result[i][2], 0)
            entities.append(entity)

        tabledata = np.array(
            tabledata,
            dtype=[
                ("index", int),
                ("z", int),
                ("x", int),
                ("y", int),
                ("area", int),
            ],
        )

        if self.table_control is None:
            self.table_control = TableWidget()
            self.add_row(self.table_control.w, max_height=500)

        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()

        self.entities_arr = np.array(entities)

    def display_component_results2(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
            )
            tabledata.append(entry)

            entity = (result[i][1], result[i][2], result[i][3], 0)
            entities.append(entity)

        self.entities_arr = np.array(entities)

    def display_component_results3(self, result):

        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
            )
            tabledata.append(entry)
            entity = (result[i][0], result[i][1], result[i][2], 0)
            entities.append(entity)

        self.entities_arr = np.array(entities)

    def display_splitter_plot(self, feature_arrays, titles=[], vert_line_at=None):
        for i, feature_array in enumerate(feature_arrays):
            self.plots.append(MplCanvas(self, width=5, height=5, dpi=100))
            max_height = 600
            self.plots[i].setProperty("header", False)
            self.plots[i].setMaximumHeight(max_height)
            self.vbox.addWidget(self.plots[i])
            self.total_height += 500 + self.spacing
            self.setMinimumHeight(self.total_height)

            colors = ["r", "y", "b", "c", "m", "g"]
            y, x, _ = self.plots[i].axes.hist(feature_array, bins=16, color=colors[i])
            self.plots[i].axes.set_title(titles[i])

            if vert_line_at:
                print(f"Plotting vertical line at: {vert_line_at[i]} {y.max()}")
                self.plots[i].axes.axvline(x=vert_line_at[i], ymin=0, ymax=y.max(), color="k")

    def display_clustering_results(self, labels):
        for ctl in self.object_analyzer_controls:
            ctl.setParent(None)
            ctl = None
        self.labels_combo = MultiComboBox()
        self.vbox.addWidget(self.labels_combo)
        self.object_analyzer_controls.append(self.labels_combo)
        for el in np.unique(labels):
            self.labels_combo.addItem(str(el), value=str(el))
        self.load_cluster_as_objects_btn = PushButton("Load Cluster as Objects")
        self.vbox.addWidget(self.load_cluster_as_objects_btn)
        self.object_analyzer_controls.append(self.load_cluster_as_objects_btn)
        self.load_cluster_as_objects_btn.clicked.connect(self.load_cluster_as_objects)

    def view_entities(self):
        logger.debug(f"Transferring entities to viewer")
        cfg.ppw.clientEvent.emit(
            {
                "source": "analyzer",
                "data": "view_objects",
                "entities": self.entities_arr,
                "flipxy": self.flipxy_checkbox.value(),
            }
        )

    def load_cluster_as_objects(self):
        logger.debug("Load cluster as objects")
        from survos2.entity.entities import load_entities_via_file

        labels = self.labels_combo.value()
        labels = [int(l) for l in labels]
        print(f"Selected cluster: {labels}")

        selected = []
        for l in labels:
            selected.append(np.where(self.entities_arr[:, 3] == l)[0])
        selected = np.concatenate(selected)
        selected_b = np.zeros_like(self.entities_arr[:, 3])
        for l in selected:
            selected_b[l] = 1
        selected_b = selected_b > 0
        self.entities_arr = self.entities_arr[selected_b]
        load_entities_via_file(self.entities_arr, flipxy=False)
        cfg.ppw.clientEvent.emit(
            {"source": "analyzer_plugin", "data": "refresh_plugin", "plugin_name": "objects"}
        )

    def export_csv(self):
        full_path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output filename", ".", filter="*.csv"
        )
        if isinstance(full_path, tuple):
            full_path = full_path[0]

        out_df = pd.DataFrame(self.tabledata)
        out_df.to_csv(full_path)
        logger.debug(f"Exported to csv {full_path}")

    def add_source_selector(self):
        # radio buttons to select source type
        self.radio_group = QtWidgets.QButtonGroup()
        self.radio_group.setExclusive(True)
        pipelines_rb = QRadioButton("Pipelines")
        pipelines_rb.setChecked(True)
        pipelines_rb.toggled.connect(self._pipelines_rb_checked)
        self.radio_group.addButton(pipelines_rb, 1)
        analyzers_rb = QRadioButton("Analyzers")
        analyzers_rb.toggled.connect(self._analyzers_rb_checked)
        self.radio_group.addButton(analyzers_rb, 2)
        annotations_rb = QRadioButton("Annotation")
        self.radio_group.addButton(annotations_rb, 3)
        annotations_rb.toggled.connect(self._annotations_rb_checked)
        self.add_row(HWidgets(pipelines_rb, analyzers_rb, annotations_rb))

        self.source_container = QtWidgets.QWidget()
        source_vbox = VBox(self, spacing=4)
        source_vbox.setContentsMargins(0, 0, 0, 0)
        self.source_container.setLayout(source_vbox)
        self.add_row(self.source_container)
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        self.pipelines_widget = HWidgets("Segmentation:", self.pipelines_source, stretch=1)
        self.pipelines_widget.setParent(None)

        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)
        self.annotations_widget = HWidgets("Annotation", self.annotations_source, stretch=1)
        self.annotations_widget.setParent(None)
        self.analyzers_source = AnalyzersComboBox()
        self.analyzers_source.fill()
        self.analyzers_source.setMaximumWidth(250)
        self.analyzers_widget = HWidgets("Analyzers:", self.analyzers_source, stretch=1)
        self.analyzers_widget.setParent(None)
        self.source_container.layout().addWidget(self.pipelines_widget)
        self.current_widget = self.pipelines_widget

    def add_source_selector2(self):
        # radio buttons to select source type
        self.radio_group2 = QtWidgets.QButtonGroup()
        self.radio_group2.setExclusive(True)
        pipelines_rb2 = QRadioButton("Pipelines")
        pipelines_rb2.setChecked(True)
        pipelines_rb2.toggled.connect(self._pipelines_rb2_checked)
        self.radio_group2.addButton(pipelines_rb2, 1)
        analyzers_rb2 = QRadioButton("Analyzers")
        analyzers_rb2.toggled.connect(self._analyzers_rb2_checked)
        self.radio_group2.addButton(analyzers_rb2, 2)
        annotations_rb2 = QRadioButton("Annotation")
        self.radio_group2.addButton(annotations_rb2, 3)
        annotations_rb2.toggled.connect(self._annotations_rb2_checked)
        self.add_row(HWidgets(pipelines_rb2, analyzers_rb2, annotations_rb2))

        self.source_container2 = QtWidgets.QWidget()
        source_vbox = VBox(self, spacing=4)
        source_vbox.setContentsMargins(0, 0, 0, 0)
        self.source_container2.setLayout(source_vbox)
        self.add_row(self.source_container2)
        self.pipelines_source2 = PipelinesComboBox()
        self.pipelines_source2.fill()
        self.pipelines_source2.setMaximumWidth(250)
        self.pipelines_widget2 = HWidgets("Segmentation:", self.pipelines_source2, stretch=1)
        self.pipelines_widget2.setParent(None)

        self.annotations_source2 = LevelComboBox(full=True)
        self.annotations_source2.fill()
        self.annotations_source2.setMaximumWidth(250)
        self.annotations_widget2 = HWidgets("Annotation", self.annotations_source2, stretch=1)
        self.annotations_widget2.setParent(None)
        self.analyzers_source2 = AnalyzersComboBox()
        self.analyzers_source2.fill()
        self.analyzers_source2.setMaximumWidth(250)
        self.analyzers_widget2 = HWidgets("Analyzers:", self.analyzers_source2, stretch=1)
        self.analyzers_widget2.setParent(None)
        self.source_container2.layout().addWidget(self.pipelines_widget2)
        self.current_widget2 = self.pipelines_widget2

    def _pipelines_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.pipelines_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.pipelines_widget
            self.annotations_selected = False

    def _analyzers_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.analyzers_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.analyzers_widget
            self.annotations_selected = False

    def _annotations_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.annotations_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.annotations_widget
            self.annotations_selected = True

    def _pipelines_rb2_checked(self, enabled):
        if enabled:
            self.source_container2.layout().addWidget(self.pipelines_widget2)
            if self.current_widget2:
                self.current_widget2.setParent(None)
            self.current_widget2 = self.pipelines_widget2

    def _analyzers_rb2_checked(self, enabled):
        if enabled:
            self.source_container2.layout().addWidget(self.analyzers_widget2)
            if self.current_widget2:
                self.current_widget2.setParent(None)
            self.current_widget2 = self.analyzers_widget2

    def _annotations_rb2_checked(self, enabled):
        if enabled:
            self.source_container2.layout().addWidget(self.annotations_widget2)
            if self.current_widget2:
                self.current_widget2.setParent(None)
            self.current_widget2 = self.annotations_widget2

    def _setup_ops(self):
        print(f"Current number of op cards {len(self.op_cards)}")
        if self.table_control:
            self.table_control.w.setParent(None)
            self.table_control = None
        if len(self.plots) > 0:
            for plot in self.plots:
                plot.setParent(None)
                plot = None
            self.plots = []

    def _setup_object_analyzer_plots(self):
        if len(self.object_analyzer_plots) > 0:
            for plot in self.object_analyzer_plots:
                plot.setParent(None)
                plot = None
            self.object_analyzer_plots = []

    def _add_rule(self):
        op_card = RuleCard(
            title="Rule", editable=True, collapsible=False, removable=True, parent=self
        )
        self.op_cards.append(op_card)
        self.add_row(op_card)
        self.add_to_widget_list(op_card)

    def load_as_objects(self):
        logger.debug(f"Load analyzer result as objects {self.entities_arr}")
        # from survos2.entity.entities import load_entities_via_file
        # load_entities_via_file(self.entities_arr, flipxy=True)
        entities = np.array(make_entity_df(self.entities_arr, flipxy=True))
        result = Launcher.g.post_array(
            entities, group="objects", workspace=DataModel.g.current_workspace, name="objects"
        )
        cfg.ppw.clientEvent.emit(
            {"source": "analyzer_plugin", "data": "refresh_plugin", "plugin_name": "objects"}
        )


class RuleCard(Card):
    def __init__(self, title, collapsible=True, removable=True, editable=True, parent=None):
        super().__init__(
            title=title,
            collapsible=collapsible,
            removable=removable,
            editable=editable,
            parent=parent,
        )
        self.title = title
        self.parent = parent
        self.feature_name_combo_box = SimpleComboBox(full=True, values=feature_names)
        self.feature_name_combo_box.fill()
        self.split_op_combo_box = SimpleComboBox(full=True, values=["None", ">", "<"])
        self.split_op_combo_box.fill()
        self.split_threshold = LineEdit(default=0, parse=float)
        measure_widget = HWidgets("Measurement:", self.feature_name_combo_box, stretch=1)
        splitop_widget = HWidgets(
            "Split operation:",
            self.split_op_combo_box,
            self.split_threshold,
            stretch=1,
        )
        self.vbox.addWidget(measure_widget)
        self.add_to_widget_list(measure_widget)
        self.vbox.addWidget(splitop_widget)
        self.setMinimumHeight(260)
        self.add_to_widget_list(splitop_widget)

    def card_deleted(self):
        logger.debug(f"Deleted Rule {self.title}")
        self.parent.op_cards.remove(self)
        self.setParent(None)
