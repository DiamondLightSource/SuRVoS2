from survos2.frontend.plugins.pipelines import PipelinesComboBox
from survos2.frontend.plugins.features import FeatureComboBox
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton

from survos2.frontend.components.base import *
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.plugins.export import SuperRegionSegmentComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.frontend.components.entity import TableWidget
from survos2.frontend.utils import FileWidget


def _fill_analyzers(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    result = Launcher.g.run("analyzer", "existing", **params)

    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


class SimpleComboBox(QtWidgets.QComboBox):
    def __init__(self, full=False, parent=None, values=["one", "two"]):
        self.full = full
        self.values = values
        self._items = []
        super().__init__(parent=parent)
        self.setStyleSheet(
            """
            QComboBox::item:disabled, QMenu[combo=true]::item:disabled {
                background-color: #00838F;
                color: #AACAFF;
            }

            QComboBox::item:selected, QMenu[combo=true]::item:selected
            {
                background-color: #B2EBF2;
            }

        """
        )

    def key(self):
        return self.value(key=True)

    def keys(self):
        return [self.key()]

    def items(self):
        return (item for item in self._items)

    def removeItem(self, idx):
        self._items.pop(idx)
        super().removeItem(idx)

    def value(self, key=False):
        if self.currentIndex() < 0:
            return None
        item = self._items[self.currentIndex()]
        return item[0] if key or item[2] is None else item[2]

    def fill(self):
        for i, feature in enumerate(self.values):
            self.addItem(str(i), feature)

    def addItem(self, key, value=None, icon=None, data=None):
        self._items.append((key, value, data))
        if icon:
            super().addItem(icon, value if value else key)
        else:
            super().addItem(value if value else key)


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
                if result[fid]["kind"] == "analyzer":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class AnalyzerPlugin(Plugin):

    __icon__ = "fa.picture-o"
    __pname__ = "analyzer"
    __views__ = ["slice_viewer"]
    __tab__ = "analyzer"

    def __init__(self, parent=None):
        self.analyzers_combo = ComboBox()
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)
        vbox.addWidget(self.analyzers_combo)
        self.analyzers_combo.currentIndexChanged.connect(self.add_analyzer)
        self.existing_analyzer = {}
        self.analyzers_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.analyzers_layout)
        self._populate_analyzers()

    def _populate_analyzers(self):
        self.analyzer_params = {}
        self.analyzers_combo.clear()
        self.analyzers_combo.addItem("Add analyzer")

        result = None
        result = Launcher.g.run("analyzer", "available", workspace=True)

        if not result:
            logger.debug("No analyzers.")
        else:

            all_categories = sorted(set(p["category"] for p in result))

            logger.debug(f"Analyzer {result}")
            for i, category in enumerate(all_categories):
                self.analyzers_combo.addItem(category)
                self.analyzers_combo.model().item(
                    i + len(self.analyzer_params) + 1
                ).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.analyzer_params[f["name"]] = f["params"]
                    self.analyzers_combo.addItem(f["name"])

    def add_analyzer(self, idx):
        if idx != 0:

            analyzer_type = self.analyzers_combo.itemText(idx)
            if analyzer_type == "":
                return
            self.analyzers_combo.setCurrentIndex(0)
            logger.debug(f"Adding analyzer {analyzer_type}")
            from survos2.api.analyzer import __analyzer_names__

            order = __analyzer_names__.index(analyzer_type)
            params = dict(analyzer_type=analyzer_type, order=order, workspace=True)
            result = Launcher.g.run("analyzer", "create", **params)

            if result:
                analyzer_id = result["id"]
                analyzer_name = result["name"]
                analyzer_type = result["kind"]
                self._add_analyzer_widget(
                    analyzer_id, analyzer_name, analyzer_type, True
                )

    def _add_analyzer_widget(
        self, analyzer_id, analyzer_name, analyzer_type, expand=False
    ):
        widget = AnalyzerCard(analyzer_id, analyzer_name, analyzer_type)
        widget.showContent(expand)
        self.analyzers_layout.addWidget(widget)

        self.existing_analyzer[analyzer_id] = widget
        return widget

    def clear(self):
        for analyzer in list(self.existing_analyzer.keys()):
            self.existing_analyzer.pop(analyzer).setParent(None)
        self.existing_analyzer = {}

    def setup(self):
        self._populate_analyzers()

        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )

        result = Launcher.g.run("analyzer", "existing", **params)

        if result:
            logger.debug(f"Populating analyzer widgets with {result}")
            # Remove analyzer that no longer exist in the server
            for analyzer in list(self.existing_analyzer.keys()):
                if analyzer not in result:
                    self.existing_analyzer.pop(analyzer).setParent(None)
            # Populate with new analyzer if any
            for analyzer in sorted(result):
                if analyzer in self.existing_analyzer:
                    continue
                params = result[analyzer]
                analyzer_id = params.pop("id", analyzer)
                analyzer_name = params.pop("name", analyzer)
                analyzer_type = params.pop("kind", analyzer)
                logger.debug(f"Adding analyzer of type {analyzer_type}")

                if params.pop("kind", analyzer) != "unknown":
                    widget = self._add_analyzer_widget(
                        analyzer_id, analyzer_name, analyzer_type
                    )
                    widget.update_params(params)
                    self.existing_analyzer[analyzer_id] = widget
                else:
                    logger.debug(
                        "+ Skipping loading analyzer: {}, {}".format(
                            analyzer_id, analyzer_name, analyzer_type
                        )
                    )


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class AnalyzerCard(Card):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.analyzer_id = analyzer_id
        self.analyzer_name = analyzer_name
        self.analyzer_type = analyzer_type
        self.annotations_source = (
            "001_level"  # default annotation level to use for labels
        )
        additional_buttons = []

        if self.analyzer_type == "label_splitter":
            self._add_pipelines_source()
            self._add_feature_source()
            self.feature_name_combo_box = SimpleComboBox(
                full=True, values=["z", "x", "y", "Sum", "Mean", "Std", "Var", "bb_vol"]
            )
            self.feature_name_combo_box.fill()
            self.split_op_combo_box = SimpleComboBox(
                full=True, values=["None", ">", "<"]
            )
            self.split_op_combo_box.fill()

            self.split_threshold = LineEdit(default=0, parse=int)
            widget = HWidgets(
                "Measurement:", self.feature_name_combo_box, Spacing(35), stretch=1
            )
            self.add_row(widget)
            widget = HWidgets(
                "Split operation:",
                self.split_op_combo_box,
                self.split_threshold,
                Spacing(35),
                stretch=1,
            )
            self.add_row(widget)
            self.export_csv_btn = PushButton("Export CSV")
            self.add_row(HWidgets(None, self.export_csv_btn, Spacing(35)))
            self.export_csv_btn.clicked.connect(self.export_csv)
            self._add_view_btn()
        elif self.analyzer_type == "image_stats":
            self._add_features_source()
            self.plot_btn = PushButton("Plot")
            additional_buttons.append(self.plot_btn)
            self.plot_btn.clicked.connect(self.clustering_plot)
        elif self.analyzer_type == "level_image_stats":
            self._add_annotations_source()
            self.label_index = LineEdit(default=1, parse=float)
            widget = HWidgets("Level index:", self.label_index, Spacing(35), stretch=1)
            self.add_row(widget)
        elif self.analyzer_type == "binary_image_stats":
            self._add_feature_source()
            self.threshold = LineEdit(default=0.5, parse=float)
            widget = HWidgets("Threshold:", self.threshold, Spacing(35), stretch=1)
            self.add_row(widget)
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "object_stats":
            self._add_features_source()
            self._add_objects_source()
            self.stat_name_combo_box = SimpleComboBox(
                full=True, values=["Mean", "Std", "Var"]
            )
            self.stat_name_combo_box.fill()
            widget = HWidgets(
                "Statistic name:", self.stat_name_combo_box, Spacing(35), stretch=1
            )
            self.add_row(widget)
        elif self.analyzer_type == "object_detection_stats":
            self._add_features_source()
            self._add_object_detection_stats_source()
        elif self.analyzer_type == "find_connected_components":
            additional_buttons.append(self._add_pipelines_source())
            self.label_index = LineEdit(default=0, parse=int)
            widget = HWidgets("Label Index:", self.label_index, Spacing(35), stretch=1)
            self.add_row(widget)
            self.load_as_objects_btn = additional_buttons[-1]
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
            self._add_view_btn()
        elif self.analyzer_type == "detector_predict":
            self._add_features_source()
            self._add_objects_source()
            self._add_model_file()
        elif self.analyzer_type == "remove_masked_objects":
            self._add_feature_source()
            self._add_objects_source()
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "spatial_clustering":
            self._add_feature_source()
            self._add_objects_source()
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)

        self.calc_btn = PushButton("Compute")
        self.add_row(HWidgets(None, self.calc_btn, Spacing(35)))
        if len(additional_buttons) > 0:
            self.add_row(HWidgets(None, *additional_buttons, Spacing(35)))
        self.calc_btn.clicked.connect(self.calculate_analyzer)
        self.table_control = None

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def load_data(self, path):
        self.model_fullname = path
        print(f"Setting model fullname: {self.model_fullname}")

    def load_as_objects(self):
        logger.debug("Load analyzer result as objects")
        from survos2.entity.entities import load_entities_as_objects

        load_entities_as_objects(self.entities_arr, flipxy=False)
        cfg.ppw.clientEvent.emit(
            {"source": "analyzer_plugin", "data": "refresh", "value": None}
        )

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)
        widget = HWidgets("Objects:", self.objects_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_object_detection_stats_source(self):
        self.gold_objects_source = ObjectComboBox(full=True)
        self.gold_objects_source.fill()
        self.gold_objects_source.setMaximumWidth(250)
        widget = HWidgets(
            "Gold Objects:", self.gold_objects_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

        self.predicted_objects_source = ObjectComboBox(full=True)
        self.predicted_objects_source.fill()
        self.predicted_objects_source.setMaximumWidth(250)
        widget = HWidgets(
            "Predicted Objects:", self.predicted_objects_source, Spacing(35), stretch=1
        )
        self.add_row(widget)

    def _add_annotations_source(self, label="Annotation"):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source, Spacing(35), stretch=1)

        self.add_row(widget)

    def _add_pipelines_source(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets(
            "Segmentation:", self.pipelines_source, Spacing(35), stretch=1
        )
        self.add_row(widget)
        load_as_objects = PushButton("Load as Objects")
        return load_as_objects

    def _add_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Feature:", self.feature_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_analyzer)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as image", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)
        self.add_row(
            HWidgets(
                None, load_as_float_btn, load_as_annotation_btn, view_btn, Spacing(35)
            )
        )

    def view_analyzer(self):
        logger.debug(f"View analyzer_id {self.analyzer_id}")
        if self.annotations_source:
            level_id = self.annotations_source
            logger.debug(f"Assigning annotation level {level_id}")

            cfg.ppw.clientEvent.emit(
                {
                    "source": "analyzer",
                    "data": "view_pipeline",
                    "pipeline_id": self.analyzer_id,
                    "level_id": level_id,
                }
            )

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
                {"source": "workspace_gui", "data": "refresh", "value": None}
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
                {"source": "workspace_gui", "data": "refresh", "value": None}
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
        pass

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
                ("Sum", float),
                ("Mean", float),
                ("Std", float),
                ("Var", float),
                ("BB Vol", float),
            ],
        )

        if self.table_control is None:
            self.table_control = TableWidget()
            self.add_row(self.table_control.w, max_height=500)

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

            entity = (result[i][1], result[i][2], result[i][3], 0)
            entities.append(entity)

        tabledata = np.array(
            tabledata,
            dtype=[
                ("index", int),
                ("area", int),
                ("z", int),
                ("x", int),
                ("y", int),
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

    def calculate_analyzer(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")

        if self.analyzer_type == "label_splitter":
            src = DataModel.g.dataset_uri(self.pipelines_source.value())

            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["pipelines_id"] = str(self.pipelines_source.value())
            all_params["feature_id"] = str(self.feature_source.value())
            all_params["split_feature_name"] = self.feature_name_combo_box.value()
            all_params["split_op"] = self.split_op_combo_box.value()
            all_params["split_threshold"] = self.split_threshold.value()

            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "label_splitter", **all_params)
            if result:
                logger.debug(f"Segmentation stats result table {len(result)}")
                self.display_splitter_results(result)

        if self.analyzer_type == "find_connected_components":
            src = DataModel.g.dataset_uri(self.pipelines_source.value())

            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["pipelines_id"] = str(self.pipelines_source.value())
            all_params["label_index"] = self.label_index.value()
            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run(
                "analyzer", "find_connected_components", **all_params
            )
            if result:
                logger.debug(f"Segmentation stats result table {len(result)}")
                self.display_component_results(result)

        if self.analyzer_type == "level_image_stats":
            anno_id = DataModel.g.dataset_uri(self.annotations_source.value())
            all_params = dict(dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["anno_id"] = anno_id
            all_params["label_index"] = self.label_index.value()
            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "level_image_stats", **all_params)
            if result:
                logger.debug(f"Level Image stats result table {len(result)}")
                self.display_component_results(result)

        if self.analyzer_type == "binary_image_stats":
            src = DataModel.g.dataset_uri(self.feature_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["feature_id"] = str(self.feature_source.value())
            all_params["threshold"] = self.threshold.value()

            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "binary_image_stats", **all_params)
            if result:
                logger.debug(f"Segmentation stats result table {len(result)}")
                self.display_component_results(result)

        if self.analyzer_type == "image_stats":
            src = DataModel.g.dataset_uri(self.features_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace

            all_params["feature_ids"] = str(self.features_source.value()[-1])

            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "image_stats", **all_params)
            if result:
                src_arr = decode_numpy(result)
                sc = MplCanvas(self, width=5, height=4, dpi=100)
                sc.axes.imshow(src_arr)
                self.add_row(sc, max_height=300)

        elif self.analyzer_type == "object_stats2":
            src = DataModel.g.dataset_uri(self.features_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["feature_ids"] = str(self.features_source.value()[-1])
            all_params["object_id"] = str(self.objects_source.value())
            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "object_stats", **all_params)
            if result:
                src_arr = decode_numpy(result)
                sc = MplCanvas(self, width=5, height=4, dpi=100)
                sc.axes.imshow(src_arr)
                self.add_row(sc, max_height=300)

        elif self.analyzer_type == "object_stats":
            src = DataModel.g.dataset_uri(self.features_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["feature_ids"] = str(self.features_source.value()[-1])
            all_params["object_id"] = str(self.objects_source.value())
            all_params["stat_name"] = self.stat_name_combo_box.value()
            logger.debug(f"Running analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "object_stats", **all_params)
            (point_features, img) = result
            if result:
                logger.debug(f"Object stats result table {len(point_features)}")

                tabledata = []
                for i in range(len(point_features)):
                    entry = (i, point_features[i])
                    tabledata.append(entry)

                tabledata = np.array(
                    tabledata,
                    dtype=[
                        ("index", int),
                        ("z", float),
                    ],
                )

                src_arr = decode_numpy(img)
                sc = MplCanvas(self, width=6, height=5, dpi=80)
                sc.axes.imshow(src_arr)
                sc.axes.axis("off")

                self.add_row(sc, max_height=500)

                if self.table_control is None:
                    self.table_control = TableWidget()
                    self.add_row(self.table_control.w, max_height=500)

                self.table_control.set_data(tabledata)
                self.collapse()
                self.expand()

        elif self.analyzer_type == "object_detection_stats":
            src = DataModel.g.dataset_uri(self.features_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["predicted_objects_id"] = str(
                self.predicted_objects_source.value()
            )
            all_params["gold_objects_id"] = str(self.gold_objects_source.value())

            logger.debug(f"Running object detection analyzer with params {all_params}")
            result = Launcher.g.run("analyzer", "object_detection_stats", **all_params)

            # if result:
            #     src_arr = decode_numpy(result)
            #     sc = MplCanvas(self, width=5, height=4, dpi=100)
            #     sc.axes.imshow(src_arr)
            #     self.add_row(sc, max_height=300)

        elif self.analyzer_type == "detector_predict":
            feature_names_list = [
                n.rsplit("/", 1)[-1] for n in self.features_source.value()
            ]
            src = DataModel.g.dataset_uri(
                self.features_source.value()[0], group="pipelines"
            )
            all_params = dict(src=src, dst=dst, modal=True)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["object_id"] = str(self.objects_source.value())
            all_params["feature_ids"] = feature_names_list
            all_params["model_fullname"] = self.model_fullname
            all_params["dst"] = self.pipeline_id
            result = Launcher.g.run("pipelines", self.analyzer_type, **all_params)

        elif self.analyzer_type == "spatial_clustering":
            src = DataModel.g.dataset_uri(self.feature_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["feature_id"] = str(self.feature_source.value())
            all_params["object_id"] = str(self.objects_source.value())
            result = Launcher.g.run("analyzer", "spatial_clustering", **all_params)
            logger.debug(f"spatial clustering result table {len(result)}")
            self.display_component_results2(result)
        elif self.analyzer_type == "remove_masked_objects":
            src = DataModel.g.dataset_uri(self.feature_source.value())
            all_params = dict(src=src, dst=dst, modal=False)
            all_params["workspace"] = DataModel.g.current_workspace
            all_params["feature_id"] = str(self.feature_source.value())
            all_params["object_id"] = str(self.objects_source.value())
            result = Launcher.g.run("analyzer", "remove_masked_objects", **all_params)
            logger.debug(f"remove_masked_objects result table {len(result)}")
            self.display_component_results2(result)

    def clustering_plot(self):
        src = DataModel.g.dataset_uri(self.features_source.value())
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_ids"] = str(self.features_source.value()[-1])
        all_params["object_id"] = str(self.objects_source.value())
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "image_stats", **all_params)
        if result:
            src_arr = decode_numpy(result)
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.imshow(src_arr)
            self.add_row(sc, max_height=300)

    def export_csv(self):
        full_path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output filename", ".", filter="*.csv"
        )
        if isinstance(full_path, tuple):
            full_path = full_path[0]

        out_df = pd.DataFrame(self.tabledata)
        out_df.to_csv(full_path)
        logger.debug(f"Exported to csv {full_path}")
