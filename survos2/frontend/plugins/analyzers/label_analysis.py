import numpy as np
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.qt.progress import progress
from survos2.entity.cluster.cluster_plotting import cluster_scatter, image_grid2, plot_clustered_img

from survos2.frontend.control import Launcher
from survos2.frontend.components.base import (
    HWidgets,
    Slider,
    PushButton,
    Card,
    SimpleComboBox,
    LineEdit,
    CheckBox,
)

from survos2.model import DataModel

from survos2.frontend.plugins.analyzers.base import AnalyzerCardBase, MplCanvas
from survos2.frontend.plugins.analyzers.constants import feature_names


class LabelSplitter(AnalyzerCardBase):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            analyzer_name=analyzer_name,
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )

    def setup(self):
        self.add_source_selector()
        self._add_feature_source()
        self.background_label = LineEdit(default=0, parse=int)
        widget = HWidgets("Background label:", self.background_label, stretch=1)
        self.add_row(widget)

        self.add_rules_btn = PushButton("Add Rule")
        self.add_rules_btn.clicked.connect(self._add_rule)
        self.refresh_rules_btn = PushButton("Refresh rules")
        self.refresh_rules_btn.clicked.connect(self._setup_ops)
        self.feature_name_combo_box = SimpleComboBox(full=True, values=feature_names)
        self.feature_name_combo_box.fill()
        self.add_row(HWidgets(self.add_rules_btn, self.refresh_rules_btn))
        self.add_row(HWidgets("Explore feature name: ", self.feature_name_combo_box))
        self._add_labelsplitter_view_btns()

    def calculate(self):
        if len(self.plots) > 0:
            for plot in self.plots:
                plot.setParent(None)
                plot = None
            self.plots = []

        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.pipelines_source.value(), group="pipelines")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["pipelines_id"] = str(self.pipelines_source.value())
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["analyzers_id"] = str(self.analyzers_source.value())
        all_params["annotations_id"] = str(self.annotations_source.value())
        all_params["mode"] = self.radio_group.checkedId()
        all_params["background_label"] = self.background_label.value()

        split_ops = {}
        split_feature_indexes = []
        split_feature_thresholds = []

        if len(self.op_cards) > 0:
            for j, op_card in enumerate(self.op_cards):
                split_op = {}
                split_feature_index = int(op_card.feature_name_combo_box.value())
                split_op["split_feature_index"] = str(split_feature_index)
                split_feature_indexes.append(split_feature_index)
                split_op["split_op"] = op_card.split_op_combo_box.value()
                split_op["split_threshold"] = op_card.split_threshold.value()
                split_feature_thresholds.append(float(op_card.split_threshold.value()))
                split_ops[j] = split_op

        else:
            split_op = {}
            split_feature_index = int(self.feature_name_combo_box.value())
            split_op["split_feature_index"] = str(split_feature_index)
            split_feature_indexes.append(split_feature_index)
            split_op["split_op"] = 1
            split_op["split_threshold"] = 0
            split_feature_thresholds.append(0)
            split_ops[0] = split_op

        all_params["split_ops"] = split_ops
        all_params["json_transport"] = True

        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "label_splitter", **all_params)
        print(result)
        result_features, features_array = result
        features_ndarray = np.array(features_array)
        logger.debug(f"Shape of features_array: {features_ndarray.shape}")

        if features_array:
            logger.debug(f"Segmentation stats result table: {len(features_array)}")
            feature_arrays = []
            feature_titles = []

            for s in split_feature_indexes:
                feature_title = feature_names[int(s)]
                feature_plot_array = features_ndarray[:, int(s)]
                feature_arrays.append(feature_plot_array)
                feature_titles.append(feature_title)
                logger.debug(f"Titles of feature names: {feature_titles}")
                logger.debug(f"Split feature thresholds: {split_feature_thresholds}")
            self.display_splitter_plot(
                feature_arrays, titles=feature_titles, vert_line_at=split_feature_thresholds
            )
            self.display_splitter_results(result_features)


class LabelAnalyzer(AnalyzerCardBase):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            analyzer_name=analyzer_name,
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )

    def setup(self):
        self.add_source_selector()
        self._add_feature_source()

        self.background_label = LineEdit(default=0, parse=int)
        widget = HWidgets("Background label:", self.background_label, stretch=1)
        self.add_row(widget)
        self.feature_name_combo_box = SimpleComboBox(full=True, values=feature_names)
        self.feature_name_combo_box.fill()
        self.add_row(HWidgets("Explore feature name: ", self.feature_name_combo_box))
        self._add_labelsplitter_view_btns()

    def calculate(self):
        if len(self.plots) > 0:
            for plot in self.plots:
                plot.setParent(None)
                plot = None
            self.plots = []

        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.pipelines_source.value(), group="pipelines")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["pipelines_id"] = str(self.pipelines_source.value())
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["analyzers_id"] = str(self.analyzers_source.value())
        all_params["annotations_id"] = str(self.annotations_source.value())
        all_params["mode"] = self.radio_group.checkedId()
        all_params["background_label"] = self.background_label.value()

        split_ops = {}
        split_feature_indexes = [int(self.feature_name_combo_box.value())]
        split_feature_thresholds = []
        all_params["split_ops"] = split_ops
        all_params["json_transport"] = True  # needed as api call uses a dict

        logger.debug(f"Running analyzer with params {all_params}")
        result_features, features_array, bvols = Launcher.g.run(
            "analyzer", "label_analyzer", **all_params
        )
        features_ndarray = np.array(features_array)

        if features_array:
            logger.debug(f"Segmentation stats result table: {len(features_array)}")
            feature_arrays = []
            feature_titles = []

            for j, s in enumerate(split_feature_indexes):
                feature_title = feature_names[int(s)]
                feature_plot_array = features_ndarray[:, int(s)]
                feature_arrays.append(feature_plot_array)
                feature_titles.append(feature_title)
                logger.debug(f"Titles of feature names: {feature_titles}")

            self.display_splitter_results(result_features)
            self.display_splitter_plot(feature_arrays, titles=feature_titles, vert_line_at=None)


class RemoveMaskedObjects(AnalyzerCardBase):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            analyzer_name=analyzer_name,
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )

    def setup(self):
        self._add_feature_source()
        self._add_objects_source()
        self.invert_checkbox = CheckBox(checked=True)
        self.add_row(HWidgets("Invert:", self.invert_checkbox, stretch=0))
        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["invert"] = self.invert_checkbox.value()

        result = Launcher.g.run("analyzer", "remove_masked_objects", **all_params)
        logger.debug(f"remove_masked_objects result table {len(result)}")
        self.display_component_results(result)


class FindConnectedComponents(AnalyzerCardBase):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            analyzer_name=analyzer_name,
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )

    def setup(self):
        self.add_source_selector()

        self.label_index = LineEdit(default=0, parse=int)
        widget = HWidgets("Label Index:", self.label_index, stretch=1)
        self.add_row(widget)

        self.area_min = LineEdit(default=0, parse=int)
        self.area_max = LineEdit(default=1e14, parse=int)
        widget = HWidgets("Area min:", self.area_min, "Area max: ", self.area_max)
        self.add_row(widget)

        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        # self._add_view_btn()

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.pipelines_source.value(), group="pipelines")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["pipelines_id"] = str(self.pipelines_source.value())
        all_params["analyzers_id"] = str(self.analyzers_source.value())
        all_params["annotations_id"] = str(self.annotations_source.value())
        all_params["mode"] = self.radio_group.checkedId()
        all_params["label_index"] = self.label_index.value()
        all_params["area_min"] = self.area_min.value()
        all_params["area_max"] = self.area_max.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "find_connected_components", **all_params)
        if result:
            logger.debug(f"Segmentation stats result table {len(result)}")
            self.display_component_results(result)
