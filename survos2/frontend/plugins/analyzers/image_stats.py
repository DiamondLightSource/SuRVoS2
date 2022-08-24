from survos2.frontend.plugins.analyzers.base import MplCanvas, AnalyzerCardBase
from survos2.frontend.control import Launcher
from survos2.frontend.components.base import PushButton, LineEdit, HWidgets
from survos2.frontend.plugins.plugins_components import Label
from survos2.model import DataModel
from survos2.utils import decode_numpy
from loguru import logger


class BinaryImageStats(AnalyzerCardBase):
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
        self.threshold = LineEdit(default=0.5, parse=float)
        self.area_min = LineEdit(default=0, parse=int)
        self.area_max = LineEdit(default=1e12, parse=int)
        widget = HWidgets("Threshold:", self.threshold, self.area_min, self.area_max, stretch=1)
        self.add_row(widget)
        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["threshold"] = self.threshold.value()
        all_params["area_min"] = self.area_min.value()
        all_params["area_max"] = self.area_max.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "binary_image_stats", **all_params)
        if result:
            logger.debug(f"Segmentation stats result table {len(result)}")
            self.display_component_results(result)


class ImageStats(AnalyzerCardBase):
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
        self._add_features_source()
        self.plot_btn = PushButton("Plot")
        self.additional_buttons.append(self.plot_btn)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
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


class SegmentationStats(AnalyzerCardBase):
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
        self.label_index_A = LineEdit(default=1, parse=int)
        widget = HWidgets("Label index:", self.label_index_A)
        self.add_row(widget)

        self.add_source_selector2()
        self.label_index_B = LineEdit(default=1, parse=int)
        widget = HWidgets("Label index:", self.label_index_B)
        self.add_row(widget)

        self.dice_score = Label()
        widget = HWidgets("Dice: ", self.dice_score)
        self.add_row(widget)

        self.iou_score = Label()
        widget = HWidgets("IOU (Jaccard): ", self.iou_score)
        self.add_row(widget)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace

        all_params["modeA"] = self.radio_group.checkedId()
        all_params["modeB"] = self.radio_group2.checkedId()

        all_params["label_index_A"] = self.label_index_A.value()
        all_params["label_index_B"] = self.label_index_B.value()

        all_params["pipelines_id_A"] = str(self.pipelines_source.value())
        all_params["analyzers_id_A"] = str(self.analyzers_source.value())
        all_params["annotations_id_A"] = str(self.annotations_source.value())

        all_params["pipelines_id_B"] = str(self.pipelines_source2.value())
        all_params["analyzers_id_B"] = str(self.analyzers_source2.value())
        all_params["annotations_id_B"] = str(self.annotations_source2.value())

        result = Launcher.g.run("analyzer", "segmentation_stats", **all_params)

        logger.debug(f"{result}")

        dice, iou = result
        self.dice_score.setText(str(dice))

        self.iou_score.setText(str(iou))
