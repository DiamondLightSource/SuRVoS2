from loguru import logger
from survos2.frontend.components.base import PushButton, LineEdit, HWidgets
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.analyzers.base import AnalyzerCardBase


class PointGenerator(AnalyzerCardBase):
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
        self._add_feature_source(label="Background mask")
        self.num_before_masking = LineEdit(default=500, parse=int)
        widget = HWidgets("Num before masking", self.num_before_masking)
        self.add_row(widget)
        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["bg_mask_id"] = str(self.feature_source.value())
        all_params["num_before_masking"] = int(self.num_before_masking.value())
        result = Launcher.g.run("analyzer", self.analyzer_type, **all_params)
        logger.debug(f"point_generator result table {len(result)}")
        self.display_component_results3(result)
