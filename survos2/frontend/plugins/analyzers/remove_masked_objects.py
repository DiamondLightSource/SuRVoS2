
import numpy as np
from loguru import logger

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

from survos2.frontend.plugins.analyzers.base import AnalyzerCardBase


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
        all_params["workspace"] = DataModel.g.current_workspace
        
        result = Launcher.g.run("analyzer", "remove_masked_objects", **all_params)
        logger.debug(f"remove_masked_objects result table {len(result)}")
        
        self.display_component_results2(result)
