import numpy as np
from loguru import logger
from qtpy import QtWidgets
from survos2.model import DataModel
from survos2.frontend.components.base import LineEdit, ComboBox, HWidgets
from survos2.frontend.plugins.pipeline.base import PipelineCardBase


class LabelPostprocess(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_annotations_source(label="Layer Over: ")
        self._add_annotations_source2(label="Layer Base: ")
        # self.label_index = LineEdit(default=-1, parse=int)

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(
            self.annotations_source.value().rsplit("/", 1)[-1], group="annotations"
        )
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        print(self.annotations_source.value())

        if self.annotations_source.value():
            all_params["level_over"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        else:
            all_params["level_over"] = "None"
        all_params["level_base"] = str(self.annotations_source2.value().rsplit("/", 1)[-1])
        all_params["dst"] = self.dst
        all_params["selected_label_for_over"] = int(self.widgets["selected_label_for_over"].value())
        all_params["offset"] = int(self.widgets["offset"].value())
        all_params["base_offset"] = int(self.widgets["base_offset"].value())

        return all_params
