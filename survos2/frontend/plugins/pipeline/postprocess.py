

import numpy as np
from loguru import logger
from qtpy import QtWidgets
from survos2.model import DataModel
from survos2.frontend.components.base import LineEdit, ComboBox, HWidgets
from survos2.frontend.plugins.pipeline.base import PipelineCardBase

class LabelPostprocess(PipelineCardBase):
    def __init__(self,fid, ftype, fname, fparams, parent=None):
        super().__init__(
            fid=fid,
            ftype=ftype,
            fname=fname,
            fparams=fparams
        )
    def setup(self):
        self._add_annotations_source(label="Layer Over: ")
        self._add_annotations_source2(label="Layer Base: ")
        self.label_index = LineEdit(default=-1, parse=int)
    def compute_pipeline(self):
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
        all_params["dst"] = self.dst
        all_params["selected_label"] = int(self.widgets["selected_label"].value())
        all_params["offset"] = int(self.widgets["offset"].value())
        return all_params


class FeaturePostprocess(PipelineCardBase):
    def __init__(self,fid, ftype, fname, fparams, parent=None):
        super().__init__(
            fid=fid,
            ftype=ftype,
            fname=fname,
            fparams=fparams
        )
    def setup(self):
        self._add_feature_source()
        self._add_feature_source2()
        self.label_index = LineEdit(default=-1, parse=int)
        self.op_type = ComboBox()
        self.op_type.addItem(key="*")
        self.op_type.addItem(key="+")
        widget = HWidgets("Operation:", self.op_type, stretch=0)
        self.add_row(widget)
    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_A"] = str(self.feature_source.value())
        all_params["feature_B"] = str(self.feature_source2.value())
        all_params["dst"] = self.dst
        all_params["op"] = str(self.op_type.value())
        return all_params

