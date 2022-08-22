import numpy as np
from loguru import logger
from qtpy import QtWidgets
from survos2.model import DataModel
from survos2.frontend.components.base import LineEdit
from survos2.frontend.plugins.pipeline.base import PipelineCardBase


class Cleaning(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_feature_source()
        self._add_annotations_source()

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=self.dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        return all_params


class PerObjectCleaning(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_feature_source()
        self._add_objects_source()

    def compute_pipeline(self):
        all_params = dict(dst=self.dst, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["patch_size"] = self.widgets["patch_size"].value()
        return all_params
