from loguru import logger
from survos2.model import DataModel
from survos2.frontend.plugins.pipeline.base import PipelineCardBase


class RasterizePoints(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        self._add_annotations_source()
        self._add_feature_source()
        self._add_objects_source()
        self._add_param("acwe", type="SmartBoolean", default=False)
        self._add_param("size", type="FloatOrVector", default=(10.0, 10.0, 10.0))
        self._add_param("balloon", type="Float", default=0.0)
        self._add_param("threshold", type="Float", default=1.0)
        self._add_param("iterations", type="Int", default=3)
        self._add_param("smoothing", type="Int", default=1)
        self._add_param("selected_class", type="Int", default=1)

    def compute_pipeline(self):
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, modal=True)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = self.feature_source.value()
        all_params["object_id"] = str(self.objects_source.value())
        all_params["acwe"] = self.widgets["acwe"].value()
        all_params["dst"] = self.dst
        return all_params
