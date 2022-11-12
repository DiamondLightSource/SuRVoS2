import ast
import logging
import numpy as np
from loguru import logger
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal

from survos2.frontend.components.base import (
    VBox,
    ComboBox,
    PluginNotifier,
    LazyComboBox,
    LazyMultiComboBox,
    DataTableWidgetItem,
)

from survos2.frontend.control import Launcher
from survos2.frontend.plugins.annotation_tool import AnnotationComboBox
from survos2.frontend.plugins.annotations import LevelComboBox

from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.pipeline.rasterize import RasterizePoints
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox, RealSlider
from survos2.frontend.plugins.superregions import RegionComboBox
from survos2.frontend.plugins.features import FeatureComboBox

from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.frontend.utils import FileWidget

from survos2.frontend.plugins.pipeline.base import PipelineCardBase
from survos2.frontend.plugins.pipeline.superregion_segment import SuperregionSegment
from survos2.frontend.plugins.pipeline.postprocess import LabelPostprocess
from survos2.frontend.plugins.pipeline.cleaning import Cleaning, PerObjectCleaning
from survos2.frontend.plugins.pipeline.rasterize import RasterizePoints
from survos2.frontend.plugins.pipeline.watershed import Watershed
from survos2.frontend.plugins.pipeline.multiaxis_cnn import TrainMultiaxisCNN, PredictMultiaxisCNN
from survos2.frontend.plugins.pipeline.cnn3d import Train3DCNN, Predict3DCNN

from survos2.frontend.plugins.base import register_plugin, Plugin

from napari.qt.progress import progress

_PipelineNotifier = PluginNotifier()


def _fill_pipelines(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    result = Launcher.g.run("pipelines", "existing", **params)

    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


class PipelinesComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run("pipelines", "existing", **params)
        if result:
            self.addCategory("Segmentations")
            for fid in result:
                self.addItem(fid, result[fid]["name"])


@register_plugin
class PipelinesPlugin(Plugin):
    __icon__ = "fa.picture-o"
    __pname__ = "pipelines"
    __views__ = ["slice_viewer"]
    __tab__ = "pipelines"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.pipeline_combo = ComboBox()
        self.vbox = VBox(self, spacing=4)
        self.vbox.addWidget(self.pipeline_combo)
        self.pipeline_combo.currentIndexChanged.connect(self.add_pipeline)
        self.existing_pipelines = dict()
        self._populate_pipelines()

    def _populate_pipelines(self):
        self.pipeline_params = {}
        self.pipeline_combo.clear()
        self.pipeline_combo.addItem("Add segmentation")

        result = Launcher.g.run("pipelines", "available", workspace=True)

        if not result:
            params = {}
            params["category"] = "superregion"
            params["name"] = "s0"
            params["type"] = "superregion_segment"
            result = {}
            result[0] = params
            self.pipeline_params["superregion_segment"] = {
                "sr_params": {
                    "type": "sr2",
                }
            }
        else:
            all_categories = sorted(set(p["category"] for p in result))

            for i, category in enumerate(all_categories):
                self.pipeline_combo.addItem(category)
                self.pipeline_combo.model().item(i + len(self.pipeline_params) + 1).setEnabled(
                    False
                )

                for f in [p for p in result if p["category"] == category]:
                    self.pipeline_params[f["name"]] = f["params"]
                    self.pipeline_combo.addItem(f["name"])

    def add_pipeline(self, idx):
        if idx <= 0:
            return
        if self.pipeline_combo.itemText(idx) == "":
            return

        logger.debug(f"Adding pipeline {self.pipeline_combo.itemText(idx)}")

        pipeline_type = self.pipeline_combo.itemText(idx)
        self.pipeline_combo.setCurrentIndex(0)

        params = dict(pipeline_type=pipeline_type, workspace=True)
        result = Launcher.g.run("pipelines", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            self._add_pipeline_widget(fid, ftype, fname, True)
            _PipelineNotifier.notify()

    def _add_pipeline_widget(self, fid, ftype, fname, expand=False):
        if ftype == "superregion_segment":
            widget = SuperregionSegment(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "label_postprocess":
            widget = LabelPostprocess(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "per_object_cleaning":
            widget = PerObjectCleaning(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "cleaning":
            widget = Cleaning(fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier)
        elif ftype == "rasterize_points":
            widget = RasterizePoints(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "watershed":
            widget = Watershed(fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier)
        elif ftype == "train_multi_axis_cnn":
            widget = TrainMultiaxisCNN(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "predict_multi_axis_cnn":
            widget = PredictMultiaxisCNN(
                fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier
            )
        elif ftype == "train_3d_cnn":
            widget = Train3DCNN(fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier)
        elif ftype == "predict_3d_cnn":
            widget = Predict3DCNN(fid, ftype, fname, self.pipeline_params[ftype], _PipelineNotifier)
        else:
            widget = None
        # widget = PipelineCard(fid, ftype, fname, self.pipeline_params[ftype])

        if widget:
            widget.showContent(expand)
            self.vbox.addWidget(widget)
            self.existing_pipelines[fid] = widget

        return widget

    def clear(self):
        for pipeline in list(self.existing_pipelines.keys()):
            self.existing_pipelines.pop(pipeline).setParent(None)
        self.existing_pipelines = {}

    def setup(self):
        self._populate_pipelines()
        params = dict(workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace)
        result = Launcher.g.run("pipelines", "existing", **params)
        logger.debug(f"Pipeline result {result}")

        if result:
            # Remove pipelines that no longer exist in the server
            for pipeline in list(self.existing_pipelines.keys()):
                if pipeline not in result:
                    self.existing_pipelines.pop(pipeline).setParent(None)

            # Populate with new pipelines if any
            for pipeline in sorted(result):
                if pipeline in self.existing_pipelines:
                    continue
                params = result[pipeline]
                logger.debug(f"Pipeline params {params}")
                fid = params.pop("id", pipeline)
                ftype = params.pop("kind")
                fname = params.pop("name", pipeline)
                widget = self._add_pipeline_widget(fid, ftype, fname)
                if widget:
                    widget._update_params(params)
                    self.existing_pipelines[fid] = widget


class PipelineFunctionTest(PipelineCardBase):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        super().__init__(fid=fid, ftype=ftype, fname=fname, fparams=fparams)

    def setup(self):
        pass

    def compute_pipeline(self):
        pass
