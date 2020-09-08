import numpy as np

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal
from survos2.frontend.components.base import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.model import DataModel
from survos2.frontend.model import ClientData
from loguru import logger

from survos2.frontend.plugins.base import LazyComboBox, LazyMultiComboBox
from survos2.frontend.plugins.regions import RegionComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.annotation_tool import MultiAnnotationComboBox

_PipelineNotifier = PluginNotifier()

from survos2.frontend.control import Launcher
from survos2.server.config import cfg


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
        logger.debug(f"Result of pipelines existing: {result}")
        if result:
            self.addCategory("Segmentations")
            for fid in result:
                if result[fid]["kind"] == "pipelines":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class PipelinesPlugin(Plugin):
    __icon__ = "fa.picture-o"
    __pname__ = "pipelines"
    __views__ = ["slice_viewer"]

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

        result = None
        result = Launcher.g.run("pipelines", "available", workspace=True)

        if not result:
            params = {}
            params["category"] = "superregion"
            params["name"] = "s0"
            params["type"] = "superregion_segment"
            result = {}
            result[0] = params
            self.pipeline_params["superregion_segment"] = {
                "sr_params": {"type": "sr2",}
            }
        else:
            all_categories = sorted(set(p["category"] for p in result))

            for i, category in enumerate(all_categories):
                self.pipeline_combo.addItem(category)
                self.pipeline_combo.model().item(
                    i + len(self.pipeline_params) + 1
                ).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.pipeline_params[f["name"]] = f["params"]
                    self.pipeline_combo.addItem(f["name"])

    def add_pipeline(self, idx):
        if idx == 0:
            return
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
        widget = PipelineCard(fid, ftype, fname, self.pipeline_params[ftype])
        widget.showContent(expand)
        self.vbox.addWidget(widget)
        self.existing_pipelines[fid] = widget
        return widget

    def setup(self):
        params = dict(workspace=True)
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
                widget.update_params(params)
                self.existing_pipelines[fid] = widget


class PipelineCard(Card):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        self.pipeline_id = fid
        self.pipeline_type = ftype
        self.pipeline_name = fname

        super().__init__(
            fname, removable=True, editable=True, collapsible=True, parent=parent
        )

        self.params = fparams
        self.widgets = dict()

        self._add_features_source()
        self._add_annotations_source()
        self._add_regions_source()

        for pname, params in fparams.items():
            if pname not in ["src", "dst"]:
                self._add_param(pname, **params)

        self._add_compute_btn()
        self._add_view_btn()

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_pipeline)
        self.add_row(HWidgets(None, view_btn, Spacing(35)))

    def _add_features_source(self):

        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_annotations_source(self):

        self.annotations_source = LevelComboBox(full=True)  # SourceComboBox()
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(
            "Annotation:", self.annotations_source, Spacing(35), stretch=1
        )

        self.add_row(widget)

    def _add_regions_source(self):

        self.regions_source = RegionComboBox(full=True)  # SourceComboBox()
        self.regions_source.fill()
        self.regions_source.setMaximumWidth(250)

        widget = HWidgets("Superregions:", self.regions_source, Spacing(35), stretch=1)

        self.add_row(widget)

    def _add_param(self, name, type="String", default=None):
        if type == "Int":
            pipeline = LineEdit(default=default, parse=int)
        elif type == "Float":
            pipeline = LineEdit(default=default, parse=float)
        elif type == "FloatOrVector":
            pipeline = LineEdit3D(default=default, parse=float)
        else:
            pipeline = None

        if pipeline:
            self.widgets[name] = pipeline
            self.add_row(HWidgets(None, name, pipeline, Spacing(35)))

    def _add_compute_btn(self):
        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self.compute_pipeline)
        self.add_row(HWidgets(None, compute_btn, Spacing(35)))

    def update_params(self, params):
        if "source" in params:
            for source in params["source"]:
                self.features_source.select(source)

        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)

    def card_deleted(self):
        params = dict(pipeline_id=self.pipeline_id, workspace=True)
        result = Launcher.g.run("pipelines", "remove", **params)
        if result["done"]:
            self.setParent(None)
            _PipelineNotifier.notify()

    def view_pipeline(self):
        logger.debug(f"View pipeline_id {self.pipeline_id}")
        cfg.ppw.clientEvent.emit(
            {
                "source": "pipelines",
                "data": "view_pipeline",
                "pipeline_id": self.pipeline_id,
            }
        )

    def compute_pipeline(self):
        src_grp = None if self.annotations_source.currentIndex() == 0 else "pipelines"
        src = DataModel.g.dataset_uri(self.annotations_source.value(), group=src_grp)
        logger.info(f"Setting src to {self.annotations_source.value()} ")

        dst = DataModel.g.dataset_uri(self.pipeline_id, group="pipelines")
        feature_names_list = [
            n.rsplit("/", 1)[-1] for n in self.features_source.value()
        ]

        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = "test_s2"
        all_params["region_id"] = str(self.regions_source.value().rsplit("/", 1)[-1])
        all_params[
            "feature_ids"
        ] = feature_names_list  # ['002_gaussian_blur', '002_gaussian_blur']
        all_params["anno_id"] = str(self.annotations_source.value().rsplit("/", 1)[-1])
        all_params["dst"] = self.pipeline_id
        all_params.update({k: v.value() for k, v in self.widgets.items()})

        logger.info(f"Computing pipelines {self.pipeline_type} {all_params}")
        Launcher.g.run("pipelines", self.pipeline_type, **all_params)

    def card_title_edited(self, newtitle):
        params = dict(pipeline_id=self.pipeline_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("pipelines", "rename", **params)

        if result["done"]:
            _PipelineNotifier.notify()

        return result["done"]
