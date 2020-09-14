import numpy as np
from loguru import logger

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal

from survos2.frontend.components.base import *
from survos2.frontend.plugins.base import *
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.plugins_components import SourceComboBox

_FeatureNotifier = PluginNotifier()


class FeatureComboBox(LazyComboBox):
    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(header=(None, "None"), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):

        _fill_features(self, full=self.full)


def _fill_features(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    result = Launcher.g.run("features", "existing", **params)
    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


@register_plugin
class FeaturesPlugin(Plugin):
    __icon__ = "fa.picture-o"
    __pname__ = "features"
    __views__ = ["slice_viewer"]
    __tab__ = "workspace"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.feature_combo = ComboBox()
        self.vbox = VBox(self, spacing=4)
        self.vbox.addWidget(self.feature_combo)
        self.feature_combo.currentIndexChanged.connect(self.add_feature)
        self.existing_features = dict()
        self._populate_features()

    def _populate_features(self):
        self.feature_params = {}
        self.feature_combo.clear()
        self.feature_combo.addItem("Add feature")
        result = Launcher.g.run("features", "available", workspace=True)

        if result:
            all_categories = sorted(set(p["category"] for p in result))
            for i, category in enumerate(all_categories):
                self.feature_combo.addItem(category)
                self.feature_combo.model().item(
                    i + len(self.feature_params) + 1
                ).setEnabled(False)
                for f in [p for p in result if p["category"] == category]:
                    self.feature_params[f["name"]] = f["params"]
                    self.feature_combo.addItem(f["name"])

    def add_feature(self, idx):
        if idx == 0:
            return
        feature_type = self.feature_combo.itemText(idx)
        self.feature_combo.setCurrentIndex(0)

        params = dict(feature_type=feature_type, workspace=True)
        result = Launcher.g.run("features", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            self._add_feature_widget(fid, ftype, fname, True)
            _FeatureNotifier.notify()

    def _add_feature_widget(self, fid, ftype, fname, expand=False):
        widget = FeatureCard(fid, ftype, fname, self.feature_params[ftype])
        widget.showContent(expand)
        self.vbox.addWidget(widget)
        self.existing_features[fid] = widget
        return widget

    def setup(self):
        params = dict(workspace=True)
        result = Launcher.g.run("features", "existing", **params)

        logger.debug(f"Feature result {result}")

        if result:
            # Remove features that no longer exist in the server
            for feature in list(self.existing_features.keys()):
                if feature not in result:
                    self.existing_features.pop(feature).setParent(None)

            # Populate with new features if any
            for feature in sorted(result):
                if feature in self.existing_features:
                    continue
                params = result[feature]
                logger.debug(f"Feature params {params}")
                fid = params.pop("id", feature)
                ftype = params.pop("kind")
                fname = params.pop("name", feature)
                widget = self._add_feature_widget(fid, ftype, fname)
                widget.update_params(params)
                self.existing_features[fid] = widget


class FeatureCard(Card):
    # clientEvent  = Signal(object)
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        self.feature_id = fid
        self.feature_type = ftype
        self.feature_name = fname

        super().__init__(
            fname, removable=True, editable=True, collapsible=True, parent=parent
        )

        self.params = fparams
        self.widgets = dict()

        self._add_source()
        for pname, params in fparams.items():
            if pname not in ["src", "dst"]:
                self._add_param(pname, **params)
        self._add_compute_btn()
        self._add_view_btn()

    def _add_source(self):
        chk_clamp = CheckBox("Clamp")
        self.cmb_source = SourceComboBox(self.feature_id)
        self.cmb_source.fill()
        widget = HWidgets(chk_clamp, self.cmb_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_param(self, name, type="String", default=None):
        if type == "Int":
            feature = LineEdit(default=default, parse=int)
        elif type == "Float":
            feature = LineEdit(default=default, parse=float)
        elif type == "FloatOrVector":
            feature = LineEdit3D(default=default, parse=float)
        else:
            feature = None

        if feature:
            self.widgets[name] = feature
            self.add_row(HWidgets(None, name, feature, Spacing(35)))

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_feature)
        self.add_row(HWidgets(None, view_btn, Spacing(35)))

    def _add_compute_btn(self):
        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self.compute_feature)
        self.add_row(HWidgets(None, compute_btn, Spacing(35)))

    def update_params(self, params):
        src = params.pop("source", None)
        if src is not None:
            self.cmb_source.select(src)
        for k, v in params.items():
            if k in self.widgets:
                self.widgets[k].setValue(v)

    def card_deleted(self):
        params = dict(feature_id=self.feature_id, workspace=True)
        result = Launcher.g.run("features", "remove", **params)
        if result["done"]:
            self.setParent(None)
            _FeatureNotifier.notify()

    def view_feature(self):
        logger.debug(f"View feature_id {self.feature_id}")
        from survos2.server.config import cfg

        cfg.ppw.clientEvent.emit(
            {
                "source": "features",
                "data": "view_feature",
                "feature_id": self.feature_id,
            }
        )

    def compute_feature(self):
        src_grp = None if self.cmb_source.currentIndex() == 0 else "features"

        src = DataModel.g.dataset_uri(self.cmb_source.value(), group=src_grp)
        logger.info(f"Setting src: {self.cmb_source.value()} ")

        dst = DataModel.g.dataset_uri(self.feature_id, group="features")

        logger.info(f"Setting dst: {self.feature_id}")
        logger.info(f"widgets.items() {self.widgets.items()}")

        all_params = dict(src=src, dst=dst, modal=False)
        all_params.update({k: v.value() for k, v in self.widgets.items()})

        logger.info(f"Computing features: {self.feature_type} {all_params}")
        Launcher.g.run("features", self.feature_type, **all_params)

    def card_title_edited(self, newtitle):
        params = dict(feature_id=self.feature_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("features", "rename", **params)

        if result["done"]:
            _FeatureNotifier.notify()

        return result["done"]
