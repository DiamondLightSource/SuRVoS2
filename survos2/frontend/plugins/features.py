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
from survos2.server.state import cfg

from napari.qt import progress


_FeatureNotifier = PluginNotifier()


class FeatureComboBox(LazyComboBox):
    def __init__(self, full=False, parent=None):
        self.full = full
        super().__init__(header=(None, "None"), parent=parent)
        _FeatureNotifier.listen(self.update)

    def fill(self):
        _fill_features(self, full=self.full)


def _fill_features(combo, full=False, filter=True, ignore=None):
    params = dict(
        workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace,
        full=full,
        filter=filter,
    )
    logger.debug(f"Filling features from session: {params}")
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
    __tab__ = "features"

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
        if idx <= 0:
            return
        logger.debug(f"Adding feature {self.feature_combo.itemText(idx)}")
        if self.feature_combo.itemText(idx) == "":
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
        if ftype in self.feature_params:
            widget = FeatureCard(fid, ftype, fname, self.feature_params[ftype])
            widget.showContent(expand)
            self.vbox.addWidget(widget)
            self.existing_features[fid] = widget
            return widget

    def clear(self):
        for feature in list(self.existing_features.keys()):
            self.existing_features.pop(feature).setParent(None)
        self.existing_features = {}

    def setup(self):
        self._populate_features()
        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )
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

                logger.debug(f"fid {fid} ftype {ftype} fname {fname}")
                if params.pop("kind", "unknown") != "unknown":

                    widget = self._add_feature_widget(fid, ftype, fname)
                    widget.update_params(params)
                    self.existing_features[fid] = widget

                else:
                    logger.debug(
                        "+ Skipping loading feature: {}, {}".format(fid, fname)
                    )
                    if ftype:
                        widget = self._add_feature_widget(fid, ftype, fname)
                        if widget:
                            widget.update_params(params)
                            self.existing_features[fid] = widget


class FeatureCard(Card):
    def __init__(self, fid, ftype, fname, fparams, parent=None):
        self.feature_id = fid
        self.feature_type = ftype
        self.feature_name = fname

        super().__init__(
            fname, removable=True, editable=True, collapsible=True, parent=parent
        )

        self.params = fparams
        self.widgets = dict()

        # from qtpy.QtWidgets import QProgressBar
        # self.pbar = QProgressBar(self)
        # self.add_row(self.pbar)

        self._add_source()
        for pname, params in fparams.items():
            if pname not in ["src", "dst", "threshold"]:
                self._add_param(pname, **params)

        if self.feature_type == "wavelet":
            self.wavelet_type = ComboBox()
            self.wavelet_type.addItem(key="sym2")
            self.wavelet_type.addItem(key="sym3")
            self.wavelet_type.addItem(key="sym4")
            self.wavelet_type.addItem(key="sym6")
            self.wavelet_type.addItem(key="sym7")
            self.wavelet_type.addItem(key="sym8")
            self.wavelet_type.addItem(key="sym9")
            self.wavelet_type.addItem(key="haar")
            self.wavelet_type.addItem(key="db3")
            self.wavelet_type.addItem(key="db4")
            self.wavelet_type.addItem(key="db5")
            self.wavelet_type.addItem(key="db6")
            self.wavelet_type.addItem(key="db7")
            self.wavelet_type.addItem(key="db35")
            self.wavelet_type.addItem(key="coif1")
            self.wavelet_type.addItem(key="coif3")
            self.wavelet_type.addItem(key="coif7")
            self.wavelet_type.addItem(key="bior1.1")
            self.wavelet_type.addItem(key="bior2.2")
            self.wavelet_type.addItem(key="bior3.5")

            widget = HWidgets(
                "Wavelet type:", self.wavelet_type, Spacing(35), stretch=0
            )
            self.add_row(widget)

            self.wavelet_threshold = RealSlider(value=0.0, vmax=128, vmin=0)
            widget = HWidgets(
                "Threshold:", self.wavelet_threshold, Spacing(35), stretch=0
            )
            self.add_row(widget)

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
        elif type == "IntOrVector":
            feature = LineEdit3D(default=default, parse=int)
        elif type == "SmartBoolean":
            feature = CheckBox(checked=True)
        else:
            feature = None

        if feature:
            self.widgets[name] = feature
            self.add_row(HWidgets(None, name, feature, Spacing(35)))

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_feature)
        self.add_row(
            HWidgets(
                None,
                view_btn,
            )
        )

    def _add_compute_btn(self):
        compute_btn = PushButton("Compute", accent=True)
        compute_btn.clicked.connect(self.compute_feature)
        self.add_row(
            HWidgets(
                None,
                compute_btn,
            )
        )

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

        cfg.ppw.clientEvent.emit(
            {
                "source": "features",
                "data": "remove_layer",
                "layer_name": self.feature_id,
            }
        )

    def view_feature(self):
        logger.debug(f"View feature_id {self.feature_id}")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing feature")
            pbar.update(1)
            cfg.ppw.clientEvent.emit(
                {
                    "source": "features",
                    "data": "view_feature",
                    "feature_id": self.feature_id,
                }
            )
            pbar.update(1)    

    def compute_feature(self):
        with progress(total=3) as pbar:
            pbar.set_description("Calculating feature")
            # self.pbar.setValue(25)
            pbar.update(1)
            src_grp = None if self.cmb_source.currentIndex() == 0 else "features"

            src = DataModel.g.dataset_uri(self.cmb_source.value(), group=src_grp)
            logger.info(f"Setting src: {self.cmb_source.value()} ")

            dst = DataModel.g.dataset_uri(self.feature_id, group="features")
            # self.pbar.setValue(50)

            logger.info(f"Setting dst: {self.feature_id}")
            logger.info(f"widgets.items() {self.widgets.items()}")
            pbar.update(1)

            all_params = dict(src=src, dst=dst, modal=True)

            if self.feature_type == "wavelet":
                all_params["wavelet"] = str(self.wavelet_type.value())
                all_params["threshold"] = self.wavelet_threshold.value()

            all_params.update({k: v.value() for k, v in self.widgets.items()})

            logger.info(f"Computing features: {self.feature_type} {all_params}")
            # result = Launcher.g.run("features", self.feature_type, **all_params)
            Launcher.g.run("features", self.feature_type, **all_params)

            # pbar.update(1)
            # if result is not None:
            #     self.pbar.setValue(100)

    def card_title_edited(self, newtitle):
        params = dict(feature_id=self.feature_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("features", "rename", **params)

        if result["done"]:
            _FeatureNotifier.notify()

        return result["done"]
