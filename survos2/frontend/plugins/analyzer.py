import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton

from survos2.frontend.components.base import *
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy


def _fill_analyzers(combo, full=False, filter=True, ignore=None):
    params = dict(workspace=True, full=full, filter=filter)
    result = Launcher.g.run("analyzer", "existing", **params)

    if result:
        for fid in result:
            if fid != ignore:
                combo.addItem(fid, result[fid]["name"])


class AnalyzersComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)
        result = Launcher.g.run("analyzer", "existing", **params)
        logger.debug(f"Result of analyzer existing: {result}")
        if result:
            self.addCategory("analyzer")
            for fid in result:
                if result[fid]["kind"] == "analyzer":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class AnalyzerPlugin(Plugin):

    __icon__ = "fa.picture-o"
    __pname__ = "analyzer"
    __views__ = ["slice_viewer"]
    __tab__ = "analyzer"

    def __init__(self, parent=None):
        self.analyzers_combo = ComboBox()
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)
        vbox.addWidget(self.analyzers_combo)
        self.analyzers_combo.currentIndexChanged.connect(self.add_analyzer)
        self.existing_analyzer = {}
        self.analyzers_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.analyzers_layout)
        self._populate_analyzers()

    def _populate_analyzers(self):
        self.analyzer_params = {}
        self.analyzers_combo.clear()
        self.analyzers_combo.addItem("Add analyzer")

        result = None
        result = Launcher.g.run("analyzer", "available", workspace=True)

        if not result:
            logger.debug("No analyzers")
            params = {}
            params["category"] = "IMAGE"
            params["name"] = "s0"
            params["type"] = "simple_stats"
            result = {}
            result[0] = params
        else:

            all_categories = sorted(set(p["category"] for p in result))

            logger.debug(f"Analyzer {result}")
            for i, category in enumerate(all_categories):
                self.analyzers_combo.addItem(category)
                self.analyzers_combo.model().item(
                    i + len(self.analyzer_params) + 1
                ).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.analyzer_params[f["name"]] = f["params"]
                    self.analyzers_combo.addItem(f["name"])

    def add_analyzer(self, idx):
        if idx != 0:
            logger.debug(f"Adding analyzer {idx}")
            analyzer_type = self.analyzers_combo.itemText(idx)
            self.analyzers_combo.setCurrentIndex(0)

            params = dict(analyzer_type=analyzer_type, workspace=True)
            result = Launcher.g.run("analyzer", "create", **params)

            if result:
                analyzer_id = result["id"]
                analyzername = result["name"]
                self._add_analyzer_widget(analyzer_id, analyzername, True)

    def _add_analyzer_widget(self, analyzer_id, analyzername, expand=False):
        widget = AnalyzerCard(analyzer_id, analyzername)
        widget.showContent(expand)
        self.analyzers_layout.addWidget(widget)

        self.existing_analyzer[analyzer_id] = widget
        return widget

    def setup(self):
        params = dict(order=1, workspace=True)

        params["id"] = 0
        params["name"] = "analysis1"
        params["kind"] = "simple_stats"
        
        #result = {}
        #result[0] = params

        result = Launcher.g.run("analyzer", "existing", **params)
        
        if result:
            logger.debug("Populating analyzer widgets with {result}")
            # Remove analyzer that no longer exist in the server
            for analyzer in list(self.existing_analyzer.keys()):
                if analyzer not in result:
                    self.existing_analyzer.pop(analyzer).setParent(None)

            # Populate with new analyzer if any
            for analyzer in sorted(result):
                if analyzer in self.existing_analyzer:
                    continue
                params = result[analyzer]
                analyzer_id = params.pop("id", analyzer)
                analyzername = params.pop("name", analyzer)

                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_analyzer_widget(analyzer_id, analyzername)
                    widget.update_params(params)
                    self.existing_analyzer[analyzer_id] = widget
                else:
                    logger.debug(
                        "+ Skipping loading analyzer: {}, {}".format(
                            analyzer_id, analyzername
                        )
                    )


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class AnalyzerCard(Card):
    def __init__(self, analyzer_id, analyzername, parent=None):
        super().__init__(
            title=analyzername,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.analyzer_id = analyzer_id
        self.analyzername = analyzername

        self._add_features_source()
        self._add_objects_source()

        self.calc_btn = PushButton("Compute")
        self.plot_btn = PushButton("Plot")
        self.add_row(HWidgets(None, self.calc_btn, Spacing(35)))
        self.add_row(HWidgets(None, self.plot_btn, Spacing(35)))
        self.calc_btn.clicked.connect(self.calculate_analyzer)
        self.plot_btn.clicked.connect(self.clustering_plot)

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)

        widget = HWidgets("Objects:", self.objects_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def _add_features_source(self):

        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source, Spacing(35), stretch=1)
        self.add_row(widget)

    def card_deleted(self):
        params = dict(analyzer_id=self.analyzer_id, workspace=True)
        result = Launcher.g.run("analyzer", "remove", **params)
        if result["done"]:
            self.setParent(None)

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited analyzer title {newtitle}")
        params = dict(analyzer_id=self.analyzer_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("analyzer", "rename", **params)
        return result["done"]

    def update_params(self, params):
        pass

    def calculate_analyzer(self):
        src = DataModel.g.dataset_uri(self.features_source.value())
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_ids"] = str(self.features_source.value()[-1])
        all_params["object_id"] = str(self.objects_source.value())
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "simple_stats", **all_params)

        if result:
            src_arr = decode_numpy(result)
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.imshow(src_arr)
            # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
            self.add_row(sc, max_height=300)

    def clustering_plot(self):
        src = DataModel.g.dataset_uri(self.features_source.value())
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_ids"] = str(self.features_source.value()[-1])
        all_params["object_id"] = str(self.objects_source.value())
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "simple_stats", **all_params)
        if result:
            src_arr = decode_numpy(result)
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.imshow(src_arr)
            # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
            self.add_row(sc, max_height=300)
