import numpy as np

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton

from survos2.frontend.plugins.base import *
from survos2.frontend.components.base import *
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.components.icon_buttons import IconButton
from survos2.improc.utils import DatasetManager
from survos2.server.state import cfg

from survos2.frontend.components.entity import (
    TableWidget,
    SmallVolWidget,
    setup_entity_table,
)


@register_plugin
class AnalyzerPlugin(Plugin):

    __icon__ = "fa.picture-o"
    __pname__ = "analyzer"
    __views__ = ["slice_viewer"]
    __tab__ = "analyzer"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)

        self(
            IconButton("fa.plus", "Add Analyzer", accent=True),
            connect=("clicked", self.add_analyzer),
        )

        self.existing_analyzer = {}
        self.analyzer_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.analyzer_layout)

    def add_analyzer(self):
        params = dict(order=1, workspace=True)
        result = Launcher.g.run("analyzer", "create", **params)

        if result:
            analyzer_id = result["id"]
            analyzername = result["name"]

            self._add_analyzer_widget(analyzer_id, analyzername, True)

    def _add_analyzer_widget(self, analyzer_id, analyzername, expand=False):
        widget = AnalyzerCard(analyzer_id, analyzername)
        widget.showContent(expand)
        self.analyzer_layout.addWidget(widget)

        self.existing_analyzer[analyzer_id] = widget
        return widget

    def setup(self):
        params = dict(order=1, workspace=True)

        params["id"] = 0
        params["name"] = "analysis1"
        params["kind"] = "analyzer"
        result = {}
        result[0] = params

        result = Launcher.g.run("analyzer", "existing", **params)
        if result:
            # Remove analyzer that no longer exist in the server
            for entity in list(self.existing_analyzer.keys()):
                if entity not in result:
                    self.existing_analyzer.pop(entity).setParent(None)

            # Populate with new entity if any
            for entity in sorted(result):

                if entity in self.existing_analyzer:
                    continue
                params = result[entity]
                analyzer_id = params.pop("id", entity)
                analyzername = params.pop("name", entity)

                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_analyzer_widget(analyzer_id, analyzername)
                    widget.update_params(params)
                    self.existing_analyzer[analyzer_id] = widget
                else:
                    logger.debug(
                        "+ Skipping loading entity: {}, {}".format(
                            analyzer_id, analyzername
                        )
                    )


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

        self.calc_btn = PushButton("Compute")
        self.add_row(HWidgets(None, self.calc_btn, Spacing(35)))

        self.calc_btn.clicked.connect(self.calculate_analyzer)

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
        logger.debug(f"Edited entity title {newtitle}")
        params = dict(analyzer_id=self.analyzer_id, new_name=newtitle, workspace=True)
        result = Launcher.g.run("analyzer", "rename", **params)
        return result["done"]

    def update_params(self, params):
        pass

    def calculate_analyzer(self):
        src = DataModel.g.dataset_uri(self.features_source.value())
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzers")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_ids"] = str(self.features_source.value()[-1])

        Launcher.g.run("analyzer", "simple_stats", **all_params)
