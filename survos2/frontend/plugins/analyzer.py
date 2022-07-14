
import numpy as np
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.qt.progress import progress
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton
from survos2.entity.cluster.cluster_plotting import (cluster_scatter,
                                                     image_grid2,
                                                     plot_clustered_img)
from survos2.entity.cluster.clusterer import select_clusters
from survos2.frontend.components.base import *
from survos2.frontend.components.entity import TableWidget
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.analyzers.base import MplCanvas, AnalyzerCardBase
from survos2.frontend.plugins.analyzers.image_stats import ImageStats, BinaryImageStats, SegmentationStats
from survos2.frontend.plugins.analyzers.patch_clusterer import ObjectAnalyzer, PatchStats, BinaryClassifier
from survos2.frontend.plugins.analyzers.geometry import PointGenerator
from survos2.frontend.plugins.analyzers.label_analysis import LabelAnalyzer, LabelSplitter, RemoveMaskedObjects, FindConnectedComponents
from survos2.frontend.plugins.analyzers.spatial_clustering import SpatialClustering
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.export import SuperRegionSegmentComboBox
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.pipelines import PipelinesComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.utils import FileWidget
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
            logger.debug("No analyzers.")
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

            analyzer_type = self.analyzers_combo.itemText(idx)
            if analyzer_type == "":
                return
            self.analyzers_combo.setCurrentIndex(0)
            logger.debug(f"Adding analyzer {analyzer_type}")
            from survos2.api.analyzer import __analyzer_names__

            order = __analyzer_names__.index(analyzer_type)
            params = dict(analyzer_type=analyzer_type, order=order, workspace=True)
            result = Launcher.g.run("analyzer", "create", **params)

            if result:
                analyzer_id = result["id"]
                analyzer_name = result["name"]
                analyzer_type = result["kind"]
                self._add_analyzer_widget(
                    analyzer_id, analyzer_name, analyzer_type, True
                )

    def _add_analyzer_widget(
        self, analyzer_id, analyzer_name, analyzer_type, expand=False
    ):
        
        if analyzer_type=="label_splitter":
            widget = LabelSplitter(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="label_analyzer":
            widget = LabelAnalyzer(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="remove_masked_objects":
            widget = RemoveMaskedObjects(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="image_stats":
            widget = ImageStats(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="binary_image_stats":
            widget = BinaryImageStats(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="binary_classifier":
            widget = BinaryClassifier(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="find_connected_components":
            widget = FindConnectedComponents(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="spatial_clustering":
            widget = SpatialClustering(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="point_generator":
            widget = PointGenerator(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="segmentation_stats":
            widget = SegmentationStats(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="patch_stats":
            widget = PatchStats(analyzer_id, analyzer_name, analyzer_type)
        elif analyzer_type=="object_analyzer":
            widget = ObjectAnalyzer(analyzer_id, analyzer_name, analyzer_type)
        else:
            print("No matching analyzer type.")
            return None
        
        widget.showContent(expand)
        self.analyzers_layout.addWidget(widget)
        self.existing_analyzer[analyzer_id] = widget
        return widget

    def clear(self):
        for analyzer in list(self.existing_analyzer.keys()):
            self.existing_analyzer.pop(analyzer).setParent(None)
        self.existing_analyzer = {}

    def setup(self):
        self._populate_analyzers()

        params = dict(
            workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace
        )

        result = Launcher.g.run("analyzer", "existing", **params)

        if result:
            logger.debug(f"Populating analyzer widgets with {result}")
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
                analyzer_name = params.pop("name", analyzer)
                analyzer_type = params.pop("kind", analyzer)
                logger.debug(f"Adding analyzer of type {analyzer_type}")

                if params.pop("kind", analyzer) != "unknown":
                    widget = self._add_analyzer_widget(
                        analyzer_id, analyzer_name, analyzer_type
                    )
                    widget.update_params(params)
                    self.existing_analyzer[analyzer_id] = widget
                else:
                    logger.debug(
                        "+ Skipping loading analyzer: {}, {}".format(
                            analyzer_id, analyzer_name, analyzer_type
                        )
                    )
        
# class ObjectDetectionStats():
#     def __init__(self, card):
#         card._add_features_source()
#         card._add_object_detection_stats_source()

class AnalyzerFunctionTest(AnalyzerCardBase):
    def __init__(self,analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            analyzer_name=analyzer_name,
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
    def setup(self):
        pass
    def calculate(self):
        pass







