import numpy as np
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from napari.qt.progress import progress
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton
from survos2.entity.cluster.cluster_plotting import cluster_scatter, image_grid2, plot_clustered_img
from survos2.entity.cluster.clusterer import select_clusters
from survos2.frontend.components.base import VBox, SimpleComboBox, HWidgets, PushButton, LineEdit
from survos2.frontend.control import Launcher
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.frontend.plugins.analyzers.base import AnalyzerCardBase, MplCanvas


class DBSCAN_Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DBSCAN_Panel, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.eps = LineEdit(default=0.1, parse=float)
        widget = HWidgets("EPS:", self.eps, stretch=1)
        vbox.addWidget(widget)


class HDBSCAN_Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(HDBSCAN_Panel, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.min_cluster_size = LineEdit(default=2, parse=int)
        self.min_samples = LineEdit(default=1, parse=int)
        widget = HWidgets(
            "Min Cluster Size:", self.min_cluster_size, "Min Samples:", self.min_samples
        )
        vbox.addWidget(widget)


class SpatialClustering(AnalyzerCardBase):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
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
        self._add_feature_source()
        self._add_objects_source()
        self.DBSCAN_Panel = DBSCAN_Panel()
        self.HDBSCAN_Panel = HDBSCAN_Panel()

        self.clustering_method_combo_box = SimpleComboBox(full=True, values=["DBSCAN", "HDBSCAN"])
        widget = HWidgets("Method:", self.clustering_method_combo_box)
        self.add_row(widget)
        self.clustering_method_combo_box.fill()
        self.clustering_method_combo_box.currentIndexChanged.connect(
            self._on_clustering_method_changed
        )
        self.clustering_method_container = QtWidgets.QWidget()
        clustering_method_vbox = VBox(self, spacing=4)
        clustering_method_vbox.setContentsMargins(0, 0, 0, 0)
        self.clustering_method_container.setLayout(clustering_method_vbox)
        self.add_row(self.clustering_method_container, max_height=500)
        self.clustering_method_container.layout().addWidget(self.DBSCAN_Panel)

        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())

        methods = ["DBSCAN", "HDBSCAN"]
        algorithm = str(methods[int(self.clustering_method_combo_box.value())])
        if algorithm == "DBSCAN":
            all_params["params"] = {
                "algorithm": algorithm,
                "eps": self.DBSCAN_Panel.eps.value(),
                "min_samples": 1,
            }
        elif algorithm == "HDBSCAN":
            all_params["params"] = {
                "algorithm": algorithm,
                "min_cluster_size": self.HDBSCAN_Panel.min_cluster_size.value(),
                "min_samples": self.HDBSCAN_Panel.min_samples.value(),
            }

        result = Launcher.g.run("analyzer", "spatial_clustering", **all_params)

        self.display_component_results2(result)

    def _on_clustering_method_changed(self, idx):
        if idx == 1:
            self.DBSCAN_Panel.setParent(None)
            self.clustering_method_container.layout().addWidget(self.HDBSCAN_Panel)
        elif idx == 0:
            self.HDBSCAN_Panel.setParent(None)
            self.clustering_method_container.layout().addWidget(self.DBSCAN_Panel)
