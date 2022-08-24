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
from survos2.frontend.components.base import (
    VBox,
    SimpleComboBox,
    HWidgets,
    PushButton,
    CheckBox,
    LineEdit3D,
    LineEdit,
)

from survos2.frontend.components.entity import TableWidget
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher

from survos2.frontend.plugins.annotations import LevelComboBox

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
from survos2.frontend.plugins.analyzers.base import AnalyzerCardBase, MplCanvas

umap_metrics = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "mahalanobis",
    "canberra",
    "braycurtis",
    "haversine",
    "cosine",
    "correlation",
]


class BinaryClassifier(AnalyzerCardBase):
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
        self._add_feature_source(label="Proposal segmentation")
        self._add_objects_source()
        self._add_objects_source2(title="Background Objects:")
        self._add_feature_source2(label="Mask:")
        self._add_feature_source3(label="Feature:")
        self.area_min = LineEdit(default=0, parse=int)
        self.area_max = LineEdit(default=1e14, parse=int)
        self.score_threshold = LineEdit(default=0.95, parse=float)
        self.num_before_masking = LineEdit(default=500, parse=int)
        self.bvol_dim = LineEdit3D(default=32)
        widget = HWidgets(
            "Area min:", self.area_min, "Area max: ", self.area_max, "Patch dim: ", self.bvol_dim
        )
        self.add_row(widget)
        widget = HWidgets(
            "Score threshold:", self.score_threshold, "Num before masking", self.num_before_masking
        )
        self.add_row(widget)
        self.load_as_objects_btn = PushButton("Load as Objects")
        self.additional_buttons.append(self.load_as_objects_btn)
        self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        self._add_model_file()

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["proposal_id"] = str(self.feature_source.value())
        all_params["mask_id"] = str(self.feature_source2.value())
        all_params["feature_id"] = str(self.feature_source3.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["background_id"] = str(self.objects_source2.value())
        all_params["area_max"] = int(self.area_max.value())
        all_params["area_min"] = int(self.area_min.value())
        all_params["bvol_dim"] = self.bvol_dim.value()
        all_params["num_before_masking"] = int(self.num_before_masking.value())
        all_params["score_thresh"] = float(self.score_threshold.value())
        all_params["dst"] = self.analyzer_id
        all_params["model_fullname"] = self.model_fullname
        result = Launcher.g.run("analyzer", self.analyzer_type, **all_params)
        logger.debug(f"remove_masked_objects result table {len(result)}")
        self.display_component_results3(result)


class PatchStats(AnalyzerCardBase):
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
        self.stat_name_combo_box = SimpleComboBox(
            full=True, values=["Mean", "Std", "Var", "Median", "Sum"]
        )
        self.stat_name_combo_box.fill()
        self.box_dimension = LineEdit(default=16, parse=int)
        widget = HWidgets(
            "Statistic name:", self.stat_name_combo_box, "Box dimension: ", self.box_dimension
        )
        self.add_row(widget)

    def calculate(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value(), group="features")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["stat_name"] = self.stat_name_combo_box.value()
        all_params["box_size"] = self.box_dimension.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "patch_stats", **all_params)

        (point_features, img) = result
        if result:
            logger.debug(f"Object stats result table {len(point_features)}")
            tabledata = []
            for i in range(len(point_features)):
                entry = (i, point_features[i])
                tabledata.append(entry)
            tabledata = np.array(
                tabledata,
                dtype=[
                    ("index", int),
                    ("z", float),
                ],
            )
            src_arr = decode_numpy(img)
            sc = MplCanvas(self, width=6, height=5, dpi=80)
            sc.axes.imshow(src_arr)
            sc.axes.axis("off")
            self.add_row(sc, max_height=500)
            if self.table_control is None:
                self.table_control = TableWidget()
                self.add_row(self.table_control.w, max_height=500)

            self.table_control.set_data(tabledata)
            self.collapse()
            self.expand()


class ObjectAnalyzer(AnalyzerCardBase):
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
        self.flipxy = CheckBox(checked=True)
        self.patch_size = LineEdit3D(default=32, parse=int)
        self.TSNE_Panel = TSNE_Panel()
        self.UMAP_Panel = UMAP_Panel()

        self.feature_extraction_method_combo_box = SimpleComboBox(full=True, values=["CNN", "HOG"])
        self.feature_extraction_method_combo_box.fill()

        self.axis_combo_box = SimpleComboBox(full=True, values=["0", "1", "2"])
        self.axis_combo_box.fill()

        self.embedding_method_combo_box = SimpleComboBox(full=True, values=["TSNE", "UMAP"])
        self.embedding_method_combo_box.fill()

        widget = HWidgets(
            "Extraction Method:",
            self.feature_extraction_method_combo_box,
            "Patch size: ",
            self.patch_size,
            "Flip XY:",
            self.flipxy,
            stretch=2,
        )

        self.add_row(widget)
        widget = HWidgets(
            "Embedding Method:",
            self.embedding_method_combo_box,
            "Axis:",
            self.axis_combo_box,
            self.patch_size,
            self.flipxy,
            stretch=2,
        )
        self.add_row(widget)

        self.embedding_method_combo_box.currentIndexChanged.connect(
            self._on_embedding_method_changed
        )
        self.embedding_method_container = QtWidgets.QWidget()
        embedding_method_vbox = VBox(self, spacing=4)
        embedding_method_vbox.setContentsMargins(0, 0, 0, 0)
        self.embedding_method_container.setLayout(embedding_method_vbox)
        self.add_row(self.embedding_method_container, max_height=500)
        self.embedding_method_container.layout().addWidget(self.TSNE_Panel)
        self.min_cluster_size = LineEdit(default=3, parse=int)
        widget = HWidgets("min_cluster_size:", self.min_cluster_size)
        self.add_row(widget)

        self.plot_clusters = CheckBox(checked=False)
        widget = HWidgets("Plot clusters:", self.plot_clusters)
        self.add_row(widget)

    def calculate(self):
        if len(self.object_analyzer_plots) > 0:
            for plot in self.object_analyzer_plots:
                plot.setParent(None)
                plot = None
        self.object_analyzer_plots = []
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["bvol_dim"] = self.patch_size.value()
        methods = ["TSNE", "UMAP"]
        all_params["embedding_method"] = str(methods[int(self.embedding_method_combo_box.value())])
        all_params["axis"] = int(self.axis_combo_box.value())
        all_params["flipxy"] = self.flipxy.value()
        feature_extraction_methods = ["CNN", "HOG"]
        all_params["feature_extraction_method"] = str(
            feature_extraction_methods[int(self.feature_extraction_method_combo_box.value())]
        )
        embedding_params = {}
        if all_params["embedding_method"] == "TSNE":
            embedding_params["n_components"] = int(self.TSNE_Panel.n_components.value())
            embedding_params["n_iter"] = int(self.TSNE_Panel.n_iter.value())
            embedding_params["perplexity"] = int(self.TSNE_Panel.perplexity.value())
        else:
            embedding_params["n_components"] = int(self.UMAP_Panel.n_components.value())
            embedding_params["n_neighbors"] = int(self.UMAP_Panel.n_neighbors.value())
            embedding_params["min_dist"] = float(self.UMAP_Panel.min_dist.value())
            embedding_params["spread"] = float(self.UMAP_Panel.spread.value())
            embedding_params["metric"] = str(
                umap_metrics[int(self.UMAP_Panel.metric_combo_box.value(key=True))]
            )
        all_params["embedding_params"] = embedding_params
        all_params["min_cluster_size"] = int(self.min_cluster_size.value())

        logger.debug(f"Running analyzer with params {all_params}")
        _, labels, entities_arr, selected_images_arr, standard_embedding = Launcher.g.run(
            "analyzer", "object_analyzer", **all_params
        )

        entities_arr = np.array(entities_arr)
        entities_arr[:, 3] = labels
        self.entities_arr = entities_arr
        selected_images_arr = decode_numpy(selected_images_arr)

        if standard_embedding:
            sc = MplCanvas(self, width=8, height=8, dpi=100)
            # sc.axes.imshow(src_arr)
            sc.axes.margins(0)
            sc.axes.axis("off")
            # if all_params["bvol_dim"][0] < 32:
            #     skip_px = 1
            # else:
            #     skip_px = 2
            skip_px = 2
            standard_embedding = decode_numpy(standard_embedding)
            plot_clustered_img(
                standard_embedding,
                np.array(labels),
                ax=sc.axes,
                images=selected_images_arr[:, ::skip_px, ::skip_px],
            )
            self.object_analyzer_plots.append(sc)

            if self.plot_clusters.value():
                labels = np.array(labels)
                for l in np.unique(labels):
                    selected_images = selected_images_arr[labels == l]
                    if len(selected_images) < 7:
                        n_cols = len(selected_images)
                    else:
                        n_cols = 6
                    n_rows = min(5, (len(selected_images) // n_cols) + 1)
                    logger.debug(f"Making MplGridCanvas with {n_rows} rows and {n_cols} columns.")
                    sc2 = MplGridCanvas(
                        self, width=8, height=8, num_rows=n_rows, num_cols=n_cols, dpi=100
                    )
                    image_grid2(selected_images, n_cols, sc2.fig, sc2.axesarr, bigtitle=str(l))
                    self.object_analyzer_plots.append(sc2)

            for k in self.object_analyzer_plots:
                # self.vbox.addWidget(k)
                self.add_row(k, max_height=500)

            self.display_clustering_results(labels)

    def _on_embedding_method_changed(self, idx):
        if idx == 0:
            self.UMAP_Panel.setParent(None)
            self.embedding_method_container.layout().addWidget(self.TSNE_Panel)
        elif idx == 1:
            self.TSNE_Panel.setParent(None)
            self.embedding_method_container.layout().addWidget(self.UMAP_Panel)


class TSNE_Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TSNE_Panel, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.n_components = LineEdit(default=8, parse=int)
        self.n_iter = LineEdit(default=1000, parse=int)
        self.perplexity = LineEdit(default=50, parse=int)
        widget = HWidgets("n_components:", self.n_components, "n_iter:", self.n_iter, stretch=1)
        vbox.addWidget(widget)
        widget = HWidgets(
            "perplexity:",
            self.perplexity,
        )
        vbox.addWidget(widget)


class UMAP_Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(UMAP_Panel, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.n_components = LineEdit(default=2, parse=int)
        self.n_neighbors = LineEdit(default=10, parse=int)
        self.min_dist = LineEdit(default=0.3, parse=float)
        self.spread = LineEdit(default=1.0, parse=float)

        widget = HWidgets(
            "n_components:", self.n_components, "n_neighbors:", self.n_neighbors, stretch=1
        )
        vbox.addWidget(widget)

        self.metric_combo_box = SimpleComboBox(full=True, values=umap_metrics)
        self.metric_combo_box.fill()
        widget = HWidgets(
            "min_dist:", self.min_dist, "spread:", self.spread, "metric:", self.metric_combo_box
        )
        vbox.addWidget(widget)


class MplGridCanvas(FigureCanvasQTAgg):
    def __init__(
        self, parent=None, width=5, height=4, num_rows=2, num_cols=2, dpi=100, suptitle="Feature"
    ):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axesarr = np.zeros((num_rows, num_cols), dtype=object)

        for i in range(num_rows):
            for j in range(num_cols):
                idx = i * num_cols + j
                self.axesarr[i, j] = self.fig.add_subplot(num_rows, num_cols, idx + 1)

        self.suptitle = suptitle
        super(MplGridCanvas, self).__init__(self.fig)

    def set_suptitle(self, suptitle):
        self.suptitle = suptitle
        self.fig.suptitle(suptitle)
