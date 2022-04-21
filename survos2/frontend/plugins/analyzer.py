from survos2.frontend.plugins.pipelines import PipelinesComboBox
from survos2.frontend.plugins.features import FeatureComboBox
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from napari.qt.progress import progress

from survos2.frontend.components.base import *
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.objects import ObjectComboBox
from survos2.frontend.plugins.annotations import LevelComboBox
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.plugins.export import SuperRegionSegmentComboBox
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.utils import decode_numpy
from survos2.frontend.components.entity import TableWidget
from survos2.frontend.utils import FileWidget

feature_names = ["z",
                 "y",
                 "x", 
                 "Sum", 
                 "Mean", 
                 "Std", 
                 "Var", 
                 "bb_vol", 
                 "bb_vol_log10", 
                 "bb_vol_depth", 
                 "bb_vol_height", 
                 "bb_vol_width", 
                 "ori_vol", 
                 "ori_vol_log10", 
                 "ori_vol_depth", 
                 "ori_vol_height", 
                 "ori_vol_width", 
                 "seg_surface_area", 
                 "seg_volume", 
                 "seg_sphericity"]

umap_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski","mahalanobis", "canberra", "braycurtis", "haversine", "cosine", "correlation"]

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
        widget = AnalyzerCard(analyzer_id, analyzer_name, analyzer_type)
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




class TSNE_Panel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TSNE_Panel, self).__init__(parent=parent)
        vbox = QtWidgets.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        self.setLayout(vbox)
        self.n_components = LineEdit(default=8, parse=int)
        self.n_iter = LineEdit(default=1000, parse=int)
        self.perplexity = LineEdit(default=50, parse=int)
        widget = HWidgets(
            "n_components:", self.n_components, "n_iter:", self.n_iter, stretch=1
        )
        vbox.addWidget(widget)
        widget = HWidgets(
            "perplexity:", self.perplexity, 
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
        
        self.metric_combo_box = SimpleComboBox(
                full=True, values=umap_metrics)
        self.metric_combo_box.fill()
        widget = HWidgets("min_dist:", self.min_dist, "spread:", self.spread, "metric:", self.metric_combo_box)
        vbox.addWidget(widget)



class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, suptitle="Feature"):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.suptitle = suptitle
        #if self.suptitle:
        #    self.fig.suptitle(self.suptitle)
        super(MplCanvas, self).__init__(self.fig)
    def set_suptitle(self,suptitle):
        self.suptitle = suptitle
        self.fig.suptitle(suptitle)


class RuleCard(Card):
    def __init__(self, title, collapsible=True, removable=True, editable=True,parent=None):
        super().__init__(
            title=title, collapsible=collapsible, removable=removable, editable=editable, parent=parent
        )
        self.title = title
        self.parent = parent
        self.feature_name_combo_box = SimpleComboBox(
            full=True, values=feature_names
        )
        self.feature_name_combo_box.fill()
        self.split_op_combo_box = SimpleComboBox(
            full=True, values=["None", ">", "<"]
        )
        self.split_op_combo_box.fill()
        self.split_threshold = LineEdit(default=0, parse=float)
        measure_widget = HWidgets(
            "Measurement:", self.feature_name_combo_box,  stretch=1
        )
        splitop_widget = HWidgets(
            "Split operation:",
            self.split_op_combo_box,
            self.split_threshold,
            
            stretch=1,
        )
        self.vbox.addWidget(measure_widget)
        self.add_to_widget_list(measure_widget)   
        self.vbox.addWidget(splitop_widget)
        self.setMinimumHeight(260)
        self.add_to_widget_list(splitop_widget)
        
    def card_deleted(self):
        logger.debug(f"Deleted Rule {self.title}")
        self.parent.op_cards.remove(self)
        self.setParent(None)
        

class AnalyzerCard(Card):
    def __init__(self, analyzer_id, analyzer_name, analyzer_type, parent=None):
        super().__init__(
            title=analyzer_name,
            collapsible=True,
            removable=True,
            editable=True,
            parent=parent,
        )
        self.analyzer_id = analyzer_id
        self.analyzer_name = analyzer_name
        self.analyzer_type = analyzer_type
        self.annotations_source = (
            "001_level"  # default annotation level to use for labels
        )
        additional_buttons = []
        self.annotations_selected = False
        self.op_cards = []
        self.model_fullname = "None"

        if self.analyzer_type == "label_splitter":
            self.add_source_selector()
 
            self.background_label = LineEdit(default=0, parse=float)
            widget = HWidgets("Background label:", self.background_label, stretch=1)
            self.add_row(widget)

            self.add_rules_btn = PushButton("Add Rule")
            self.add_rules_btn.clicked.connect(self._add_rule)
            self.refresh_rules_btn = PushButton("Refresh plots")
            self.refresh_rules_btn.clicked.connect(self._setup_ops)

            widget = HWidgets(self.add_rules_btn, self.refresh_rules_btn,stretch=0)
            self.add_row(widget)

            self.export_csv_btn = PushButton("Export CSV")
            self.load_as_objects_btn = PushButton("Load as Objects")

            self.add_row(HWidgets(None, self.load_as_objects_btn, self.export_csv_btn, ))
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
            self.export_csv_btn.clicked.connect(self.export_csv)

            self.feature_name_combo_box = SimpleComboBox(
            full=True, values=feature_names
            )
            self.feature_name_combo_box.fill()
                       
            self.add_row(HWidgets("Explore feature name: ", self.feature_name_combo_box, stretch=0))
            self._add_view_btn() 
        elif self.analyzer_type == "image_stats":
            self._add_features_source()
            self.plot_btn = PushButton("Plot")
            additional_buttons.append(self.plot_btn)
            #self.plot_btn.clicked.connect(self.clustering_plot)
        elif self.analyzer_type == "level_image_stats":
            self._add_annotations_source()
            self.statistic_name_combo_box = SimpleComboBox(
                full=True, values=["Mean", "Std", "Var"]
            )
            widget = HWidgets(
                "Statistic:", self.statistic_name_combo_box,  stretch=1
            )
            self.add_row(widget)
            self.label_index = LineEdit(default=1, parse=float)
            widget = HWidgets("Level index:", self.label_index,  stretch=1)
            self.add_row(widget)
        elif self.analyzer_type == "binary_image_stats":
            self._add_feature_source()
            self.threshold = LineEdit(default=0.5, parse=float)
            self.area_min = LineEdit(default=0, parse=int)
            self.area_max = LineEdit(default=1e12, parse=int)
            

            widget = HWidgets("Threshold:", self.threshold, self.area_min, self.area_max, stretch=1)
            self.add_row(widget)
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "patch_stats":
            self._add_features_source()
            self._add_objects_source()
            self.stat_name_combo_box = SimpleComboBox(
                full=True, values=["Mean", "Std", "Var", "Median", "Sum"]
            )
            self.stat_name_combo_box.fill()
            self.box_dimension = LineEdit(default=16, parse=int)
            widget = HWidgets(
                "Statistic name:", self.stat_name_combo_box,  "Box dimension: ", self.box_dimension
            )
            self.add_row(widget)
        elif self.analyzer_type == "object_detection_stats":
            self._add_features_source()
            self._add_object_detection_stats_source()
        elif self.analyzer_type == "find_connected_components":
            self.add_source_selector()

            self.label_index = LineEdit(default=0, parse=int)
            widget = HWidgets("Label Index:", self.label_index,  stretch=1)
            self.add_row(widget)

            self.area_min = LineEdit(default=0, parse=int)
            self.area_max = LineEdit(default=1e14, parse=int)
            widget = HWidgets("Area min:", self.area_min, "Area max: ", self.area_max)
            self.add_row(widget)
            
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
            self._add_view_btn()

        elif self.analyzer_type == "point_generator":
            self._add_feature_source(label="Background mask")
            self.num_before_masking = LineEdit(default=500, parse=int)
            widget = HWidgets("Num before masking", self.num_before_masking)
            self.add_row(widget)
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "binary_classifier":
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
            widget = HWidgets("Area min:", self.area_min, "Area max: ", self.area_max, "Patch dim: ", self.bvol_dim)
            self.add_row(widget)
            widget = HWidgets("Score threshold:", self.score_threshold, "Num before masking", self.num_before_masking)
            self.add_row(widget)
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
            self._add_model_file()
        elif self.analyzer_type == "remove_masked_objects":
            self._add_feature_source()
            self._add_objects_source()
            self.invert_checkbox = CheckBox(checked=True)
            self.add_row(
                HWidgets("Invert:", self.invert_checkbox, stretch=0)
            )
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "spatial_clustering":
            self._add_feature_source()
            self._add_objects_source()
            self.eps = LineEdit(default=0.1, parse=float)
            widget = HWidgets("DBScan EPS:", self.eps,  stretch=1)
            self.add_row(widget)
            self.load_as_objects_btn = PushButton("Load as Objects")
            additional_buttons.append(self.load_as_objects_btn)
            self.load_as_objects_btn.clicked.connect(self.load_as_objects)
        elif self.analyzer_type == "object_analyzer":
            self._add_feature_source()
            self._add_objects_source()

            self.patch_size = LineEdit3D(default=32, parse=int)
            self.TSNE_Panel = TSNE_Panel()
            self.UMAP_Panel = UMAP_Panel()    
            self.axis_combo_box = SimpleComboBox(
                full=True, values=["0", "1","2"]
            )
            self.axis_combo_box.fill()
            
            self.embedding_method_combo_box = SimpleComboBox(
                full=True, values=["TSNE", "UMAP"]
            )
            self.embedding_method_combo_box.fill()
            
            widget = HWidgets(
                "Method:", self.embedding_method_combo_box, "Axis:", self.axis_combo_box, self.patch_size, stretch=2
            )
            self.add_row(widget)
            self.embedding_method_combo_box.currentIndexChanged.connect(self._on_embedding_method_changed)
            self.embedding_method_container = QtWidgets.QWidget()
            embedding_method_vbox = VBox(self, spacing=4)
            embedding_method_vbox.setContentsMargins(0, 0, 0, 0)
            self.embedding_method_container.setLayout(embedding_method_vbox)
            self.add_row(self.embedding_method_container, max_height=500)
            self.embedding_method_container.layout().addWidget(self.TSNE_Panel)
            self.min_cluster_size = LineEdit(default=3, parse=int)
            widget = HWidgets(
                "min_cluster_size:", self.min_cluster_size)
            self.add_row(widget)
        self.calc_btn = PushButton("Compute")
        self.add_row(HWidgets(None, self.calc_btn))

        if len(additional_buttons) > 0:
            self.add_row(HWidgets(None, *additional_buttons, ))
        self.calc_btn.clicked.connect(self.calculate_analyzer)
        self.table_control = None
        self.plots = []
        self.object_analyzer_plots = []
        self.object_analyzer_controls = []
    
    def add_source_selector(self):    
        # radio buttons to select source type
        self.radio_group = QtWidgets.QButtonGroup()
        self.radio_group.setExclusive(True)
        pipelines_rb = QRadioButton("Pipelines")
        pipelines_rb.setChecked(True)
        pipelines_rb.toggled.connect(self._pipelines_rb_checked)
        self.radio_group.addButton(pipelines_rb, 1)
        analyzers_rb = QRadioButton("Analyzers")
        analyzers_rb.toggled.connect(self._analyzers_rb_checked)
        self.radio_group.addButton(analyzers_rb, 2)
        annotations_rb = QRadioButton("Annotation")
        self.radio_group.addButton(annotations_rb, 3)
        annotations_rb.toggled.connect(self._annotations_rb_checked)
        self.add_row(HWidgets(pipelines_rb, analyzers_rb, annotations_rb))
        
        self.source_container = QtWidgets.QWidget()
        source_vbox = VBox(self, spacing=4)
        source_vbox.setContentsMargins(0, 0, 0, 0)
        self.source_container.setLayout(source_vbox)
        self.add_row(self.source_container)
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        self.pipelines_widget = HWidgets(
        "Segmentation:", self.pipelines_source,  stretch=1
        )
        self.pipelines_widget.setParent(None)

        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)
        self.annotations_widget = HWidgets("Annotation", self.annotations_source,  stretch=1)
        self.annotations_widget.setParent(None)

        self.analyzers_source = AnalyzersComboBox()
        self.analyzers_source.fill()
        self.analyzers_source.setMaximumWidth(250)
        self.analyzers_widget = HWidgets(
            "Analyzers:", self.analyzers_source, stretch=1
        )
        self.analyzers_widget.setParent(None)
        self.source_container.layout().addWidget(self.pipelines_widget)
        self.current_widget = self.pipelines_widget
        self._add_feature_source()

    def _on_embedding_method_changed(self, idx):
        if idx == 0:
            self.UMAP_Panel.setParent(None)
            self.embedding_method_container.layout().addWidget(self.TSNE_Panel)
        elif idx == 1:
            self.TSNE_Panel.setParent(None)
            self.embedding_method_container.layout().addWidget(self.UMAP_Panel)
    def _pipelines_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.pipelines_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.pipelines_widget
            self.annotations_selected = False
    def _analyzers_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.analyzers_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.analyzers_widget
            self.annotations_selected = False
    def _annotations_rb_checked(self, enabled):
        if enabled:
            self.source_container.layout().addWidget(self.annotations_widget)
            if self.current_widget:
                self.current_widget.setParent(None)
            self.current_widget = self.annotations_widget
            self.annotations_selected = True
    def _setup_ops(self):
        print(f"Current number of op cards {len(self.op_cards)}")      
        if self.table_control:
            self.table_control.w.setParent(None)
            self.table_control = None
        if len(self.plots) > 0:
            for plot in self.plots:
                plot.setParent(None)
                plot = None
            self.plots = []
    def _setup_object_analyzer_plots(self):
        if len(self.object_analyzer_plots) > 0:
            for plot in self.object_analyzer_plots:
                plot.setParent(None)
                plot = None
            self.object_analyzer_plots = []
    def _add_rule(self):
        op_card = RuleCard(title="Rule", editable=True, collapsible=False, removable=True, parent=self)
        self.op_cards.append(op_card)
        self.add_row(op_card)
        self.add_to_widget_list(op_card)
    
    def load_as_objects(self):
        logger.debug("Load analyzer result as objects")
        from survos2.entity.entities import load_entities_via_file

        load_entities_via_file(self.entities_arr, flipxy=False)
        cfg.ppw.clientEvent.emit(
            {"source": "analyzer_plugin", "data": "faster_refresh", "value": None}
        )

    def _add_objects_source(self):
        self.objects_source = ObjectComboBox(full=True)
        self.objects_source.fill()
        self.objects_source.setMaximumWidth(250)
        widget = HWidgets("Objects:", self.objects_source,  stretch=1)
        self.add_row(widget)
    def _add_objects_source2(self, title="Objects"):
        self.objects_source2 = ObjectComboBox(full=True)
        self.objects_source2.fill()
        self.objects_source2.setMaximumWidth(250)
        widget = HWidgets(title, self.objects_source2,  stretch=1)
        self.add_row(widget)


    def _add_object_detection_stats_source(self):
        self.gold_objects_source = ObjectComboBox(full=True)
        self.gold_objects_source.fill()
        self.gold_objects_source.setMaximumWidth(250)
        widget = HWidgets(
            "Gold Objects:", self.gold_objects_source,  stretch=1
        )
        self.add_row(widget)

        self.predicted_objects_source = ObjectComboBox(full=True)
        self.predicted_objects_source.fill()
        self.predicted_objects_source.setMaximumWidth(250)
        widget = HWidgets(
            "Predicted Objects:", self.predicted_objects_source,  stretch=1
        )
        self.add_row(widget)

    def _add_annotations_source(self, label="Annotation"):
        self.annotations_source = LevelComboBox(full=True)
        self.annotations_source.fill()
        self.annotations_source.setMaximumWidth(250)

        widget = HWidgets(label, self.annotations_source,  stretch=1)

        self.add_row(widget)

    def _add_pipelines_source(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets(
            "Segmentation:", self.pipelines_source,  stretch=1
        )
        self.add_row(widget)


    def _add_pipelines_source2(self):
        self.pipelines_source = PipelinesComboBox()
        self.pipelines_source.fill()
        self.pipelines_source.setMaximumWidth(250)
        widget = HWidgets(
            "Segmentation:", self.pipelines_source,  stretch=1
        )
        self.add_row(widget)
        load_as_objects = PushButton("Load as Objects")
        return load_as_objects

    def _add_analyzers_source(self):
        self.analyzers_source = AnalyzersComboBox()
        self.analyzers_source.fill()
        self.analyzers_source.setMaximumWidth(250)
        widget = HWidgets(
            "Analyzers:", self.analyzers_source,  stretch=1
        )
        self.add_row(widget)

    def _add_feature_source(self, label="Feature:"):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source,  stretch=1)
        self.add_row(widget)

    def _add_feature_source3(self, label="Feature:"):
        self.feature_source3 = FeatureComboBox()
        self.feature_source3.fill()
        self.feature_source3.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source3,  stretch=1)
        self.add_row(widget)

    def _add_feature_source2(self, label="Feature:"):
        self.feature_source2 = FeatureComboBox()
        self.feature_source2.fill()
        self.feature_source2.setMaximumWidth(250)
        widget = HWidgets(label, self.feature_source2,  stretch=1)
        self.add_row(widget)

    def _add_features_source(self):
        self.features_source = MultiSourceComboBox()
        self.features_source.fill()
        self.features_source.setMaximumWidth(250)

        widget = HWidgets("Features:", self.features_source,  stretch=1)
        self.add_row(widget)

    def _add_view_btn(self):
        view_btn = PushButton("View", accent=True)
        view_btn.clicked.connect(self.view_analyzer)
        load_as_annotation_btn = PushButton("Load as annotation", accent=True)
        load_as_annotation_btn.clicked.connect(self.load_as_annotation)
        load_as_float_btn = PushButton("Load as feature", accent=True)
        load_as_float_btn.clicked.connect(self.load_as_float)
        self.add_row(
            HWidgets(
                None, load_as_float_btn, load_as_annotation_btn, view_btn, 
            )
        )
    
    def _add_view_entities(self):
        view_btn = PushButton("View Entities", accent=True)
        view_btn.clicked.connect(self.view_entities)
        self.add_row(
            HWidgets(
                None, view_btn, 
            )
        )
    def load_data(self, path):
        self.model_fullname = path
        print(f"Setting model fullname: {self.model_fullname}")

    def _add_model_file(self):
        self.filewidget = FileWidget(extensions="*.pt", save=False)
        self.add_row(self.filewidget)
        self.filewidget.path_updated.connect(self.load_data)

    def view_analyzer(self):
        logger.debug(f"View analyzer_id {self.analyzer_id}")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing analyzer")
            pbar.update(1)
            
            if self.annotations_source:
                if self.annotations_selected:
                    level_id = self.annotations_source.value().rsplit("/", 1)[-1]
                else:
                    level_id = "001_level"
                logger.debug(f"Assigning annotation level {level_id}")

                cfg.ppw.clientEvent.emit(
                    {
                        "source": "analyzer",
                        "data": "view_pipeline",
                        "pipeline_id": self.analyzer_id,
                        "level_id": level_id,
                    }
                )
            
            pbar.update(1)
            

    def load_as_float(self):
        logger.debug(f"Loading analyzer result {self.analyzer_id} as float image.")

        # get analyzer output
        src = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
        # create new float image
        params = dict(feature_type="raw", workspace=True)
        result = Launcher.g.run("features", "create", **params)

        if result:
            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

            dst = DataModel.g.dataset_uri(fid, group="features")
            with DatasetManager(dst, out=dst, dtype="float32", fillvalue=0) as DM:
                DM.out[:] = src_arr

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "faster_refresh", "value": None}
            )

    def load_as_annotation(self):
        logger.debug(f"Loading analyzer result {self.analyzer_id} as annotation.")

        # get analyzer output
        src = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
            src_arr = DM.sources[0][:]
        label_values = np.unique(src_arr)

        # create new level
        params = dict(level=self.analyzer_id, workspace=True)
        result = Launcher.g.run("annotations", "add_level", workspace=True)

        # create a blank label for each unique value in the analyzer output array
        if result:
            level_id = result["id"]

            for v in label_values:
                params = dict(
                    level=level_id,
                    idx=int(v),
                    name=str(v),
                    color="#11FF11",
                    workspace=True,
                )
                label_result = Launcher.g.run("annotations", "add_label", **params)

            # derive label colours from given annotation
            params = dict(
                level=str("001_level"),
                workspace=True,
            )
            anno_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            params = dict(level=str(level_id), workspace=True)
            level_result = Launcher.g.run("annotations", "get_levels", **params)[0]

            try:
                # set the new level color mapping to the mapping from the given annotation
                for v in level_result["labels"].keys():
                    if v in anno_result["labels"]:
                        label_hex = anno_result["labels"][v]["color"]
                        label = dict(
                            idx=int(v),
                            name=str(v),
                            color=label_hex,
                        )
                        params = dict(level=result["id"], workspace=True)
                        label_result = Launcher.g.run(
                            "annotations", "update_label", **params, **label
                        )
            except Exception as err:
                logger.debug(f"Exception {err}")

            fid = result["id"]
            ftype = result["kind"]
            fname = result["name"]
            logger.debug(f"Created new object in workspace {fid}, {ftype}, {fname}")

            dst = DataModel.g.dataset_uri(fid, group="annotations")
            with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
                DM.out[:] = src_arr

            cfg.ppw.clientEvent.emit(
                {"source": "workspace_gui", "data": "faster_refresh", "value": None}
            )

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
        logger.debug(f"Analyzer update params {params}")
        # for k, v in params.items():
        #     if k in self.widgets:
        #         self.widgets[k].setValue(v)

        if "anno_id" in params:
            if params["anno_id"] is not None:
                if isinstance(params["anno_id"], list):
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"][0])
                    )
                else:
                    self.annotations_source.select(
                        os.path.join("annotations/", params["anno_id"])
                    )
        
        if "object_id" in params:
            if params["object_id"] is not None:
                self.objects_source.select(
                    os.path.join("objects/", params["object_id"])
                )
        if "feature_id" in params:
            self.feature_source.select(params["feature_id"])
        if "feature_ids" in params:
            for source in params["feature_ids"]:
                self.features_source.select(os.path.join("features/", source))
        if "region_id" in params:
            if params["region_id"] is not None:
                self.regions_source.select(
                    os.path.join("regions/", params["region_id"])
                )

    def display_splitter_results(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
                result[i][4],
                result[i][5],
                result[i][6],
                result[i][7],
                result[i][8],
                result[i][9],
                result[i][10],
                result[i][11],
                result[i][12],
                result[i][13],
                result[i][14],
                result[i][15],
                result[i][16],
                result[i][17],
                result[i][18],
                result[i][19],

            )
            tabledata.append(entry)

            entity = (result[i][0], result[i][2], result[i][1], 0)
            entities.append(entity)

        tabledata = np.array(
            tabledata,
            dtype=[
                ("index", int),
                ("z", int),
                ("x", int),
                ("y", int),
                ("Sum", float),
                ("Mean", float),
                ("Std", float),
                ("Var", float),
                ("BB Vol", float),
                ("BB Vol Log10", float),
                ("BB Depth", float),
                ("BB Height", float),
                ("BB Width", float),
                ("OrientBB Vol", float),
                ("OrientBB Vol Log10", float),
                ("OrientBB Depth", float),
                ("OrientBB Height", float),
                ("OrientBB Width", float),
                ("Seg Surface Area", float),
                ("Seg Volume", float),
                ("Seg Sphericity", float),
            ],
        )

        if self.table_control is None:
            self.table_control = TableWidget()
            max_height = 500
            self.table_control.w.setProperty("header", False)
            self.table_control.w.setMaximumHeight(max_height)
            self.vbox.addWidget(self.table_control.w)
            self.total_height += 500 + self.spacing
            self.setMinimumHeight(self.total_height)
            
        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()
        self.tabledata = tabledata
        self.entities_arr = np.array(entities)

    def display_component_results(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][1],
                result[i][2],
                result[i][3],
                result[i][0],
            )
            tabledata.append(entry)

            entity = (result[i][0], result[i][1], result[i][2], 0)
            entities.append(entity)

        tabledata = np.array(
            tabledata,
            dtype=[
                ("index", int),
                ("z", int),
                ("x", int),
                ("y", int),
                ("area", int),
            ],
        )

        if self.table_control is None:
            self.table_control = TableWidget()
            self.add_row(self.table_control.w, max_height=500)

        self.table_control.set_data(tabledata)
        self.collapse()
        self.expand()

        self.entities_arr = np.array(entities)

    def display_component_results2(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
            )
            tabledata.append(entry)

            entity = (result[i][1], result[i][2], result[i][3], 0)
            entities.append(entity)

        self.entities_arr = np.array(entities)


    def display_component_results3(self, result):
        entities = []
        tabledata = []

        for i in range(len(result)):
            entry = (
                i,
                result[i][0],
                result[i][1],
                result[i][2],
                result[i][3],
            )
            tabledata.append(entry)

            entity = (result[i][0], result[i][1], result[i][2], 0)
            entities.append(entity)

        self.entities_arr = np.array(entities)


    def display_splitter_plot(self, feature_arrays, titles=[], vert_line_at=None):
        for i, feature_array in enumerate(feature_arrays):
            self.plots.append(MplCanvas(self, width=5, height=5, dpi=100))
            max_height = 600
            self.plots[i].setProperty("header", False)
            self.plots[i].setMaximumHeight(max_height)
            self.vbox.addWidget(self.plots[i])
            self.total_height += 500 + self.spacing
            self.setMinimumHeight(self.total_height)

            colors = ['r','y','b','c','m','g']
            y, x, _ = self.plots[i].axes.hist(feature_array, bins=16, color=colors[i])
            self.plots[i].axes.set_title(titles[i])
            
            if vert_line_at:
                print(f"Plotting vertical line at: {vert_line_at[i]} {y.max()}")
                self.plots[i].axes.axvline(x=vert_line_at[i], ymin=0, ymax=y.max(), color="k")

    def display_clustering_results(self, labels):
        for ctl in self.object_analyzer_controls:
            ctl.setParent(None)
            ctl = None
        self.labels_combo = MultiComboBox()
        self.vbox.addWidget(self.labels_combo)
        self.object_analyzer_controls.append(self.labels_combo)
        for el in np.unique(labels):
            self.labels_combo.addItem(str(el), value=str(el))
        self.load_cluster_as_objects_btn = PushButton("Load Cluster as Objects")
        self.vbox.addWidget(self.load_cluster_as_objects_btn)
        self.object_analyzer_controls.append(self.load_cluster_as_objects_btn)
        self.load_cluster_as_objects_btn.clicked.connect(self.load_cluster_as_objects)
    

    def view_entities(self):
        logger.debug(f"Transferring entities to viewer")
        cfg.ppw.clientEvent.emit(
            {
                "source": "analyzer",
                "data": "view_entities",
                "entities" : self.entities_arr,
                "flipxy": self.flipxy_checkbox.value(),
            }
        )


    def load_cluster_as_objects(self):
        logger.debug("Load cluster as objects")
        from survos2.entity.entities import load_entities_via_file
        labels = self.labels_combo.value()
        labels = [int(l) for l in labels]
        print(f"Selected cluster: {labels}")
        
        selected = []
        for l in labels:
            selected.append(np.where(self.entities_arr[:,3] == l)[0])
        selected = np.concatenate(selected)
        selected_b = np.zeros_like(self.entities_arr[:,3])
        for l in selected:
            selected_b[l] = 1
        selected_b = selected_b > 0
        self.entities_arr = self.entities_arr[selected_b]
        load_entities_via_file(self.entities_arr, flipxy=False)
        cfg.ppw.clientEvent.emit(
            {"source": "analyzer_plugin", "data": "refresh", "value": None}
        )

    # def clustering_plot(self, src_arr):
    #     if len(self.object_analyzer_plots) > 0:
    #         for plot in self.object_analyzer_plots:
    #             plot.setParent(None)
    #             plot = None
    #     self.object_analyzer_plots = []

    #     sc = MplCanvas(self, width=5, height=4, dpi=100)
    #     sc.axes.imshow(src_arr)
    #     max_height = 600
    #     sc.setProperty("header", False)
    #     sc.setMaximumHeight(max_height)
    #     self.object_analyzer_plots.append(sc)
    #     self.vbox.addWidget(self.object_analyzer_plots[0])
    #     print("Added clustering plot")

    def export_csv(self):
        full_path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output filename", ".", filter="*.csv"
        )
        if isinstance(full_path, tuple):
            full_path = full_path[0]

        out_df = pd.DataFrame(self.tabledata)
        out_df.to_csv(full_path)
        logger.debug(f"Exported to csv {full_path}")

    def calc_label_splitter(self):
        if len(self.plots) > 0:
            for plot in self.plots:
                plot.setParent(None)
                plot = None
            self.plots = []

        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.pipelines_source.value(), group="pipelines")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["pipelines_id"] = str(self.pipelines_source.value())
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["analyzers_id"] = str(self.analyzers_source.value())
        all_params["annotations_id"] = str(self.annotations_source.value())
        all_params["mode"] = self.radio_group.checkedId()

        all_params["background_label"] = self.background_label.value()

        split_ops = {}
        split_feature_indexes = []
        split_feature_thresholds = []

        if len(self.op_cards) > 0:
            for j, op_card in enumerate(self.op_cards):
                split_op = {}
                split_feature_index = int(op_card.feature_name_combo_box.value())
                split_op["split_feature_index"] = str(split_feature_index)
                split_feature_indexes.append(split_feature_index)
                split_op["split_op"] = op_card.split_op_combo_box.value()
                split_op["split_threshold"] = op_card.split_threshold.value()
                split_feature_thresholds.append(float(op_card.split_threshold.value()))
                split_ops[j] = split_op

        else:
            split_op = {}
            split_feature_index = int(self.feature_name_combo_box.value())
            split_op["split_feature_index"] = str(split_feature_index)
            split_feature_indexes.append(split_feature_index)
            split_op["split_op"] = 1
            split_op["split_threshold"] = 0
            split_feature_thresholds.append(0)
            split_ops[0] = split_op

        all_params["split_ops"] = split_ops


        logger.debug(f"Running analyzer with params {all_params}")
        result_features, features_array = Launcher.g.run(
            "analyzer", "label_splitter", **all_params
        )
        features_ndarray = np.array(features_array)
        print(f"Shape of features_array {features_ndarray.shape}")
        
        if features_array:
            logger.debug(f"Segmentation stats result table {len(features_array)}")                
            feature_arrays = []
            feature_titles = []
            
            for j,s in enumerate(split_feature_indexes):
                print(s)
                feature_title = feature_names[int(s)]
                print(feature_title)
                feature_plot_array = features_ndarray[:, int(s)] 
                feature_arrays.append(feature_plot_array)
                feature_titles.append(feature_title)
                print(f"Titles of feature names{feature_titles}")
                print(f"Split feature thresholds: {split_feature_thresholds}")
            self.display_splitter_plot(feature_arrays, titles=feature_titles, vert_line_at=split_feature_thresholds)
            self.display_splitter_results(result_features)

    def calc_level_image_stats(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        anno_id = DataModel.g.dataset_uri(self.annotations_source.value())
        all_params = dict(dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["anno_id"] = anno_id
        all_params["label_index"] = self.label_index.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "level_image_stats", **all_params)
        if result:
            logger.debug(f"Level Image stats result table {len(result)}")
            self.display_component_results(result)

    def calc_find_connected_components(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.pipelines_source.value(), group="pipelines")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["pipelines_id"] = str(self.pipelines_source.value())
        all_params["analyzers_id"] = str(self.analyzers_source.value())
        all_params["annotations_id"] = str(self.annotations_source.value())
        all_params["mode"] = self.radio_group.checkedId()
        all_params["label_index"] = self.label_index.value()
        all_params["area_min"] = self.area_min.value()
        all_params["area_max"] = self.area_max.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run(
            "analyzer", "find_connected_components", **all_params
        )
        print(result)
        if result:
            logger.debug(f"Segmentation stats result table {len(result)}")
            self.display_component_results(result)
    
    def calc_binary_image_stats(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["threshold"] = self.threshold.value()
        all_params["area_min"] = self.area_min.value()
        all_params["area_max"] = self.area_max.value()
        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "binary_image_stats", **all_params)
        if result:
            logger.debug(f"Segmentation stats result table {len(result)}")
            self.display_component_results(result)

    def calc_image_stats(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.features_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace

        all_params["feature_ids"] = str(self.features_source.value()[-1])

        logger.debug(f"Running analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "image_stats", **all_params)
        if result:
            src_arr = decode_numpy(result)
            sc = MplCanvas(self, width=5, height=4, dpi=100)
            sc.axes.imshow(src_arr)
            self.add_row(sc, max_height=300)

    def calc_object_analyzer(self):
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
        methods = ['TSNE', 'UMAP']
        all_params["embedding_method"] = str(methods[int(self.embedding_method_combo_box.value())])
        all_params["axis"] = int(self.axis_combo_box.value())
    
        embedding_params = {}
        if all_params["embedding_method"] == 'TSNE':
            embedding_params["n_components"] = int(self.TSNE_Panel.n_components.value())
            embedding_params["n_iter"] = int(self.TSNE_Panel.n_iter.value())
            embedding_params["perplexity"] = int(self.TSNE_Panel.perplexity.value())
        else:
            embedding_params["n_components"] = int(self.UMAP_Panel.n_components.value())
            embedding_params["n_neighbors"] = int(self.UMAP_Panel.n_neighbors.value())
            embedding_params["min_dist"] = float(self.UMAP_Panel.min_dist.value())
            embedding_params["spread"] = float(self.UMAP_Panel.spread.value())
            embedding_params["metric"] = str(umap_metrics[int(self.UMAP_Panel.metric_combo_box.value(key=True))])                
        all_params["embedding_params"] = embedding_params
        all_params["min_cluster_size"] = int(self.min_cluster_size.value())
        logger.debug(f"Running analyzer with params {all_params}")
        plot_img, labels, entities_arr = Launcher.g.run("analyzer", "object_analyzer", **all_params)
        entities_arr = np.array(entities_arr)
        entities_arr[:,3] = labels
        self.entities_arr = entities_arr
        
        if plot_img:
            src_arr = decode_numpy(plot_img)
            sc = MplCanvas(self, width=8, height=8, dpi=100)
            sc.axes.imshow(src_arr)
            
            sc.axes.margins(0)
            sc.axes.set_xticks([], [])
            sc.axes.set_yticks([], [])
            max_height = 800
            sc.setProperty("header", False)
            sc.setMaximumHeight(max_height)
            self.object_analyzer_plots.append(sc)
            self.vbox.addWidget(self.object_analyzer_plots[0])
            print("Added clustering plot")
            print(labels)
            self.display_clustering_results(labels)

    def calc_patch_stats(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.features_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_ids"] = str(self.features_source.value()[-1])
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


    def calc_object_detection_stats(self):  
        src = DataModel.g.dataset_uri(self.features_source.value())
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["predicted_objects_id"] = str(
            self.predicted_objects_source.value()
        )
        all_params["gold_objects_id"] = str(self.gold_objects_source.value())

        logger.debug(f"Running object detection analyzer with params {all_params}")
        result = Launcher.g.run("analyzer", "object_detection_stats", **all_params)
    def calc_binary_classifier(self):
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
     
    def calc_background_classifier(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["object_id"] = str(self.objects_source.value())
        all_params["proposal_id"] = str(self.feature_source.value())
        all_params["bg_mask_id"] = str(self.feature_source2.value())
        all_params["feature_id"] = str(self.feature_source3.value())
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

    def calc_point_generator(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["bg_mask_id"] = str(self.feature_source.value())
        all_params["num_before_masking"] = int(self.num_before_masking.value())
        result = Launcher.g.run("analyzer", self.analyzer_type, **all_params)
        logger.debug(f"point_generator result table {len(result)}")
        self.display_component_results3(result)

    def calc_spatial_clustering(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["workspace"] = DataModel.g.current_workspace
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["params"] = {"algorithm": "DBSCAN", "eps": self.eps.value(), "min_samples": 1}
        result = Launcher.g.run("analyzer", "spatial_clustering", **all_params)
        print(result)
        
        self.display_component_results2(result)

    def calc_removed_masked_objects(self):
        dst = DataModel.g.dataset_uri(self.analyzer_id, group="analyzer")
        src = DataModel.g.dataset_uri(self.feature_source.value())
        all_params = dict(src=src, dst=dst, modal=False)
        all_params["feature_id"] = str(self.feature_source.value())
        all_params["object_id"] = str(self.objects_source.value())
        all_params["invert"] = self.invert_checkbox.value()
        print(all_params)
        
        result = Launcher.g.run("analyzer", "remove_masked_objects", **all_params)
        logger.debug(f"remove_masked_objects result table {len(result)}")
        self.display_component_results(result)

    # calculate
    def calculate_analyzer(self):
         with progress(total=2) as pbar:
            pbar.set_description("Calculating pipeline")
            pbar.update(1)

            if self.analyzer_type == "label_splitter":
                self.calc_label_splitter()
            elif self.analyzer_type == "find_connected_components":
                self.calc_find_connected_components()    
            elif self.analyzer_type == "level_image_stats":
                self.calc_level_image_stats()
            elif self.analyzer_type == "binary_image_stats":
                self.calc_binary_image_stats()
            elif self.analyzer_type == "image_stats":
                self.calc_image_stats()
            elif self.analyzer_type == "object_analyzer":
                self.calc_object_analyzer()
            elif self.analyzer_type == "patch_stats":
                self.calc_patch_stats()
            elif self.analyzer_type == "object_detection_stats":
                self.calc_object_detection_stats()
            elif self.analyzer_type == "binary_classifier":
                self.calc_binary_classifier()
            elif self.analyzer_type == "spatial_clustering":
                self.calc_spatial_clustering()
            elif self.analyzer_type == "remove_masked_objects":
                self.calc_removed_masked_objects()
            elif self.analyzer_type == "point_generator":
                self.calc_point_generator()
            pbar.update(2)
