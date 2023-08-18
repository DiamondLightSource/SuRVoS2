import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton
from loguru import logger
from survos2.frontend.components.base import (
    VBox,
    LazyComboBox,
    LineEdit3D,
    Label,
    HWidgets,
    PushButton,
    PluginNotifier,
    clear_layout,
    CheckBox,
    ComboBox,
    LineEdit,
    Card,
)
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import register_plugin, Plugin
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.frontend.plugins.features import FeatureComboBox
from survos2.model import DataModel
from survos2.server.state import cfg
from survos2.improc.utils import DatasetManager
from napari.qt.progress import progress

_SuperregionsNotifier = PluginNotifier()


class RegionComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)

        result = Launcher.g.run("superregions", "existing", **params)
        logger.debug(f"Result of regions existing: {result}")
        if result:
            self.addCategory("Supervoxels")
            for fid in result:
                if result[fid]["kind"] == "supervoxels":
                    self.addItem(fid, result[fid]["name"])
                if result[fid]["kind"] == "sam":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class RegionsPlugin(Plugin):
    __icon__ = "fa.qrcode"
    __pname__ = "superregions"
    __views__ = ["slice_viewer"]
    __tab__ = "superregions"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.vbox = VBox(self, spacing=10)

        self.regions_combo = ComboBox()
        self.vbox.addWidget(self.regions_combo)
        self.existing_supervoxels = {}
        self.supervoxel_layout = VBox(margin=0, spacing=5)
        self.regions_combo.currentIndexChanged.connect(self.add_supervoxel)
        self.vbox.addLayout(self.supervoxel_layout)
        self._populate_regions()

    def _populate_regions(self):
        self.regions_params = {}
        self.regions_combo.clear()
        self.regions_combo.addItem("Add regions")

        params = dict(workspace=DataModel.g.current_session + "@" + DataModel.g.current_workspace)
        result = Launcher.g.run("superregions", "available", **params)

        if result:
            all_categories = sorted(set(p["category"] for p in result))
            for i, category in enumerate(all_categories):
                self.regions_combo.addItem(category)
                self.regions_combo.model().item(i + len(self.regions_params) + 1).setEnabled(False)

                for f in [p for p in result if p["category"] == category]:
                    self.regions_params[f["name"]] = f["params"]
                    self.regions_combo.addItem(f["name"])

    def add_supervoxel_old(self):
        params = dict(order=1, workspace=True, big=False)
        if result := Launcher.g.run("superregions", "create", **params):
            svid = result["id"]
            svname = result["name"]
            self._add_supervoxel_widget(svid, svname, True)

    def add_supervoxel(self, idx):
        if idx <= 0:
            return
        if self.regions_combo.itemText(idx) == "":
            return

        logger.info(f"Adding region {self.regions_combo.itemText(idx)}")

        regions_type = self.regions_combo.itemText(idx)
        self.regions_combo.setCurrentIndex(0)

        from survos2.api.superregions import __region_names__

        order = __region_names__.index(regions_type)

        params = dict(order=order, workspace=True)
        result = Launcher.g.run("superregions", "create", **params)

        if result:
            svid = result["id"]
            svtype = result["kind"]
            svname = result["name"]
            self._add_supervoxel_widget(svid, svtype, svname, True)

    def _add_supervoxel_widget(self, svid, svtype, svname, expand=False):
        widget = SupervoxelCard(svid, svtype, svname)
        widget.showContent(expand)
        self.supervoxel_layout.addWidget(widget)
        self.existing_supervoxels[svid] = widget
        return widget

    def clear(self):
        for region in list(self.existing_supervoxels.keys()):
            self.existing_supervoxels.pop(region).setParent(None)
        self.existing_supervoxels = {}

    def setup(self):
        self._populate_regions()
        params = dict(
            order=1, workspace=f"{DataModel.g.current_session}@{DataModel.g.current_workspace}"
        )

        result = Launcher.g.run("superregions", "existing", **params)
        logger.debug(f"Region result {result}")
        if result:
            for region in list(self.existing_supervoxels.keys()):
                if region not in result:
                    self.existing_supervoxels.pop(region).setParent(None)
            for supervoxel in sorted(result):
                if supervoxel in self.existing_supervoxels:
                    continue
                params = result[supervoxel]
                svid = params.pop("id", supervoxel)
                svname = params.pop("name", supervoxel)
                svtype = params.pop("kind", supervoxel)
                print(svid, svname, svtype)
                if svtype != "unknown":
                    widget = self._add_supervoxel_widget(svid, svtype, svname)
                    widget.update_params(params)
                    self.existing_supervoxels[svid] = widget
                else:
                    logger.debug(f"+ Skipping loading supervoxel: {svid}, {svname}")


class SupervoxelCard(Card):
    def __init__(self, svid, svtype, svname, parent=None):
        super().__init__(
            title=svname, collapsible=True, removable=True, editable=True, parent=parent
        )
        self.svid = svid
        self.svname = svname
        self.svtype = svtype

        self.svsource = FeatureComboBox()
        self.svsource.setMaximumWidth(250)


        if self.svtype == "supervoxels":
            self.svshape = LineEdit(parse=int, default=10)
            self.svshape.setMaximumWidth(250)
            self.svspacing = LineEdit3D(parse=float, default=1)
            self.svspacing.setMaximumWidth(250)
            self.svcompactness = LineEdit(parse=float, default=20)
            self.svcompactness.setMaximumWidth(250)
            self.int64_checkbox = CheckBox(checked=False)
            self.max_num_iter = LineEdit(parse=int, default=10)
            self.zero_parameter_checkbox = CheckBox(checked=False)

            self.add_row(HWidgets("Source:", self.svsource, stretch=1))
            self._add_mask_feature_source()
            self.add_row(HWidgets("Shape:", self.svshape, "Spacing:", self.svspacing))
            self.add_row(HWidgets("Compactness:", self.svcompactness, stretch=1))
            self.add_row(
                HWidgets(
                    "Int64:",
                    self.int64_checkbox,
                    "Find parameters:",
                    self.zero_parameter_checkbox,
                    "Max Iter:",
                    self.max_num_iter,
                )
            )
        elif self.svtype == "sam":
            self.points_per_side = LineEdit(parse=int, default=32)
            self.pred_iou_thresh = LineEdit(parse=float, default=0.86)
            self.stability_score_thresh = LineEdit(parse=float, default=0.5)
            self.crop_n_layers = LineEdit(parse=int, default=1)
            self.crop_n_points_downscale_factor = LineEdit(parse=int, default=1)
            self.min_mask_region_area = LineEdit(parse=int, default=1000)
            self.MAX_NUM_LABELS_PER_SLICE = LineEdit(parse=int, default=1000)
            self.skip = LineEdit(parse=int, default=10)

            self.add_row(HWidgets("Source:", self.svsource, stretch=1))
            self.add_row(HWidgets("Slice skip:", self.skip))
            self.add_row(HWidgets("points_per_side:", self.points_per_side))
            self.add_row(
                HWidgets(
                    "pred_iou_thresh",
                    self.pred_iou_thresh,
                    "stability_score_thresh:",
                    self.stability_score_thresh,
                )
            )
            self.add_row(
                HWidgets(
                    "crop_n_layers:",
                    self.crop_n_layers,
                    "crop_n_points_downscale_factor",
                    self.crop_n_points_downscale_factor,
                )
            )
            self.add_row(HWidgets("min_mask_region_area:", self.min_mask_region_area))
            self.add_row(HWidgets("MAX_NUM_LABELS_PER_SLICE:", self.MAX_NUM_LABELS_PER_SLICE))

        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)
        self.add_row(HWidgets(None, self.compute_btn, self.view_btn))
        self.compute_btn.clicked.connect(self.compute_supervoxels)
        self.view_btn.clicked.connect(self.view_regions)

    def _add_mask_feature_source(self):
        self.feature_source = FeatureComboBox()
        self.feature_source.fill()
        self.feature_source.setMaximumWidth(250)

        widget = HWidgets("Mask:", self.feature_source, stretch=1)
        self.add_row(widget)

    def card_deleted(self):
        params = dict(region_id=self.svid, workspace=True)
        result = Launcher.g.run("superregions", "remove", **params)
        if result["done"]:
            self.setParent(None)

        cfg.ppw.clientEvent.emit(
            {
                "source": "superregions",
                "data": "remove_layer",
                "layer_name": self.svid,
            }
        )

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited region title {newtitle}")
        params = dict(region_id=self.svid, new_name=newtitle, workspace=True)
        result = Launcher.g.run("superregions", "rename", **params)
        if result["done"]:
            _SuperregionsNotifier.notify()

        return result["done"]

    def view_regions(self):
        logger.debug(f"Transferring supervoxels {self.svid} to viewer")
        with progress(total=2) as pbar:
            pbar.set_description("Viewing feature")
            pbar.update(1)
            print(f"Current Supervoxels: {cfg.current_supervoxels}")
            cfg.ppw.clientEvent.emit(
                {"source": "superregions", "data": "view_regions", "region_id": self.svid}
            )
            pbar.update(1)

    def compute_supervoxels(self):
        if self.svtype == "supervoxels":
            with progress(total=4) as pbar:
                pbar.set_description("Super-region computation:")
                pbar.update(1)

                src = DataModel.g.dataset_uri(self.svsource.value(), group="features")
                dst = DataModel.g.dataset_uri(self.svid, group="superregions")
                logger.debug(f"Compute sv: Src {src} Dst {dst}")

                from survos2.model import Workspace

                ws = Workspace(DataModel.g.current_workspace)
                num_chunks = np.prod(np.array(ws.metadata()["chunk_grid"]))
                chunk_size = ws.metadata()["chunk_size"]
                logger.debug(
                    f"Using chunk_size {chunk_size} to compute number of supervoxel segments for num_chunks: {num_chunks}."
                )

                with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
                    src_dataset_shape = DM.sources[0][:].shape

                pbar.update(1)
                n_segments = int(np.prod(src_dataset_shape) / self.svshape.value() ** 3)

                out_dtype = "uint64" if self.int64_checkbox.value() else "uint32"
                params = dict(
                    src=src,
                    dst=dst,
                    compactness=round(self.svcompactness.value() / 100, 3),
                    n_segments=n_segments,
                    spacing=self.svspacing.value(),
                    modal=False,
                    out_dtype=out_dtype,
                    max_num_iter=self.max_num_iter.value(),
                    zero_parameter=self.zero_parameter_checkbox.value(),
                    mask_id=str(self.feature_source.value()),
                )

                logger.debug(f"Compute supervoxels with params {params}")

                pbar.update(1)

                result = Launcher.g.run("superregions", "supervoxels", **params)
                if result is not None:
                    pbar.update(1)
        elif self.svtype == "sam":
            with progress(total=4) as pbar:
                pbar.set_description("Super-region computation:")
                pbar.update(1)

                src = DataModel.g.dataset_uri(self.svsource.value(), group="features")
                dst = DataModel.g.dataset_uri(self.svid, group="superregions")
                logger.debug(f"Compute sv: Src {src} Dst {dst}")
                params = dict(
                    src=src,
                    dst=dst,
                    modal=False,
                    points_per_side=self.points_per_side.value(),
                    pred_iou_thresh=self.pred_iou_thresh.value(),
                    stability_score_thresh=self.stability_score_thresh.value(),
                    crop_n_layers=self.crop_n_layers.value(),
                    crop_n_points_downscale_factor=self.crop_n_points_downscale_factor.value(),
                    min_mask_region_area=self.min_mask_region_area.value(),
                    MAX_NUM_LABELS_PER_SLICE=self.MAX_NUM_LABELS_PER_SLICE.value(),
                    skip=self.skip.value(),
                )
                logger.info(f"Computing segment-anything (sam) with params {params}")
                pbar.update(1)

                result = Launcher.g.run("superregions", "sam", **params)

                if result is not None:
                    pbar.update(1)

    def update_params(self, params):
        if "shape" in params:
            self.svshape.setValue(params["shape"])
        if "compactness" in params:
            self.svcompactness.setValue(params["compactness"] * 100)
        if "spacing" in params:
            self.svspacing.setValue(params["spacing"])
        if "src" in params:
            for source in params["source"]:
                self.svsource.select(source)
