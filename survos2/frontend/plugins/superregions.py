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
    SWidget,
    clear_layout,
    QCSWidget,
    CheckBox,
    ComboBox,
    CardWithId,
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


@register_plugin
class RegionsPlugin(Plugin):

    __icon__ = "fa.qrcode"
    __pname__ = "superregions"
    __views__ = ["slice_viewer"]
    __tab__ = "superregions"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)

        self(
            IconButton("fa.plus", "Add SuperRegions", accent=True),
            connect=("clicked", self.add_supervoxel),
        )

        self.existing_supervoxels = {}
        self.supervoxel_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.supervoxel_layout)

    def add_supervoxel(self):
        params = dict(order=1, workspace=True, big=False)
        if result := Launcher.g.run("superregions", "create", **params):
            svid = result["id"]
            svname = result["name"]
            self._add_supervoxel_widget(svid, svname, True)

    def _add_supervoxel_widget(self, svid, svname, expand=False):
        widget = SupervoxelCard(svid, svname)
        widget.showContent(expand)
        self.supervoxel_layout.addWidget(widget)
        self.existing_supervoxels[svid] = widget
        return widget

    def clear(self):
        for region in list(self.existing_supervoxels.keys()):
            self.existing_supervoxels.pop(region).setParent(None)
        self.existing_supervoxels = {}

    def setup(self):
        params = dict(order=1, workspace=f"{DataModel.g.current_session}@{DataModel.g.current_workspace}")

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
                if params.pop("kind", "unknown") != "unknown":
                    widget = self._add_supervoxel_widget(svid, svname)
                    widget.update_params(params)
                    self.existing_supervoxels[svid] = widget
                else:
                    logger.debug(f"+ Skipping loading supervoxel: {svid}, {svname}")


class SupervoxelCard(Card):
    def __init__(self, svid, svname, parent=None):
        super().__init__(title=svname, collapsible=True, removable=True, editable=True, parent=parent)
        self.svid = svid
        self.svname = svname

        from survos2.frontend.plugins.features import FeatureComboBox

        self.svsource = FeatureComboBox()
        self.svsource.setMaximumWidth(250)
        self.svshape = LineEdit(parse=int, default=10)
        self.svshape.setMaximumWidth(250)
        self.svspacing = LineEdit3D(parse=float, default=1)
        self.svspacing.setMaximumWidth(250)
        self.svcompactness = LineEdit(parse=float, default=20)
        self.svcompactness.setMaximumWidth(250)

        self.int64_checkbox = CheckBox(checked=False)
        self.max_num_iter = LineEdit(parse=int, default=10)
        self.zero_parameter_checkbox = CheckBox(checked=False)
        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)

        self.add_row(HWidgets("Source:", self.svsource, stretch=1))
        self._add_feature_source()
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
        self.add_row(HWidgets(None, self.compute_btn, self.view_btn))

        self.compute_btn.clicked.connect(self.compute_supervoxels)
        self.view_btn.clicked.connect(self.view_regions)

    def _add_feature_source(self):
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
        with progress(total=4) as pbar:
            pbar.set_description("Refreshing")
            pbar.update(1)

            # src = [
            #    DataModel.g.dataset_uri("features/" + s) for s in [self.svsource.value()]
            # ]
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

            # n_segments = int(np.prod(chunk_size) // (self.svshape.value() ** 3))
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

    def update_params(self, params):
        if "shape" in params:
            self.svshape.setValue(params["shape"])
        if "compactness" in params:
            self.svcompactness.setValue(params["compactness"] * 100)
        if "spacing" in params:
            self.svspacing.setValue(params["spacing"])
        if "source" in params:
            for source in params["source"]:
                self.svsource.select(source)
