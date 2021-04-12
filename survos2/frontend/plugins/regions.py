import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton, QRadioButton

from survos2.frontend.components.base import *
from survos2.frontend.components.icon_buttons import IconButton
from survos2.frontend.control import Launcher
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.plugins_components import MultiSourceComboBox
from survos2.model import DataModel
from survos2.server.state import cfg


class RegionComboBox(LazyComboBox):
    def __init__(self, full=False, header=(None, "None"), parent=None):
        self.full = full
        super().__init__(header=header, parent=parent)

    def fill(self):
        params = dict(workspace=True, full=self.full)

        # result = [{'kind': 'supervoxels'}, ]
        result = Launcher.g.run("regions", "existing", **params)
        logger.debug(f"Result of regions existing: {result}")
        if result:
            self.addCategory("Supervoxels")
            for fid in result:
                if result[fid]["kind"] == "supervoxels":
                    self.addItem(fid, result[fid]["name"])


@register_plugin
class RegionsPlugin(Plugin):

    __icon__ = "fa.qrcode"
    __pname__ = "regions"
    __views__ = ["slice_viewer"]
    __tab__ = "regions"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        vbox = VBox(self, spacing=10)

        self(
            IconButton("fa.plus", "Add SuperVoxel", accent=True),
            connect=("clicked", self.add_supervoxel),
        )

        self.existing_supervoxels = {}
        self.supervoxel_layout = VBox(margin=0, spacing=5)
        vbox.addLayout(self.supervoxel_layout)

    def add_supervoxel(self):
        params = dict(order=1, workspace=True)
        result = Launcher.g.run("regions", "create", **params)

        if result:
            svid = result["id"]
            svname = result["name"]
            self._add_supervoxel_widget(svid, svname, True)

    def _add_supervoxel_widget(self, svid, svname, expand=False):
        widget = SupervoxelCard(svid, svname)
        widget.showContent(expand)
        self.supervoxel_layout.addWidget(widget)
        self.existing_supervoxels[svid] = widget
        return widget

    def setup(self):
        params = dict(order=1, workspace=True)

        params["id"] = 0
        params["name"] = "sv1"
        params["kind"] = "supervoxels"
        result = {}
        result[0] = params

        result = Launcher.g.run("regions", "existing", **params)
        if result:
            # Remove regions that no longer exist in the server
            for region in list(self.existing_supervoxels.keys()):
                if region not in result:
                    self.existing_supervoxels.pop(region).setParent(None)

            # Populate with new region if any
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
                    logger.debug(
                        "+ Skipping loading supervoxel: {}, {}".format(svid, svname)
                    )


class SupervoxelCard(Card):
    def __init__(self, svid, svname, parent=None):
        super().__init__(
            title=svname, collapsible=True, removable=True, editable=True, parent=parent
        )
        self.svid = svid
        self.svname = svname

        from qtpy.QtWidgets import QProgressBar

        self.pbar = QProgressBar(self)
        self.add_row(self.pbar)

        from survos2.frontend.plugins.features import FeatureComboBox

        self.svsource = FeatureComboBox()
        self.svsource.setMaximumWidth(250)
        self.svshape = LineEdit(parse=int, default=10)
        self.svshape.setMaximumWidth(250)
        self.svspacing = LineEdit3D(parse=float, default=1)
        self.svspacing.setMaximumWidth(250)
        self.svcompactness = LineEdit(parse=float, default=20)
        self.svcompactness.setMaximumWidth(250)
        self.compute_btn = PushButton("Compute")
        self.view_btn = PushButton("View", accent=True)

        self.add_row(HWidgets("Source:", self.svsource, stretch=1))
        self.add_row(HWidgets("Shape:", self.svshape, stretch=1))
        self.add_row(HWidgets("Spacing:", self.svspacing, stretch=1))
        self.add_row(HWidgets("Compactness:", self.svcompactness, stretch=1))
        self.add_row(HWidgets(None, self.compute_btn))

        self.add_row(HWidgets(None, self.view_btn))

        self.compute_btn.clicked.connect(self.compute_supervoxels)
        self.view_btn.clicked.connect(self.view_regions)

    def card_deleted(self):
        params = dict(region_id=self.svid, workspace=True)
        result = Launcher.g.run("regions", "remove", **params)
        if result["done"]:
            self.setParent(None)

        cfg.ppw.clientEvent.emit(
            {"source": "regions", "data": "remove_layer", "layer_name": self.svid,}
        )

    def card_title_edited(self, newtitle):
        logger.debug(f"Edited region title {newtitle}")
        params = dict(region_id=self.svid, new_name=newtitle, workspace=True)
        result = Launcher.g.run("regions", "rename", **params)
        return result["done"]

    def view_regions(self):
        logger.debug(f"Transferring supervoxels {self.svid} to viewer")

        print(f"Current Supervoxels: {cfg.current_supervoxels}")
        cfg.ppw.clientEvent.emit(
            {"source": "regions", "data": "view_regions", "region_id": self.svid}
        )

    def compute_supervoxels(self):
        self.pbar.setValue(10)
        src = [
            DataModel.g.dataset_uri("features/" + s) for s in [self.svsource.value()]
        ]
        dst = DataModel.g.dataset_uri(self.svid, group="regions")
        logger.debug(f"Compute sv: Src {src} Dst {dst}")

        from survos2.model import Workspace

        ws = Workspace(DataModel.g.current_workspace)
        num_chunks = np.prod(np.array(ws.metadata()["chunk_grid"]))
        chunk_size = ws.metadata()["chunk_size"]
        logger.debug(
            f"Using chunk_size {chunk_size} to compute number of supervoxel segments for num_chunks: {num_chunks}."
        )

        n_segments = int(np.prod(chunk_size) // (self.svshape.value() ** 3))

        params = dict(
            src=src,
            dst=dst,
            compactness=round(self.svcompactness.value() / 100, 3),
            # shape=self.svshape.value(),
            n_segments=n_segments,
            spacing=self.svspacing.value(),
            modal=False,
        )
        logger.debug(f"Compute supervoxels with params {params}")

        self.pbar.setValue(20)

        result = Launcher.g.run("regions", "supervoxels", **params)
        if result is not None:
            self.pbar.setValue(100)

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
