import numpy as np
from loguru import logger
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Signal
from qtpy.QtWidgets import QPushButton
from vispy import scene

from survos2.frontend.components.base import *
from survos2.frontend.components.base import HWidgets, Slider
from survos2.frontend.plugins.annotations import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.base import ComboBox
from survos2.frontend.plugins.features import *
from survos2.frontend.plugins.superregions import *
from survos2.frontend.utils import FileWidget
from survos2.server.state import cfg
from survos2.model.model import DataModel
from survos2.config import Config
from napari.qt.progress import progress
from survos2.config import Config


class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.slice_mode = False
        self.slider = Slider(value=0, vmax=cfg.slice_max - 1)
        self.slider.setMinimumWidth(150)
        self.slider.sliderReleased.connect(self._params_updated)

        button_refresh = QPushButton("Refresh", self)
        button_refresh.clicked.connect(self.button_refresh_clicked)

        button_load_workspace = QPushButton("Load", self)
        button_load_workspace.clicked.connect(self.button_load_workspace_clicked)

        button_clear_client = QPushButton("Clear client")
        button_clear_client.clicked.connect(self.button_clear_client_clicked)

        button_transfer = QPushButton("Transfer Layer to Server")
        button_transfer.clicked.connect(self.button_transfer_clicked)

        # self.button_slicemode = QPushButton("Slice mode", self)
        # self.button_slicemode.clicked.connect(self.button_slicemode_clicked)

        workspaces = os.listdir(DataModel.g.CHROOT)
        self.workspaces_list = ComboBox()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)
        workspaces_widget = HWidgets("Switch Workspaces:", self.workspaces_list)
        self.workspaces_list.setEditable(True)
        self.workspaces_list.activated[str].connect(self.workspaces_selected)

        self.hbox_layout0 = QtWidgets.QHBoxLayout()
        hbox_layout_ws = QtWidgets.QHBoxLayout()
        hbox_layout1 = QtWidgets.QHBoxLayout()

        self.hbox_layout0.addWidget(self.slider)
        self.slider.hide()

        hbox_layout_ws.addWidget(workspaces_widget)
        hbox_layout_ws.addWidget(button_load_workspace)

        hbox_layout1.addWidget(button_transfer)
        # hbox_layout1.addWidget(self.button_slicemode)
        hbox_layout1.addWidget(button_refresh)
        hbox_layout1.addWidget(button_clear_client)

        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=2)

        self.tabwidget = QTabWidget()
        vbox.addWidget(self.tabwidget)

        self.tabs = [(QWidget(), t) for t in ["Workspace", "Histogram"]]

        # create tabs for button/info panel
        tabs = []
        for t in self.tabs:
            self.tabwidget.addTab(t[0], t[1].capitalize())
            t[0].layout = QVBoxLayout()
            t[0].setLayout(t[0].layout)
            tabs.append(t)

        self.plotbox = VBox(self, spacing=1)
        self.display_histogram_plot(np.array([0, 1]))

        # add tabs to button/info panel
        tabs[0][0].layout.addLayout(self.hbox_layout0)
        tabs[0][0].layout.addLayout(hbox_layout1)
        tabs[0][0].layout.addLayout(hbox_layout_ws)

        tabs[1][0].layout.addLayout(self.plotbox)

    def refresh_workspaces(self):
        workspaces = os.listdir(DataModel.g.CHROOT)
        self.workspaces_list.clear()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)
        self.slider.setMinimumWidth(cfg.base_dataset_shape[0])

    def workspaces_selected(self):
        selected_workspace = self.workspaces_list.value()
        self.workspaces_list.blockSignals(True)
        self.workspaces_list.select(selected_workspace)
        self.workspaces_list.blockSignals(False)

        cfg.ppw.clientEvent.emit(
            {
                "source": "panel_gui",
                "data": "set_workspace",
                "workspace": selected_workspace,
            }
        )

    def sessions_selected(self):
        cfg.ppw.clientEvent.emit(
            {
                "source": "panel_gui",
                "data": "set_session",
                "session": self.session_list.value(),
            }
        )

    def button_setroi_clicked(self):
        roi_start = self.roi_start.value()
        roi_end = self.roi_end.value()
        roi = [
            roi_start[0],
            roi_start[1],
            roi_start[2],
            roi_end[0],
            roi_end[1],
            roi_end[2],
        ]
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "make_roi_ws", "roi": roi})

    def button_load_workspace_clicked(self):
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "refresh", "value": None})

    def button_refresh_clicked(self):
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "faster_refresh", "value": None})

    def button_clear_client_clicked(self):
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})

    def button_pause_save_clicked(self):
        if cfg.pause_save:
            self.button_pause_save.setText("Pause Saving to Server")
        else:
            self.button_pause_save.setText("Resume saving to Server")

        cfg.pause_save = not cfg.pause_save

    def button_transfer_clicked(self):
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "transfer_layer", "value": None})

    def button_slicemode_clicked(self):
        if self.slice_mode:
            self.slider.hide()
            self.button_slicemode.setText("Slice Mode")
        else:
            logger.info(f"Slice mode {cfg.slice_max}")

            self.slider.vmax = cfg.slice_max - 1
            self.slider.setRange(0, cfg.slice_max - 1)
            self.slider.show()
            self.button_slicemode.setText("Volume Mode")
        self.slice_mode = not self.slice_mode

        cfg.ppw.clientEvent.emit(
            {
                "source": "button_slicemode",
                "data": "slice_mode",
            }
        )

    def _params_updated(self):
        cfg.ppw.clientEvent.emit(
            {"source": "slider", "data": "jump_to_slice", "frame": self.slider.value()}
        )

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            childWidget = child.widget()
            if childWidget:
                childWidget.setParent(None)
                childWidget.deleteLater()

    def display_histogram_plot(self, array, title="Plot", vert_line_at=None):
        array = array.ravel()
        self.clear_layout(self.plotbox)
        self.plot = MplCanvas(self, width=3, height=1, dpi=80)
        max_height = 180
        self.plot.setProperty("header", False)
        self.plot.setMaximumHeight(max_height)
        self.plot.axes.margins(0)
        self.plot.axes.axis("off")

        self.plotbox.addWidget(self.plot)
        self.setMinimumHeight(max_height)
        y, x, _ = self.plot.axes.hist(array, bins=256, color="gray", lw=0, ec="skyblue")
        self.plot.fig.set_facecolor((0.15, 0.15, 0.18))
        # self.plot.fig.set_alpha(0.5)
        # self.plot.axes.set_title(title)
        self.plot.axes.vlines(x=np.min(x), ymin=0, ymax=np.max(y), colors="w")
        self.plot.axes.vlines(x=np.max(x), ymin=0, ymax=np.max(y), colors="w")

        self.plotbox.addWidget(
            HWidgets(
                "Min: " + str(np.min(x)),
                "Max: " + str(np.max(x)),
                stretch=[0, 0],
            )
        )


class PluginPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.pluginContainer = PluginContainer()

        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=5)
        vbox.addWidget(self.pluginContainer)
        self.setLayout(vbox)

        for plugin_name in list_plugins():
            plugin = get_plugin(plugin_name)
            name = plugin["name"]
            title = plugin["title"]
            plugin_cls = plugin["cls"]  # full classname
            tab = plugin["tab"]

            logger.debug(f"Plugin loaded: {name}, {title}, {plugin_cls}")
            self.pluginContainer.load_plugin(name, title, plugin_cls)
            self.pluginContainer.show_plugin(name, tab)

        logger.debug(f"Plugins loaded: {list_plugins()}")
        self.cfg = cfg  # access in survos console: viewer.window._dock_widgets['Workspace'].children()[4].cfg

    def setup(self):
        with progress(total=len(list_plugins())) as pbar:
            pbar.set_description("Refreshing viewer")
            pbar.update(1)

            for plugin_name in list_plugins():
                pbar.update(1)
                plugin = get_plugin(plugin_name)
                name = plugin["name"]
                title = plugin["title"]
                plugin_cls = plugin["cls"]  # full classname
                tab = plugin["tab"]
                self.pluginContainer.show_plugin(name, tab)

        for l in cfg.viewer.layers:
            cfg.viewer.layers.remove(l)
        cfg.emptying_viewer = False

    def setup2(self):
        for plugin_name in list_plugins():
            plugin = get_plugin(plugin_name)
            name = plugin["name"]
            tab = plugin["tab"]
            self.pluginContainer.show_plugin2(name, tab)

    def setup_features(self):
        plugin = get_plugin("features")
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin2(name, tab)

    def setup_named_plugin(self, name):
        plugin = get_plugin(name)
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin(name, tab)

    def faster_setup_named_plugin(self, name):
        plugin = get_plugin(name)
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin2(name, tab)


class QtPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.canvas = scene.SceneCanvas(bgcolor="k", keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.canvas.native)

        _ = scene.visuals.Line(
            pos=np.array([[0, 0], [700, 500]]),
            color="w",
            parent=self.canvas.scene,
        )
