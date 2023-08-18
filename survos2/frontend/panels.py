import os
import numpy as np
from loguru import logger
from qtpy import QtWidgets, QtCore
from qtpy.QtCore import QSize, Signal
import subprocess
from survos2.frontend.control import Launcher

from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtWidgets import QFileDialog, QGridLayout, QGroupBox, QLabel
from vispy import scene
from survos2.frontend.components.base import HWidgets, Slider
from survos2.frontend.plugins.base import list_plugins, get_plugin, PluginContainer, register_plugin
from survos2.frontend.components.base import (
    VBox,
    Label,
    HWidgets,
    PushButton,
    SWidget,
    clear_layout,
    QCSWidget,
    ComboBox,
)
from survos2.frontend.plugins.features import MplCanvas
from survos2.server.state import cfg
from survos2.model.model import DataModel
from napari.qt.progress import progress
from survos2.config import Config

CHROOT = Config["model.chroot"]


class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent = Signal(object)

    def __init__(self, *args, **kwargs):
        run_config = {
            "server_ip": "127.0.0.1",
            "server_port": "8034",
            "workspace_name": "blank",
        }
        self.run_config = run_config
        QtWidgets.QWidget.__init__(self, *args, **kwargs)

        button_refresh = QPushButton("Refresh", self)
        button_refresh.clicked.connect(self.button_refresh_clicked)

        button_load_workspace = QPushButton("Load", self)
        button_load_workspace.clicked.connect(self.button_load_workspace_clicked)

        button_clear_client = QPushButton("Clear layer list")
        button_clear_client.clicked.connect(self.button_clear_client_clicked)

        button_transfer = QPushButton("Transfer Layer to Workspace")
        button_transfer.clicked.connect(self.button_transfer_clicked)

        hbox_layout_chroot = QtWidgets.QHBoxLayout()
        chroot_fields = self.get_chroot_fields()
        hbox_layout_chroot.addWidget(chroot_fields)


        hbox_layout_workspace = QtWidgets.QHBoxLayout()        
        workspaces = os.listdir(DataModel.g.CHROOT)
        workspaces.sort()
        self.workspaces_list = ComboBox()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)
        workspaces_widget = HWidgets(self.workspaces_list)
        self.workspaces_list.setEditable(True)

        workspace_group = QGroupBox("Workspace:")
        hbox_layout_ws = QtWidgets.QHBoxLayout()
        hbox_layout_ws.addWidget(workspaces_widget)
        hbox_layout_ws.addWidget(button_load_workspace)
        workspace_group.setLayout(hbox_layout_ws)
        hbox_layout_workspace.addWidget(workspace_group)


        hbox_layout1 = QtWidgets.QHBoxLayout()
        hbox_layout1.addWidget(button_transfer)
        hbox_layout1.addWidget(button_refresh)
        hbox_layout1.addWidget(button_clear_client)

        vbox = VBox(self, margin=(1, 1, 1, 1), spacing=2)

        self.tabwidget = QTabWidget()
        vbox.addWidget(self.tabwidget)

        self.tabs = [(QWidget(), t) for t in ["Main", "Server", "Histogram"]]

        # create tabs for button/info panel
        tabs = []
        for t in self.tabs:
            self.tabwidget.addTab(t[0], t[1].capitalize())
            t[0].layout = QVBoxLayout()
            t[0].setLayout(t[0].layout)
            tabs.append(t)

        self.plotbox = VBox(self, spacing=1)
        self.serverbox = VBox(self, spacing=1)
        self.display_histogram_plot(np.array([0, 1]))

        self.setup_adv_run_fields()
        
        # add tabs to button/info panel
        tabs[0][0].layout.addLayout(hbox_layout_chroot)
        tabs[0][0].layout.addLayout(hbox_layout_workspace)
        tabs[0][0].layout.addLayout(hbox_layout1)
        tabs[1][0].layout.addLayout(self.get_run_layout())
        tabs[2][0].layout.addLayout(self.plotbox)
        
        self.selected_workspace = self.workspaces_list.value()
        self.workspaces_list.blockSignals(True)
        self.workspaces_list.select(self.selected_workspace)
        self.workspaces_list.blockSignals(False)
    
    def get_chroot_fields(self):
        chroot_fields = QGroupBox("Project Directory:")
        chroot_fields.setMaximumHeight(60)
        chroot_layout = QGridLayout()
        self.given_chroot_linedt = QLineEdit(CHROOT)
        select_chroot_path_btn = PushButton("Select Path")
        select_chroot_path_btn.clicked.connect(self.select_chroot_path)

        chroot_layout.addWidget(self.given_chroot_linedt, 1, 0, 1, 2)
        set_chroot_button = QPushButton("Set Workspaces Root")
        chroot_layout.addWidget(select_chroot_path_btn, 1, 2)
        chroot_fields.setLayout(chroot_layout)
        set_chroot_button.clicked.connect(self.set_chroot)
        return chroot_fields
    
    def set_chroot(self):
        CHROOT = self.given_chroot_linedt.text()
        Config.update({"model": {"chroot": CHROOT}})
        logger.debug(f"Setting CHROOT to {CHROOT}")
        DataModel.g.CHROOT = CHROOT
        self.refresh_workspaces()

    def select_chroot_path(self):
        full_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select project path",
            ".",
            options=QFileDialog.DontUseNativeDialog | QtWidgets.QFileDialog.ShowDirsOnly,
        )
        if isinstance(full_path, tuple):
            full_path = full_path[0]
        if full_path != "":
            self.given_chroot_linedt.setText(full_path)
            self.set_chroot()
            cfg.bpw.refresh_workspaces()

            # edit the settings file to store the chosen chroot path
            import pathlib

            import ruamel.yaml

            current_path = pathlib.Path(__file__).parent.resolve()
            yaml = ruamel.yaml.YAML()
            yaml.preserve_quotes = True

            settings_path = os.path.abspath(os.path.join(current_path, '..', '..', 'settings.yaml'))

            with open(settings_path) as f:
                settings = yaml.load(f)

            for entry in settings:
                if entry == "model":
                    settings["model"]["chroot"] = full_path

            with open(settings_path,"w") as f:
                yaml.dump(settings, f)

    def toggle_advanced(self):
        """Controls displaying/hiding the advanced run fields on radio button toggle."""
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_run_fields.show()
        else:
            self.adv_run_fields.hide()

    def setup_adv_run_fields(self):
        """Sets up the QGroupBox that displays the advanced optiona for starting SuRVoS2."""
        self.adv_run_fields = QGroupBox()
        adv_run_layout = QGridLayout()
        self.run_button = QPushButton("Start Server")
        self.stop_button = QPushButton("Stop Server")
        adv_run_layout.addWidget(self.run_button, 0, 0, 1, 2)
        adv_run_layout.addWidget(self.stop_button, 0, 2, 1, 2)
        self.run_button.clicked.connect(self.run_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)

        adv_run_layout.addWidget(QLabel("Server IP Address:"), 1, 0, 1, 2)
        self.server_ip_linedt = QLineEdit(self.run_config["server_ip"])
        adv_run_layout.addWidget(self.server_ip_linedt, 1, 2, 1, 2)
        adv_run_layout.addWidget(QLabel("Server Port:"), 2, 0, 1, 2)
        self.server_port_linedt = QLineEdit(self.run_config["server_port"])
        adv_run_layout.addWidget(self.server_port_linedt, 2, 2, 1, 2)

        self.existing_button = QPushButton("Use Existing Server")
        adv_run_layout.addWidget(self.existing_button, 3, 2, 1, 2)

        self.existing_button.clicked.connect(self.existing_clicked)
        self.adv_run_fields.setLayout(adv_run_layout)

    def stop_clicked(self):
        logger.info("Stopping server")
        if cfg["server_process"] is not None:
            cfg["server_process"].kill()

    def run_clicked(self):
        """Starts SuRVoS2 server"""
        self.workspaces_selected()
        command_dir = os.path.abspath(os.path.dirname(__file__))  # os.getcwd()

        # Set current dir to survos root
        from pathlib import Path

        command_dir = Path(command_dir).absolute().parent.resolve()
        os.chdir(command_dir)

        self.run_config["server_port"] = self.server_port_linedt.text()
        cfg.server_process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "start_server:app",
                "--port",
                self.run_config["server_port"],
            ]
        )
        try:
            outs, errs = cfg.server_process.communicate(timeout=10)
            print(f"OUTS: {outs, errs}")
        except subprocess.TimeoutExpired:
            pass

        logger.info(f"Setting remote: {self.server_port_linedt.text()}")
        remote_ip_port = "127.0.0.1:" + self.server_port_linedt.text()
        logger.info(f"Setting remote: {remote_ip_port}")
        resp = Launcher.g.set_remote(remote_ip_port)
        logger.info(f"Response from server to setting remote: {resp}")

        if hasattr(self, "selected_workspace"):
            logger.info(f"Setting workspace to: {self.selected_workspace}")
            resp = Launcher.g.run("workspace", "set_workspace", workspace=self.selected_workspace)
            logger.info(f"Response from server to setting workspace: {resp}")

            cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "refresh", "value": None})

    def existing_clicked(self):
        ssh_ip = self.server_ip_linedt.text()
        remote_ip_port = ssh_ip + ":" + self.server_port_linedt.text()
        logger.info(f"setting remote: {remote_ip_port}")
        resp = Launcher.g.set_remote(remote_ip_port)
        logger.info(f"Response from server to setting remote: {resp}")

        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "refresh", "value": None})

    def get_run_layout(self):
        """Gets the QGroupBox that contains the fields for starting SuRVoS.

        Returns:
            PyQt5.QWidgets.GroupBox: GroupBox with run fields.
        """

        run_layout = QGridLayout()
        run_layout.addWidget(self.adv_run_fields, 3, 0)
        
        return run_layout

    def refresh_workspaces(self):
        workspaces = os.listdir(DataModel.g.CHROOT)
        workspaces.sort()
        self.workspaces_list.clear()
        for s in workspaces:
            self.workspaces_list.addItem(key=s)

    def workspaces_selected(self):
        self.selected_workspace = self.workspaces_list.value()
        self.workspaces_list.blockSignals(True)
        self.workspaces_list.select(self.selected_workspace)
        self.workspaces_list.blockSignals(False)

        cfg.ppw.clientEvent.emit(
            {
                "source": "panel_gui",
                "data": "set_workspace",
                "workspace": self.selected_workspace,
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

    def startup_server(self):
        from survos2.frontend.nb_utils import start_server

        port = str(Config["api"]["port"])
        server_process = start_server(port)
        cfg["server_process"] = server_process
        remote_ip_port = "127.0.0.1:" + port
        logger.info(f"Setting remote: {remote_ip_port}")
        resp = Launcher.g.set_remote(remote_ip_port)

        if self.selected_workspace != "":
            workspace = self.selected_workspace
            logger.info(f"Setting workspace to: {workspace}")
            resp = Launcher.g.run("workspace", "set_workspace", workspace)
            logger.info(f"Response from server to setting workspace: {resp}")

    def button_load_workspace_clicked(self):
        self.workspaces_selected()

        with progress(total=2) as pbar:
            pbar.set_description("Refreshing viewer")
            pbar.update(1)
            cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})
            cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "refresh", "value": None})


            if "server_process" not in cfg:
                self.run_clicked()

            pbar.update(2)

            self.refresh_workspaces()

            search_str = self.selected_workspace
            index = self.workspaces_list.findText(search_str, QtCore.Qt.MatchFixedString)
            if index >= 0:
                self.workspaces_list.setCurrentText(search_str)
                logger.debug(f"Setting index of current workspaces to {index}")

            #cfg.ppw.switch_tab_index(1)
                
    def button_refresh_clicked(self):
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "faster_refresh", "value": None})

    def button_clear_client_clicked(self):
        self.refresh_workspaces()
        cfg.ppw.clientEvent.emit({"source": "panel_gui", "data": "empty_viewer", "value": None})

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
                tab = plugin["tab"]
                self.pluginContainer.show_plugin(name, tab)

        for l in cfg.viewer.layers:
            cfg.viewer.layers.remove(l)
        cfg.emptying_viewer = False

    def setup_fast(self):
        for plugin_name in list_plugins():
            plugin = get_plugin(plugin_name)
            name = plugin["name"]
            tab = plugin["tab"]
            self.pluginContainer.show_plugin_fast(name, tab)

    def setup_features(self):
        plugin = get_plugin("features")
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin_fast(name, tab)

    def setup_named_plugin(self, name):
        plugin = get_plugin(name)
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin(name, tab)

    def faster_setup_named_plugin(self, name):
        plugin = get_plugin(name)
        name = plugin["name"]
        tab = plugin["tab"]
        self.pluginContainer.show_plugin_fast(name, tab)

    def switch_tab_index(self, index):
        self.pluginContainer.tabwidget.setCurrentIndex(index)

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
