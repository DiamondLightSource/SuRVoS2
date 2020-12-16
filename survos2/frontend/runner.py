import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import h5py as h5
import numpy as np
import pyqtgraph.parametertree.parameterTypes as pTypes
import yaml
from loguru import logger
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QApplication, QDialog, QDialogButtonBox,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QRadioButton, QSlider,
                             QSpacerItem, QSpinBox, QTabWidget, QVBoxLayout,
                             QWidget)
from pyqtgraph.parametertree import (Parameter, ParameterItem, ParameterTree,
                                     registerParameterType)
from survos2.frontend.main import init_ws
from survos2.frontend.utils import ComboDialog, FileWidget, MplCanvas
from survos2.model.workspace import WorkspaceException

# example set of params for testing
ptree_init2 = [
    {
        "name": "Name",
        "type": "group",
        "children": [
            {
                "name": "datasets_dir",
                "type": "str",
                "value": "D:/datasets/survos_brain/ws3",
            },
            {"name": "PATCH_DIM", "type": "int", "value": 32},
            {"name": "class_names", "type": "list", "values": ["class1", "class2"]},
            {"name": "flipxy", "type": "bool", "value": False},
            {"name": "project_dir", "type": "str", "value": "projects/brain/"},
            {"name": "workspace", "type": "str", "value": "test_s2"},
            {"name": "dataset_name", "type": "str", "value": "data"},
            {"name": "entities_relpath", "type": "str", "value": "entities_brain.csv"},
            {"name": "patch_size", "type": "list", "value": [30, 300, 300]},
            {
                "name": "precrop_roi",
                "type": "list",
                "values": [0, 64, 100, 228, 100, 228],
            },
        ],
    },
]


class LoadDataDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Select Data to Load")
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setLayout(main_layout)
        container = QWidget(self)
        hbox = QHBoxLayout(container)
        container.setMaximumWidth(950)
        container.setMaximumHeight(530)
        container.setLayout(hbox)
        container.setObjectName("loaderContainer")
        container.setStyleSheet('QWidget#loaderContainer {'
                                '  background-color: #fefefe; '
                                '  border-radius: 10px;'
                                '}')
        lvbox = QVBoxLayout()
        rvbox = QVBoxLayout()
        lvbox.setAlignment(Qt.AlignTop)
        rvbox.setAlignment(Qt.AlignTop)
        hbox.addLayout(lvbox, 1)
        hbox.addLayout(rvbox, 1)

        main_layout.addWidget(container)

        lvbox.addWidget(QLabel('Preview Dataset'))


        self.slider = QSlider(1)
        lvbox.addWidget(self.slider)

        self.canvas = MplCanvas()
        lvbox.addWidget(self.canvas)

         # INPUT
        rvbox.addWidget(QLabel('Input Dataset:'))
        self.winput = FileWidget(extensions='*.h5 *.hdf5', save=False)
        rvbox.addWidget(self.winput)
        rvbox.addWidget(QLabel('Internal HDF5 data path:'))
        self.int_h5_pth = QLabel('None selected')
        rvbox.addWidget(self.int_h5_pth)
        rvbox.addWidget(QWidget(), 1)

        # Save | Cancel
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        rvbox.addWidget(self.buttonBox)
        self.winput.path_updated.connect(self.load_data)
        self.slider.sliderReleased.connect(self.update_image)

    def load_data(self, path):
        if path is not None and len(path) > 0:
            dataset = None
            if path.endswith('.h5') or path.endswith('.hdf5'):
                available_hdf5 = self.available_hdf5_datasets(path)
                selected, accepted = ComboDialog.getOption(available_hdf5, parent=self, title="HDF5: Select internal path")
                if accepted == QDialog.Rejected:
                    return
                dataset = selected
                self.int_h5_pth.setText(selected)

            self.data = self.volread(path=path)
            self.dataset = dataset
            self.update_image(load=True)
    
    def volread(self, path=None):
        _, file_extension = os.path.splitext(path)
        data = None
        logger.info('Loading file handle')
        if file_extension in ['.hdf5', '.h5']:
            data = h5.File(path, 'r')
        else:
            raise Exception('File format not supported')
        return data

    def scan_datasets_group(self, group, shape=None, dtype=None, path=''):
        datasets = []
        for name, ds in group.items():
            curr_path = '{}/{}'.format(path, name)
            if hasattr(ds, 'shape'):
                if len(ds.shape) == 3 \
                        and (shape is None or ds.shape == shape) \
                        and (dtype is None or ds.dtype == dtype):
                    datasets.append(curr_path)
            else:
                extra = self.scan_datasets_group(ds, shape=shape, path=curr_path)
                if len(extra) > 0:
                    datasets += extra
        return datasets


    def available_hdf5_datasets(self, path, shape=None, dtype=None):
        datasets = []
        with h5.File(path, 'r') as f:
            datasets = self.scan_datasets_group(f, shape=shape, dtype=dtype)
        return datasets

    @pyqtSlot()
    def update_image(self, load=False):
        slider = self.sender()
        idx = slider.value()
        if idx is None or load:
            data_shape = self.data[self.dataset].shape
            idx = data_shape[0]//2
            self.slider.blockSignals(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(data_shape[0] - 1)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.canvas.ax.set_ylim([data_shape[1] + 1, -1])
            self.canvas.ax.set_xlim([-1, data_shape[2] + 1])
           
        img = self.data[self.dataset][idx]
        self.canvas.ax.imshow(img, 'gray')
        self.canvas.ax.grid(False)
        self.canvas.redraw()


class ConfigEditor(QWidget):
    def __init__(self, run_config, workspace_config, pipeline_config, *args, **kwargs):
        super(ConfigEditor, self).__init__()

        self.run_config = run_config
        self.workspace_config = workspace_config
        self.pipeline_config = pipeline_config

        self.server_process = None
        self.client_process = None

        pipeline_config_ptree = self.init_ptree(self.pipeline_config, name="Pipeline")

        self.layout = QVBoxLayout()
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()

        tabwidget.addTab(tab1, "Setup and Start Survos")
        tabwidget.addTab(tab2, "Pipeline")

        self.create_workspace_button = QPushButton("Create workspace")
        run_button = QPushButton("Run SuRVoS")
        select_data_button = QPushButton("Select")
        advanced_button = QRadioButton("Advanced")
        advanced_button.toggled.connect(self.toggle_advanced)

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)

        workspace_fields = QGroupBox("Create Workspace:")
        wf_layout = QGridLayout()
        wf_layout.addWidget(QLabel("Data File Path:"), 0, 0)
        current_data_path = Path(self.workspace_config['datasets_dir'], self.workspace_config['vol_fname'])
        self.data_filepth_linedt = QLineEdit(str(current_data_path))
        wf_layout.addWidget(self.data_filepth_linedt, 1, 0, 1, 2)
        wf_layout.addWidget(select_data_button, 1, 2)
        wf_layout.addWidget(QLabel("HDF5 Internal Data Path:"), 2, 0, 1, 1)
        ws_dataset_name = self.workspace_config['dataset_name']
        internal_h5_path = ws_dataset_name if str(ws_dataset_name).startswith('/') else '/' + ws_dataset_name
        self.h5_intpth_linedt = QLineEdit(internal_h5_path)
        wf_layout.addWidget(self.h5_intpth_linedt, 2, 1, 1, 1)
        wf_layout.addWidget(QLabel("Workspace Name:"), 3, 0)
        self.ws_name_linedt_1 = QLineEdit(self.workspace_config['workspace_name'])
        wf_layout.addWidget(self.ws_name_linedt_1, 3, 1)
        wf_layout.addWidget(QLabel("Downsample Factor:"), 4, 0)
        self.downsample_spinner = QSpinBox()
        self.downsample_spinner.setRange(1, 10)
        self.downsample_spinner.setSpecialValueText("None")
        self.downsample_spinner.setMaximumWidth(60)
        self.downsample_spinner.setValue(int(self.workspace_config['downsample_by']))
        wf_layout.addWidget(self.downsample_spinner, 4, 1, 1, 1)
        wf_layout.addWidget(self.create_workspace_button, 5, 0, 1, 3)
        workspace_fields.setLayout(wf_layout)
        tab1.layout.addWidget(workspace_fields)

        self.adv_run_fields = QGroupBox("Advanced Run Settings:")
        adv_run_layout = QGridLayout()
        adv_run_layout.addWidget(QLabel("Server IP Address:"), 0, 0)
        self.server_ip_linedt = QLineEdit(self.run_config['server_ip'])
        adv_run_layout.addWidget(self.server_ip_linedt, 0, 1)
        adv_run_layout.addWidget(QLabel("Server Port:"), 1, 0)
        self.server_port_linedt = QLineEdit(self.run_config['server_port'])
        adv_run_layout.addWidget(self.server_port_linedt, 1, 1)
        self.adv_run_fields.setLayout(adv_run_layout)
        self.adv_run_fields.hide()

        run_fields = QGroupBox("Run SuRVoS:")
        run_layout = QGridLayout()
        run_layout.addWidget(QLabel("Workspace Name:"), 0, 0)
        self.ws_name_linedt_2 = QLineEdit(self.workspace_config['workspace_name'])
        run_layout.addWidget(self.ws_name_linedt_2, 0, 1)
        run_layout.addItem(QSpacerItem(80, 20), 0, 2)
        run_layout.addWidget(advanced_button, 1, 0)
        run_layout.addWidget(self.adv_run_fields, 2, 0)
        run_layout.addWidget(run_button, 3, 0, 1, 3)
        run_fields.setLayout(run_layout)
        tab1.layout.addWidget(run_fields)
        
        output_config_button = QPushButton("Save config")
        tab2.layout = QVBoxLayout()
        tab2.setLayout(tab2.layout)
        tab2.layout.addWidget(pipeline_config_ptree)
        tab2.layout.addWidget(output_config_button)

        run_button.clicked.connect(self.run_clicked)
        self.create_workspace_button.clicked.connect(self.create_workspace_clicked)
        output_config_button.clicked.connect(self.output_config_clicked)
        select_data_button.clicked.connect(self.launch_data_loader)

        self.layout.addWidget(tabwidget)

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("SuRVoS Settings Editor")
        current_fpth = os.path.dirname(os.path.abspath(__file__))
        self.setWindowIcon(QIcon(os.path.join(current_fpth, "resources", "logo.png")))
        self.setLayout(self.layout)
        self.show()

    @pyqtSlot()
    def launch_data_loader(self):
        path = None
        int_h5_pth = None
        dialog = LoadDataDialog(self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            path = dialog.winput.path.text()
            int_h5_pth = dialog.int_h5_pth.text()
        if path and int_h5_pth:
            self.data_filepth_linedt.setText(path)
            self.h5_intpth_linedt.setText(int_h5_pth)

    @pyqtSlot()
    def toggle_advanced(self):
        rbutton = self.sender()
        if rbutton.isChecked():
            self.adv_run_fields.show()
        else:
            self.adv_run_fields.hide()

    def setup_ptree_params(self, p, config_dict):
        def parameter_tree_change(param, changes):
            for param, change, data in changes:
                path = p.childPath(param)

                if path is not None:
                    childName = ".".join(path)
                else:
                    childName = param.name()

                sibs = param.parent().children()

                config_dict[path[-1]] = data

        p.sigTreeStateChanged.connect(parameter_tree_change)

        def valueChanging(param, value):
            pass

        for child in p.children():
            child.sigValueChanging.connect(valueChanging)

            for ch2 in child.children():
                ch2.sigValueChanging.connect(valueChanging)

        return p

    def dict_to_params(self, param_dict, name="Group"):
        ptree_param_dicts = []
        ctr = 0
        for key in param_dict.keys():
            entry = param_dict[key]

            if type(entry) == str:
                d = {"name": key, "type": "str", "value": entry}
            elif type(entry) == int:
                d = {"name": key, "type": "int", "value": entry}
            elif type(entry) == list:
                d = {"name": key, "type": "list", "values": entry}
            elif type(entry) == float:
                d = {"name": key, "type": "float", "value": entry}
            elif type(entry) == bool:
                d = {"name": key, "type": "bool", "value": entry}
            elif type(entry) == dict:
                d = self.dict_to_params(entry, name="Subgroup")[0]
                d["name"] = key + "_" + str(ctr)
                ctr += 1
            else:
                print(f"Can't handle type {type(entry)}")

            ptree_param_dicts.append(d)

        ptree_init = [{"name": name, "type": "group", "children": ptree_param_dicts}]

        return ptree_init

    def init_ptree(self, config_dict, name="Group"):
        ptree_init = self.dict_to_params(config_dict, name)
        parameters = Parameter.create(
            name="ptree_init", type="group", children=ptree_init
        )
        params = self.setup_ptree_params(parameters, config_dict)
        ptree = ParameterTree()
        ptree.setParameters(params, showTop=False)

        return ptree

    @pyqtSlot()
    def create_workspace_clicked(self):
        logger.debug("Creating workspace: ")
        # Set the path to the data file
        vol_path = Path(self.data_filepth_linedt.text())
        if not vol_path.is_file():
            err_str = f"No data file exists at {vol_path}!"
            logger.error(err_str)
            self.button_feedback_response(err_str, self.create_workspace_button, "maroon")
        else:
            self.workspace_config['datasets_dir'] = str(vol_path.parent)
            self.workspace_config['vol_fname'] = str(vol_path.name)
            dataset_name = self.h5_intpth_linedt.text()
            self.workspace_config['dataset_name'] = str(dataset_name).strip('/')
            # Set the workspace name
            ws_name = self.ws_name_linedt_1.text()
            self.workspace_config['workspace_name'] = ws_name
            # Set the downsample factor
            ds_factor = self.downsample_spinner.value()
            self.workspace_config['downsample_by'] = ds_factor
            try:
                response = init_ws(self.workspace_config)
                _, error = response
                if not error:
                    self.button_feedback_response("Workspace created sucessfully", self.create_workspace_button, "green")
                    # Update the workspace name in the 'Run' section
                    self.ws_name_linedt_2.setText(self.ws_name_linedt_1.text())
            except WorkspaceException as e:
                logger.exception(e)
                self.button_feedback_response(str(e), self.create_workspace_button, "maroon")

    def button_feedback_response(self, message, button, colour_str):
        msg_old = button.text()
        col_old = button.palette().button().color
        txt_col_old = button.palette().buttonText().color
        button.setText(message)
        button.setStyleSheet(f"background-color: {colour_str}; color: white")
        timer = QTimer()
        timer.singleShot(2000, lambda: self.reset_button(button, msg_old,
                                                         col_old, txt_col_old))
        
    @pyqtSlot()
    def reset_button(self, button, msg_old, col_old, txt_col_old):
        button.setStyleSheet(f"background-color: {col_old().name()}")
        button.setStyleSheet(f"color: {txt_col_old().name()}")
        button.setText(msg_old)
        button.update()

    @pyqtSlot()
    def output_config_clicked(self):
        out_fname = "pipeline_cfg.yml"
        logger.debug(f"Outputting pipeline config: {out_fname}")
        with open(out_fname, "w") as outfile:
            yaml.dump(
                self.pipeline_config, outfile, default_flow_style=False, sort_keys=False
            )

    @pyqtSlot()
    def run_clicked(self):
        command_dir = os.getcwd()
        script_fullname = os.path.join(command_dir, "survos.py")
        if not os.path.isfile(script_fullname):
            raise Exception("{}: Script not found".format(script_fullname))
        # Retrieve the parameters from the fields
        self.run_config['workspace_name'] = self.ws_name_linedt_2.text()
        self.run_config['server_ip'] = self.server_ip_linedt.text()
        self.run_config['server_port'] = self.server_port_linedt.text()

        self.server_process = subprocess.Popen(
            [
                "python",
                script_fullname,
                "start_server",
                self.run_config["workspace_name"],
                self.run_config["server_port"],
            ]
        )

        self.client_process = subprocess.Popen(
            [
                "python",
                script_fullname,
                "nu_gui",
                self.run_config["workspace_name"],
                str(self.run_config["server_ip"])
                + ":"
                + str(self.run_config["server_port"]),
            ]
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    run_config = {
        "server_ip": "127.0.0.1",
        "server_port": "8134",
        "workspace_name": "test_hunt_d4b",
    }

    workspace_config = {
        "dataset_name": "mydata",
        "datasets_dir": "C:\Work\Data",
        "vol_fname": "afile.h5",
        "workspace_name": "test_brain2",
        "downsample_by": "1",
    }

    from survos2.server.config import cfg

    pipeline_config = dict(cfg)

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    app = QApplication([])
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setOrganizationName("DLS")
    app.setApplicationName("SuRVoS2")
    config_editor = ConfigEditor(run_config, workspace_config, pipeline_config)
    app.exec_()

    if config_editor.server_process is not None:
        config_editor.server_process.kill()
