import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import h5py as h5
import pyqtgraph.parametertree.parameterTypes as pTypes
import yaml
from loguru import logger
from numpy import clip, product
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
        self.data_limits = None
        self.roi_changed =  False

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
        slider_vbox = QVBoxLayout()
        slider_hbox = QHBoxLayout()
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setContentsMargins(0, 0, 0, 0)
        slider_vbox.setSpacing(0)
        self.slider_min_label = QLabel(alignment=Qt.AlignLeft)
        self.slider_max_label = QLabel(alignment=Qt.AlignRight)
        slider_vbox.addWidget(self.slider)
        slider_vbox.addLayout(slider_hbox)
        slider_hbox.addWidget(self.slider_min_label, Qt.AlignLeft)
        slider_hbox.addWidget(self.slider_max_label, Qt.AlignRight)
        slider_vbox.addStretch()
        lvbox.addLayout(slider_vbox)

        self.canvas = MplCanvas()
        self.canvas.roi_updated.connect(self.on_roi_box_update)
        lvbox.addWidget(self.canvas)

         # INPUT
        rvbox.addWidget(QLabel('Input Dataset:'))
        self.winput = FileWidget(extensions='*.h5 *.hdf5', save=False)
        rvbox.addWidget(self.winput)
        rvbox.addWidget(QLabel('Internal HDF5 data path:'))
        self.int_h5_pth = QLabel('None selected')
        rvbox.addWidget(self.int_h5_pth)

        apply_roi_button = QPushButton("Apply ROI")
        reset_button = QPushButton("Reset ROI")
        roi_fields = QGroupBox("Select Region of Interest:")
        roi_layout = QGridLayout()
        roi_layout.addWidget(QLabel("Drag a box in the image window or type manually"), 0, 0, 1, 3)
        roi_layout.addWidget(QLabel("Axis"), 1, 0)
        roi_layout.addWidget(QLabel("Start Value:"), 1, 1)
        roi_layout.addWidget(QLabel("End Value:"), 1, 2)
        roi_layout.addWidget(apply_roi_button, 1, 3)
        roi_layout.addWidget(reset_button, 2, 3)
        roi_layout.addWidget(QLabel("x:"), 2, 0)
        self.xstart_linedt = QLineEdit("0")
        self.xstart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.xstart_linedt, 2, 1)
        self.xend_linedt = QLineEdit("0")
        self.xend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.xend_linedt, 2, 2)
        roi_layout.addWidget(QLabel("y:"), 3, 0)
        self.ystart_linedt = QLineEdit("0")
        self.ystart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.ystart_linedt, 3, 1)
        self.yend_linedt = QLineEdit("0")
        self.yend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.yend_linedt, 3, 2)
        roi_layout.addWidget(QLabel("z:"), 4, 0)
        self.zstart_linedt = QLineEdit("0")
        self.zstart_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.zstart_linedt, 4, 1)
        self.zend_linedt = QLineEdit("0")
        self.zend_linedt.textChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.zend_linedt, 4, 2)
        roi_layout.addWidget(QLabel("Downsample Factor:"), 5, 0)
        self.downsample_spinner = QSpinBox()
        self.downsample_spinner.setRange(1, 10)
        self.downsample_spinner.setSpecialValueText("None")
        self.downsample_spinner.setMaximumWidth(60)
        self.downsample_spinner.valueChanged.connect(self.on_roi_param_changed)
        roi_layout.addWidget(self.downsample_spinner, 5, 1)
        roi_layout.addWidget(QLabel("Estimated datasize (MB):"), 5, 3)
        self.data_size_label = QLabel("0")
        roi_layout.addWidget(self.data_size_label, 5, 4)
        roi_fields.setLayout(roi_layout)
        rvbox.addWidget(QWidget(), 1)
        rvbox.addWidget(roi_fields)

        apply_roi_button.clicked.connect(self.on_roi_apply_clicked)
        reset_button.clicked.connect(self.on_roi_reset_clicked)

        # Save | Cancel
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        rvbox.addWidget(self.buttonBox)
        self.winput.path_updated.connect(self.load_data)
        self.slider.sliderReleased.connect(self.update_image)

    @pyqtSlot()
    def on_slider_min_changed(self, value):
        self.slider_min_label.setText(value)

    @pyqtSlot()
    def on_roi_reset_clicked(self):
        self.data_limits = None
        self.reset_roi_fields()
        self.update_image(load=True)

    @pyqtSlot()
    def on_roi_apply_clicked(self):
        self.data_limits = self.get_roi_limits()
        self.roi_changed = self.check_if_roi_changed(self.data_limits)
        self.update_image()

    @pyqtSlot()
    def on_roi_param_changed(self):
        limits = self.get_roi_limits()
        x_start, x_end, y_start, y_end, z_start, z_end = self.clip_roi_box_vals(limits)
        x_size = x_end - x_start
        y_size = y_end - y_start
        z_size = z_end - z_start
        self.update_est_data_size(z_size, y_size, x_size)

    def get_roi_limits(self):
        x_start = self.get_linedt_value(self.xstart_linedt)
        x_end = self.get_linedt_value(self.xend_linedt)
        y_start = self.get_linedt_value(self.ystart_linedt)
        y_end = self.get_linedt_value(self.yend_linedt)
        z_start = self.get_linedt_value(self.zstart_linedt)
        z_end = self.get_linedt_value(self.zend_linedt)
        return x_start, x_end, y_start, y_end, z_start, z_end

    def get_linedt_value(self, linedt):
        if linedt.text():
            return int(linedt.text())
        return 0

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
            self.data_shape = self.data[self.dataset].shape
            self.reset_roi_fields()
            self.update_image(load=True)

    def reset_roi_fields(self):
        self.xstart_linedt.setText("0")
        self.xend_linedt.setText(str(self.data_shape[2]))
        self.ystart_linedt.setText("0")
        self.yend_linedt.setText(str(self.data_shape[1]))
        self.zstart_linedt.setText("0")
        self.zend_linedt.setText(str(self.data_shape[0]))
        self.roi_changed = False

    def check_if_roi_changed(self, roi_limits):
        x_start, x_end, y_start, y_end, z_start, z_end = roi_limits
        if not x_start == y_start == z_start == 0:
            return True
        if (x_end != self.data_shape[2]) or (y_end != self.data_shape[1]) or (z_end != self.data_shape[0]):
            return True
        return False

    def on_roi_box_update(self, size_tuple):
        # Append the z values
        z_start = int(self.zstart_linedt.text())
        z_end = int(self.zend_linedt.text())
        size_tuple += (z_start, z_end)
        # Clip the values
        x_start, x_end, y_start, y_end, z_start, z_end = self.clip_roi_box_vals(size_tuple)
        self.xstart_linedt.setText(str(x_start))
        self.xend_linedt.setText(str(x_end))
        self.ystart_linedt.setText(str(y_start))
        self.yend_linedt.setText(str(y_end))
        self.zstart_linedt.setText(str(z_start))
        self.zend_linedt.setText(str(z_end))

    def clip_roi_box_vals(self, vals):
        x_start, x_end, y_start, y_end, z_start, z_end = map(round, vals)
        x_start, x_end = clip([x_start, x_end], 0, self.data_shape[2])
        y_start, y_end = clip([y_start, y_end], 0, self.data_shape[1])
        z_start, z_end = clip([z_start, z_end], 0, self.data_shape[0])
        return x_start, x_end, y_start, y_end, z_start, z_end
    
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
        # Only update index if called by slider
        if isinstance(self.sender(), QSlider):
            idx = self.sender().value()
        else:
            idx = None
        # Set limits
        if self.data_limits:
            x_start, x_end, y_start, y_end, z_start, z_end = self.data_limits
            x_size = x_end - x_start
            y_size = y_end - y_start
            z_size = z_end - z_start
        else:
            z_size, y_size, x_size = self.data_shape
            x_start, x_end, y_start, y_end, z_start, z_end = (0, x_size, 0, y_size, 0, z_size)
        # Show central slice if loading data or changing roi
        if idx is None or load:
            idx = z_size//2
            self.slider.blockSignals(True)
            self.slider.setMinimum(z_start)
            self.slider_min_label.setNum(z_start)
            self.slider.setMaximum(z_end - 1)
            self.slider_max_label.setNum(z_end)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.canvas.ax.set_ylim([y_size + 1, -1])
            self.canvas.ax.set_xlim([-1, x_size + 1])

        #self.update_est_data_size(z_size, y_size, x_size)  
        img = self.data[self.dataset][idx]
        self.canvas.ax.imshow(img[y_start:y_end, x_start:x_end], 'gray')
        self.canvas.ax.grid(False)
        self.canvas.redraw()

    def update_est_data_size(self, z_size, y_size, x_size):
        data_size_tup = tuple(map(int, (z_size, y_size, x_size)))
        est_data_size = (product(data_size_tup) * 4) / 10 **6
        est_data_size /= self.downsample_spinner.value()
        self.data_size_label.setText(f"{est_data_size:.2f}")


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
        # ROI
        self.roi_fields = QGroupBox("ROI:")
        roi_fields_layout = QHBoxLayout()
        # z
        roi_fields_layout.addWidget(QLabel("z:"), 0)
        self.zstart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.zstart_roi_val, 1)
        roi_fields_layout.addWidget( QLabel("-"), 2)
        self.zend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.zend_roi_val, 3)
        # y
        roi_fields_layout.addWidget(QLabel("y:"), 4)
        self.ystart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.ystart_roi_val, 5)
        roi_fields_layout.addWidget( QLabel("-"), 6)
        self.yend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.yend_roi_val, 7)
        # x
        roi_fields_layout.addWidget(QLabel("x:"), 8)
        self.xstart_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.xstart_roi_val, 9)
        roi_fields_layout.addWidget( QLabel("-"), 10)
        self.xend_roi_val = QLabel("0")
        roi_fields_layout.addWidget(self.xend_roi_val, 11)
        
        self.roi_fields.setLayout(roi_fields_layout)
        wf_layout.addWidget(self.roi_fields, 4, 2, 1, 2)
        self.roi_fields.hide()
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
            down_factor = dialog.downsample_spinner.value()
        if path and int_h5_pth:
            self.data_filepth_linedt.setText(path)
            self.h5_intpth_linedt.setText(int_h5_pth)
            self.downsample_spinner.setValue(down_factor)
            if dialog.roi_changed:
                self.roi_fields.show()
            else:
                self.roi_fields.hide()

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
        try:
            outs, errs = self.server_process.communicate(timeout=10)
            print(f"OUTS: {outs, errs}")
        except subprocess.TimeoutExpired:
            pass

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
