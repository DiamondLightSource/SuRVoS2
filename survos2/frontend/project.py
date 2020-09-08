"""
Uses pyqtgraph parameter tree which displays params as formatted below
params = [
    {'name': 'Basic parameter data types', 'type': 'group', 'children': [
        {'name': 'Integer', 'type': 'int', 'value': 10},
        {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1},
        {'name': 'String', 'type': 'str', 'value': "hi"},
        {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
        {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3,3,3]}, 'value': 2},
        {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
        {'name': 'Color', 'type': 'color', 'value': "FF0", 'tip': "This is a color button"},
        {'name': 'Gradient', 'type': 'colormap'},
        {'name': 'Subgroup', 'type': 'group', 'children': [
            {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
            {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
        ]},
        {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
        {'name': 'Action Parameter', 'type': 'action'},
    ]}
    
]
"""


import json
import argparse
import numpy as np
import pyqtgraph as pg
import sys
import signal
import yaml

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import (
    Parameter,
    ParameterTree,
    ParameterItem,
    registerParameterType,
)
from qtpy.QtWidgets import (
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLineEdit,
    QLabel,
)

from loguru import logger

from survos2.frontend.main import init_ws


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


class ConfigEditor(QWidget):
    def __init__(self, run_config, workspace_config, pipeline_config, *args, **kwargs):
        super(ConfigEditor, self).__init__()

        self.run_config = run_config
        self.workspace_config = workspace_config
        self.pipeline_config = pipeline_config

        run_config_ptree = self.init_ptree(self.run_config, name="Run")
        workspace_config_ptree = self.init_ptree(
            self.workspace_config, name="Workspace"
        )
        pipeline_config_ptree = self.init_ptree(self.pipeline_config, name="Pipeline")

        self.layout = QVBoxLayout()
        tabwidget = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()

        tabwidget.addTab(tab1, "Start Survos")
        tabwidget.addTab(tab2, "Workspace")
        # tabwidget.addTab(tab3, "Pipeline")
        #
        create_workspace_button = QPushButton("Create workspace")
        run_button = QPushButton("Run button")

        tab1.layout = QVBoxLayout()
        tab1.setLayout(tab1.layout)
        tab1.layout.addWidget(run_config_ptree)
        tab1.layout.addWidget(run_button)

        tab2.layout = QVBoxLayout()
        tab2.setLayout(tab2.layout)
        tab2.layout.addWidget(workspace_config_ptree)
        tab2.layout.addWidget(create_workspace_button)

        output_config_button = QPushButton("Save config")
        tab3.layout = QVBoxLayout()
        tab3.setLayout(tab3.layout)
        tab3.layout.addWidget(pipeline_config_ptree)
        tab3.layout.addWidget(output_config_button)

        run_button.clicked.connect(self.run_clicked)
        output_config_button.clicked.connect(self.output_config_clicked)
        create_workspace_button.clicked.connect(self.create_workspace_clicked)

        self.layout.addWidget(tabwidget)

        self.setGeometry(300, 300, 450, 650)
        self.setWindowTitle("SuRVoS Settings Editor")
        self.setLayout(self.layout)
        self.show()

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
                # logger.debug(f"Parameter: {path}")
                # logger.debug(f"Value: {param.value}")
                # logger.debug(f"Data {data}")

        p.sigTreeStateChanged.connect(parameter_tree_change)

        def valueChanging(param, value):
            pass

        for child in p.children():
            child.sigValueChanging.connect(valueChanging)

            for ch2 in child.children():
                ch2.sigValueChanging.connect(valueChanging)

        return p

    def dict_to_params(self, param_dict, name="Group"):
        print(param_dict.keys())
        ptree_param_dicts = []
        ctr = 0
        for key in param_dict.keys():
            entry = param_dict[key]
            print(type(entry))
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
                print(d)
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

    def create_workspace_clicked(self, event):
        print(self.workspace_config)
        logger.debug("Creating workspace: ")
        init_ws(self.workspace_config)

    def output_config_clicked(self, event):
        print(self.pipeline_config)
        out_fname = "pipeline_cfg.yml"
        logger.debug(f"Outputting pipeline config: {out_fname}")
        with open(out_fname, "w") as outfile:
            yaml.dump(
                self.pipeline_config, outfile, default_flow_style=False, sort_keys=False
            )

    def run_clicked(self, event):
        print(self.run_config)
        import os

        command_dir = os.getcwd()
        script_fullname = os.path.join(command_dir, "survos.py")
        if not os.path.isfile(script_fullname):
            raise Exception("{}: Script not found".format(script_fullname))
        print(script_fullname)

        import subprocess

        subprocess.call(
            [
                "python",
                script_fullname,
                "nu_gui",
                self.run_config["workspace_name"],
                self.run_config["server_address"],
            ]
        )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    workspace_config = {
        "dataset_name": "data",
        "datasets_dir": "D:/datasets/",
        "vol_fname": "mcd_s10_Nuc_Cyt_r1.h5",
        "workspace_name": "test_hunt2",
    }

    run_config = {
        "server_address": "172.23.5.231:8123",
        "CHROOT": "/dls/science/groups/das/SuRVoS/s2/data/",
        "workspace_name": "test_hunt2",
    }

    from survos2.server.config import cfg

    pipeline_config = dict(cfg)

    app = QApplication([])
    config_editor = ConfigEditor(run_config, workspace_config, pipeline_config)
    app.exec_()
