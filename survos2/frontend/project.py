import json
import argparse
import numpy as np
import pyqtgraph as pg
import sys
import signal

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QPushButton, QWidget

from loguru import logger
fmt = "{level} - {name} - {message}"
logger.remove() # remove default logger
logger.add(sys.stderr, level="DEBUG", format=fmt)  #minimal stderr logger


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
    ]},
    {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
        {'name': 'Save State', 'type': 'action'},
        {'name': 'Restore State', 'type': 'action', 'children': [
            {'name': 'Add missing items', 'type': 'bool', 'value': True},
            {'name': 'Remove extra items', 'type': 'bool', 'value': True},
        ]},
    ]},
]


ptree_init2 = [{'name': 'Name',
  'type': 'group',
  'children': [{'name': 'datasets_dir',
    'type': 'str',
    'value': 'D:/datasets/survos_brain/ws3'},
   {'name': 'PATCH_DIM', 'type': 'int', 'value': 32},
   {'name': 'class_names',
    'type': 'list',
    'values': ['soul', 'nonsoul', 'sense_of_humour', 'background']},
   {'name': 'flipxy', 'type': 'bool', 'value': False},
   {'name': 'project_dir', 'type': 'str', 'value': 'projects/brain/'},
   {'name': 'workspace', 'type': 'str', 'value': 'test_s2'},
   {'name': 'dataset_name', 'type': 'str', 'value': 'data'},
   {'name': 'entities_relpath', 'type': 'str', 'value': 'entities_brain.csv'},
   {'name': 'patch_size', 'type': 'list', 'value': [30, 300, 300]},
   {'name': 'precrop_roi',
    'type': 'list',
    'values': [0, 64, 100, 228, 100, 228]}]}]



def setup_ptree_params(p):
    def parameter_tree_change(param, changes):
        print("tree changes:")

        for param, change, data in changes:
            path = p.childPath(param)

            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()

            sibs = param.parent().children()

            logger.debug(f"Path: {path}")
            logger.debug(f"Parent: {param.parent}")
            logger.debug(f"Siblings: {sibs}")
            logger.debug(f"Value: {param.value}")

            logger.debug('  parameter: %s' % childName)
            logger.debug('  change:    %s' % change)
            logger.debug('  data:      %s' % str(data))


    p.sigTreeStateChanged.connect(parameter_tree_change)  # callback 1

    def valueChanging(param, value):
        print("Value: %s\n %s" % (param, value))

    for child in p.children():
        child.sigValueChanging.connect(valueChanging)  # callback 2
        
        for ch2 in child.children():
            ch2.sigValueChanging.connect(valueChanging)

    def save():
        global state
        state = p.saveState()

    def restore():
        global state
        add = p['Save/Restore functionality', 'Restore State', 'Add missing items']
        rem = p['Save/Restore functionality', 'Restore State', 'Remove extra items']
        p.restoreState(state, addChildren=add, removeChildren=rem)

    #p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
    #p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)

    return p


def dict_to_params(param_dict):

    ptree_param_dicts = []

    for key in param_dict.keys():
        entry = param_dict[key]
        print(type(entry))
        if type(entry) == str:
            d = {'name': key, 'type': 'str' , 'value' : entry}
        elif type(entry) == int:
            d = {'name': key, 'type': 'int' , 'value' : entry}
        elif type(entry) == list:
            d = {'name': key, 'type': 'list' , 'values' : entry}
        elif type(entry) == float:
            d = {'name': key, 'type': 'float' , 'value' : entry}
        elif type(entry) == bool:
            d = {'name': key, 'type': 'bool' , 'value' : entry}

        ptree_param_dicts.append(d)
        
    ptree_init = [
        {'name': 'Name', 'type': 'group', 'children': ptree_param_dicts}]

    return ptree_init



def init_ptree(wparams):
    ptree_init = dict_to_params(wparams)
    parameters = Parameter.create(name='ptree_init', type='group', children=ptree_init)
    params = setup_ptree_params(parameters)
    ptree = ParameterTree()
    ptree.setParameters(params, showTop=False)    
    print("Ptree created")
    return ptree


def init(layout):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_file', default='./projects/brain/ws_brain.json')
    args = parser.parse_args()
    
    with open(args.project_file) as project_file:    
        wparams = json.load(project_file)

    ptree = init_ptree(wparams)

    tabwidget = QTabWidget()
    tab1 = QWidget()
    tab2 = QWidget()

    tabwidget.addTab(tab1, "Project File")

    button_run = QPushButton("Run")

    tab1.layout = QVBoxLayout()
    tab1.setLayout(tab1.layout)
    tab1.layout.addWidget(ptree)
    tab1.layout.addWidget(button_run)


    layout.addWidget(tabwidget)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication([])
    window = QWidget()
    layout = QVBoxLayout()
    window.setLayout(layout)
    init(layout)
    window.show()
    app.exec_()
