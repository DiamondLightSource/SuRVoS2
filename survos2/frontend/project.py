import json
import argparse
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

app = QtGui.QApplication([])

from qtpy.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QPushButton, QWidget


class ParameterTreeWidget:
    def __init__(self, params):

        p = setup_parameter_tree(params)
        t = ParameterTree()
        t.setParameters(p, showTop=False)
        t.setWindowTitle('pyqtgraph example: Parameter Tree')

        w = QtGui.QWidget()
        layout = QtGui.QGridLayout()
        w.setLayout(layout)
        layout.addWidget(t, 1, 0, 1, 1)
        w.show()
        w.resize(800, 800)

        s = p.saveState()
        p.restoreState(s)

        self.w = w


def parameter_tree_change(param, changes):
    print("tree changes:")

    for param, change, data in changes:
        path = scfg.p.childPath(param)

        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()

        sibs = param.parent().children()

        logger.debug(f"Path: {path}")
        logger.debug(f"Parent: {param.parent}")
        logger.debug(f"Siblings: {sibs}")
        logger.debug(f"Value: {param.value}")

        if childName == "MainGroup.Integer":
            logger.debug(f"Setting value: {data}")
            scfg.marker_size = int(data)

        print('  parameter: %s' % childName)
        print('  change:    %s' % change)
        print('  data:      %s' % str(data))
        print('\n')


def setup_parameter_tree(p):
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

    p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
    p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)

    return p



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


ptree_init3 = [
    {'name': 'Name', 'type': 'group', 'children': [
        {'name': 'Some int', 'type': 'int', 'value':10},
        {'name': 'Some float', 'type': 'float', 'value': 10.5, 'step': 0.1},
        {'name': 'String of some kind', 'type': 'str', 'value': "Hi"},
        {'name': 'List', 'type': 'list', 'values': [1,2,3], 'value': 2},
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


def setup_ptree_params():
    with open('./projects/brain/ws_brain.json') as project_file:    
            wparams = json.load(project_file)

    ptree_param_dicts = []

    for key in wparams.keys():
        entry = wparams[key]
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


ptree_init = setup_ptree_params()

p = Parameter.create(name='ptree_init', type='group', children=ptree_init)

def change(param, changes):
    print("tree changes:")
    for param, change, data in changes:
        path = p.childPath(param)
        if path is not None:
            childName = '.'.join(path)
        else:
            childName = param.name()
        print('  parameter: %s'% childName)
        print('  change:    %s'% change)
        print('  data:      %s'% str(data))
        print('  ----------')
    
p.sigTreeStateChanged.connect(change)

def valueChanging(param, value):
    print("Value changing (not finalized): %s %s" % (param, value))
    
# Too lazy for recursion:
for child in p.children():
    child.sigValueChanging.connect(valueChanging)
    for ch2 in child.children():
        ch2.sigValueChanging.connect(valueChanging)
        


ptree = ParameterTree()
ptree.setParameters(p, showTop=False)
ptree.setWindowTitle('pyqtgraph example: Parameter Tree')


win = QtGui.QWidget()
layout = QVBoxLayout()
win.setLayout(layout)


tabwidget = QTabWidget()
tab1 = QWidget()
tab2 = QWidget()

tabwidget.addTab(tab1, "Project File")
tabwidget.addTab(tab2, "Run")

tab1.layout = QVBoxLayout()
tab1.setLayout(tab1.layout)
tab1.layout.addWidget(ptree)

button_run = QPushButton("Run")

tab2.layout = QVBoxLayout()
tab2.setLayout(tab2.layout)
tab2.layout.addWidget(button_run)

layout.addWidget(tabwidget)

win.show()
win.resize(800,800)


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_file', default='./projects/brain/ws_brain.json')
    args = parser.parse_args()
    
    with open(args.project_file) as project_file:    
            wparams = json.load(project_file)
            

if __name__ == "__main__":
    import sys
    init()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



