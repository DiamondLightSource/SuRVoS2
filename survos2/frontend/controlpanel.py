import numpy as np
import pandas as pd
from numba import jit
import scipy
import yaml
from loguru import logger


from scipy import ndimage
from skimage import img_as_ubyte, img_as_float
from skimage import io
import seaborn as sns
#import geopandas as gpd

from qtpy import QtWidgets
from qtpy.QtWidgets import QRadioButton, QPushButton
from qtpy.QtCore import QSize, Signal

from vispy import scene
from vispy.color import Colormap


import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.opengl as gl


from survos2.helpers import prepare_3channel, simple_norm, norm1

from survos2.frontend.components import *
from survos2.frontend.plugins.base import *
from survos2.frontend.plugins.regions import *
from survos2.frontend.plugins.features import *
from survos2.frontend.plugins.annotations import *

##################################################################
# Main panel of widgets 
#

class ButtonPanelWidget(QtWidgets.QWidget):
    clientEvent  = Signal(object)
    
    def __init__(self, *args, **kwards):
        QtWidgets.QWidget.__init__(self, *args, **kwards)
    
        #button1 = QPushButton('Predict Superregions', self)
        #button1.clicked.connect(self.button1_clicked)
        #self._times_clicked_b1 = 0
        
        button2 = QPushButton('Run Pipeline', self)
        button2.clicked.connect(self.button2_clicked)
        self._times_clicked_b2 = 0

        #button3 = QPushButton('Calc Supervoxels', self)
        #button3.clicked.connect(self.button3_clicked)
        #self._compactness= 50

        #button4 = QPushButton('Calc Features', self)
        #button4.clicked.connect(self.button4_clicked)
        #self._sigma= 50

        #button5 = QPushButton('Predict Saliency', self)
        #button5.clicked.connect(self.button5_clicked)
        
        #button6 = QPushButton('Predict Classes', self)
        #button6.clicked.connect(self.button6_clicked)
        
        #button7 = QPushButton('Test launcher', self)
        #button7.clicked.connect(self.button7_clicked)
        
        button8 = QPushButton('Spatial cluster', self)
        button8.clicked.connect(self.button8_clicked)
        
        button10 = QPushButton('View ROI', self)
        button10.clicked.connect(self.button10_clicked)
        self._selected_entity_idx = 0
        

        check1 = QtGui.QCheckBox("Z", self)
        check2 = QtGui.QCheckBox("X", self)
        check3 = QtGui.QCheckBox("Y", self)
        
        check1.setText("Z")
        check2.setText("X")
        check3.setText("Y")

        self._check1_checked = False
        check1.clicked.connect(self.check1_checked)
 
        vbox_layout = QtWidgets.QVBoxLayout()
        vbox_layout.addStretch(1)
        hbox_layout = QtWidgets.QHBoxLayout()
        #hbox_layout2 = QtWidgets.QHBoxLayout()
        #hbox_layout3 = QtWidgets.QHBoxLayout()
        hbox_layout4 = QtWidgets.QHBoxLayout()
        hbox_layout5 = QtWidgets.QHBoxLayout()

        #hbox_layout.addWidget(button1)
        hbox_layout.addWidget(button2)
        
        #hbox_layout.addWidget(button3)
        #hbox_layout2.addWidget(button4)
        
        #hbox_layout3.addWidget(button5)
        #hbox_layout3.addWidget(button6)
        #hbox_layout3.addWidget(button7)
        
        hbox_layout4.addWidget(button8)
        hbox_layout4.addWidget(button10)

        vbox = VBox(self, margin=(1, 0, 0, 0), spacing=5)
        
        vbox.addLayout(hbox_layout)
        #vbox.addLayout(hbox_layout2)
        #vbox.addLayout(hbox_layout3)
        vbox.addLayout(hbox_layout4)
        vbox.addLayout(hbox_layout5)

        label_flip = QtWidgets.QLabel('Flip coords:')
        #label_flip.setText('Label Example')
        hbox_layout5.addWidget(label_flip)
        hbox_layout5.addWidget(check1, 0)
        hbox_layout5.addWidget(check2, 1)
        hbox_layout5.addWidget(check3, 2)
        
       
        
    def button1_clicked(self):
        self._times_clicked_b1 += 1
        #self.events.on_next({'source': 'button1', 'data':'predict', 'count':self._times_clicked_b1})
        self.clientEvent.emit({'source': 'button1', 'data':'predict', 'count':self._times_clicked_b1})
    def button2_clicked(self):
        self._times_clicked_b2 += 1
        self.clientEvent.emit({'source': 'button2', 'data':'pipeline', 'count':self._times_clicked_b2})

    def button3_clicked(self):
        self.clientEvent.emit({'source': 'button3', 'data':'calc_supervoxels', 'compactness':self._compactness})

    def button8_clicked(self):
        self.clientEvent.emit({'source': 'button8', 'data':'spatial_cluster'})

    def button10_clicked(self):
        self.clientEvent.emit({'source': 'button10', 'data':'show_roi', 'selected_roi':self._selected_entity_idx})

    def button11_clicked(self):
        self.clientEvent.emit({'source': 'button11', 'data':'flip_coords', 'some_param': 0})

    def check1_checked(self):
        self._check1_checked = not self._check1_checked
        self.clientEvent.emit({'source': 'checkbox', 'data':'flip_coords', 'axis':'z', 'value':self._check1_checked})
    



class PluginPanelWidget(QtWidgets.QWidget):
    clientEvent  = Signal(object)
    
    def __init__(self, *args, **kwards):
        QtWidgets.QWidget.__init__(self, *args, **kwards)
        
        self.pluginContainer = PluginContainer()

        vbox_layout = QtWidgets.QVBoxLayout()
        vbox_layout.addStretch(1)
        vbox = VBox(self, margin=(1, 0, 0, 0), spacing=5)
        vbox.addWidget(self.pluginContainer)
        self.setLayout(vbox)   

        for plugin_name in list_plugins():

            plugin = get_plugin(plugin_name)
            name = plugin['name']
            title = plugin['title']
            plugin_cls = plugin['cls']  #full classname

            logger.debug(f"Plugin loaded: {name}, {title}, {plugin_cls}")  # e.g. regions Regions <class 'ClientWidgets.RegionsPlugin'>
            
            self.pluginContainer.load_plugin(name, title, plugin_cls)
            self.pluginContainer.show_plugin(name)

        logger.debug(f"Plugins loaded: {list_plugins()}")



class QtPlotWidget(QtWidgets.QWidget):

    def __init__(self):

        super().__init__()

        self.canvas = scene.SceneCanvas(bgcolor='k', keys=None, vsync=True)
        self.canvas.native.setMinimumSize(QSize(300, 100))
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(self.canvas.native)

        _ = scene.visuals.Line(
            pos=np.array([[0, 0], [700, 500]]),
            color='w',
            parent=self.canvas.scene,
        )

