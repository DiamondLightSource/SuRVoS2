import numpy as np
import pandas as pd
from numba import jit
import scipy
import yaml
from vispy import scene
from vispy.color import Colormap

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
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget

from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from pyqtgraph.flowchart.library.common import CtrlNode
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from survos2.server.config import appState



import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from matplotlib import offsetbox
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


scfg = appState.scfg


####################################################################################
# PyQTGraph widgets
#


class DataTreeControl:
    """Display a dictionary
    
    Returns:
        widget -- returns a PyQTGraph widget
    """
    def __init__(self, d):

        self.d = d

        #callback
        def some_func1():
            return some_func2()

        def some_func2():
            try:
                print("some_func2")
                raise Exception()
            except:
                import sys
                return sys.exc_info()[2]

        d2 = {
            'a list': [1, 2, 3, 4, 5, 6, {'nested1': 'aaaaa', 'nested2': 'bbbbb'}, "seven"],
            'a dict': {
                'x': 1,
                'y': 2,
                'z': 'three'
            },
            'an array': np.random.randint(10, size=(30, 30)),
            'a traceback': some_func1(),
            'a function': some_func1,
            'a class': pg.DataTreeWidget,
        }

        tree = pg.DataTreeWidget(data=self.d)
        tree.show()
        tree.setWindowTitle('pyqtgraph example: DataTreeWidget')
        tree.resize(600, 600)
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'b')

        self.w = tree

    def update_d(self,d):
        self.d = d


class SmallVolWidget:
    def __init__(self, smallvol):

        # w = QtGui.QWidget()
        #layout = QtGui.QGridLayout()
        #w.setLayout(layout)
        #layout.addWidget(t, 1, 0, 1, 1)
        #w.show()
        #w.resize(400, 400)

        self.imv = pg.ImageView()
        self.imv.setImage(smallvol, xvals=np.linspace(1., 3., smallvol.shape[0]))

    def set_vol(self, smallvol):
        self.imv.setImage(smallvol, xvals=np.linspace(1., 3., smallvol.shape[0]))
        self.imv.jumpFrames(smallvol.shape[0] // 2)




###################################################################
# Clustering widgets
#

class Cluster2dWidget:
    def __init__(self, pts,colors, labels_str):
        num_classes = len(np.unique(colors))


        cmap = cm.jet
        #norm = Normalize(vmin=0, vmax=8)

        #palette = np.array(sns.color_palette("hls", num_classes))
        #print(colors)


       

        w = MatplotlibWidget()
        subplot = w.getFigure().add_subplot(111)
        colormap_name = 'plasma'
        
        cm.jet.get_cmap(colormap_name)
        #cm = subplot.get_cmap(colormap_name)
        palette = cm(np.linspace(0,1,num_classes))

        scat = subplot.scatter(pts[:,1], pts[:,0], lw=0, 
            c=palette[colors.astype(np.int)], s=[20] * len(pts))
        subplot.axis('off')
        subplot.axis('tight')
        
        subplot.legend(  [mpatches.Patch(color=palette[i], label='a') for i,p in enumerate(np.unique(colors))],
         labels_str, loc = 'lower left', labelspacing=0. )

        #subplot.legend(*scat.legend_elements(), loc="lower left", title="Classes")
        #subplot.legend()
        
#        scat = a.scatter(points[:,0], points[:,1], 
 #                        c=cluster_classes, cmap="jet_r")
        #subplot.legend(handles=scat.legend_elements()[0], labels=labels)
        subplot.invert_yaxis()
        w.draw()
        w.show()
        #w.resize(500, 500)
        self.w = w




class ImageGridWidget:

    def __init__(self, image_list, n_rows, n_cols, image_titles="", figsize=(20,20)):
    
        w = MatplotlibWidget()        
        axarr = w.getFigure().add_subplot(n_rows,n_cols, figsize=figsize)
        images = [image_list[i] for i in range(n_rows * n_cols)]

        #f, axarr = plt.subplots(n_rows,n_cols, figsize=figsize)
        for i in range(n_rows):
            for j in range(n_cols):
                axarr[i,j].imshow(images[i*n_cols + j])
                axarr[i,j].set_title(image_titles[(i*n_cols + j)])

        w.draw()
        w.show()
        #w.resize(500, 500)
        self.w = w


###########################################################
# Entity widgets
#

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

# have to inherit from QGraphicsObject in order for signal to work
class TableWidget(QtWidgets.QGraphicsObject):        
    clientEvent  = Signal(object)  

    def __init__(self):
        super().__init__()
        self.w = pg.TableWidget()

        self.w.show()
        self.w.resize(500, 500)
        self.w.setWindowTitle('Entity table')
    
        self.w.cellClicked.connect(self.cell_clicked)
        self.w.doubleClicked.connect(self.double_clicked)
    

        self.w.selected_row = 0

    def set_data(self, data):
        self.w.setData(data)        

    def double_clicked(self):
        #self.w.events.on_next({'source': 'button10', 'data':'show_roi', 'selected_roi':self.w.selected_row})
        self.clientEvent.emit({'source': 'table', 'data':'show_roi', 'selected_roi':self.w.selected_row})

        for index in self.w.selectedIndexes():
            print(self.w.model().data(index))

    def cell_clicked(self, row, col):
        print("Row %d and Column %d was clicked" % (row, col))
        self.w.selected_row = row
                    

    def cell_clicked2(self, row, col):
        print("Row %d and Column %d was clicked" % (row, col))
        item = self.w.itemAt(row, col)
        self.ID = item.text()
        print(item)
        print(item.text())
        #self.w.currentRow(), i).text()
        #self.w.selectionModel().selectedRows().
        
        selindex = self.w.selectedIndexes()
        
        print(selindex)
        index = selindex[0]
        print(self.w.model().data(index))

class TreeControl:

    """Tree of widgets
    """
    def __init__(self, scfg):
        
        self.scfg = scfg
        
        w = pg.TreeWidget()
        w.setColumnCount(2)
        w.show()
        w.setWindowTitle('')

        i1  = QtGui.QTreeWidgetItem(["Segmentation"])
        i2  = QtGui.QTreeWidgetItem(["Filter"])
        #i3  = pg.TreeWidgetItem(["Item 3"])
        #i21  = QtGui.QTreeWidgetItem(["Item 2.1"])
        
        w.addTopLevelItem(i1)
        w.addTopLevelItem(i2)
        #w.addTopLevelItem(i3)        
        #i21.addChild(i211)

        b1 = QtGui.QPushButton("Predict")
        w.setItemWidget(i1, 1, b1)
        
        b2 = QtGui.QPushButton("Median")
        w.setItemWidget(i2, 1, b2)

        b2.clicked.connect(self.button2Clicked)   
        b1.clicked.connect(self.button1Clicked)   

        self.w = w

    def button1Clicked(self):
        print("Send predict event")
        self.scfg.predictEvent.notify()

    def button2Clicked(self):
        print("Clicked button 2")

def generate_test_data():
    data = np.random.normal(size=(100, 100))
    data = 25 * pg.gaussianFilter(data, (5, 5))
    data += np.random.normal(size=(100, 100))
    data[40:60, 40:60] += 15.0
    data[30:50, 30:50] += 15.0
    # data += np.sin(np.linspace(0, 100, 1000))
    # data = metaarray.MetaArray(data, info=[{'name': 'Time', 'values': np.linspace(0, 1.0, len(data))}, {}])

    return data


# from pyqtgraph examples
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



