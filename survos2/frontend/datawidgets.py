import numpy as np
import pandas as pd
from numba import jit
import scipy
import yaml

from loguru import logger
from scipy import ndimage
from skimage import img_as_ubyte, img_as_float
from skimage import io


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


scfg = appState.scfg


###########################################################
# Entity widgets
#

class SmallVolWidget:
    def __init__(self, smallvol):

        self.imv = pg.ImageView()
        self.imv.setImage(smallvol, xvals=np.linspace(1., 3., smallvol.shape[0]))

    def set_vol(self, smallvol):
        self.imv.setImage(smallvol, xvals=np.linspace(1., 3., smallvol.shape[0]))
        self.imv.jumpFrames(smallvol.shape[0] // 2)



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
