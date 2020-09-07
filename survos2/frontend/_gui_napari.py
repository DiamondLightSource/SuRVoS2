
from scipy import ndimage
import numpy as np

from skimage import img_as_ubyte, img_as_float


#import qdarkstyle
import pyqtgraph.parametertree.parameterTypes as pTypes
from qtpy import QtWidgets
from qtpy.QtCore import QSize
from vispy import scene
from vispy.color import Colormap

from napari import Viewer as NapariViewer


#
# SuRVoS 2 imports
#

from survos2.improc import map_blocks
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import logger

from survos2.entity.anno.geom import prepare_points3d
from survos2.helpers import AttrDict, simple_norm
from survos2.frontend.model import ClientData

import survos2.api.workspace as ws



class NapariWidget(NapariViewer):
    def __init__(self, img=np.array([0]), anno=np.array([0]), supervoxel_vol=[], features=[], layer_names=[], opacities=[]):

        super(NapariWidget, self).__init__()

        self.features_stack = features.features_stack
        self.supervoxel_vol = supervoxel_vol
    
        logger.debug(f"Shape of loaded supervoxels: {supervoxel_vol.shape}")

        # blank volume 
        if img.shape[0] == 1:
            img = np.zeros((30, 300, 300), dtype=int)
        self.img = img

        self.seg = np.zeros(self.img.shape, dtype=int)
        if anno.shape[0] == 1:
            anno = np.zeros(self.img.shape, dtype=int)

        self.anno = anno

        self.bind_key('Shift-E', self.erode)
        self.bind_key('Shift-M', self.median)
        self.bind_key('Shift-D', self.dilate)


    def dilate(self, viewer):
        logger.debug("Erode")
        str_3D = ndimage.morphology.generate_binary_structure(3, 1)

        img_to_ndimage = self.img.copy()
        img_to_ndimage -= np.min(img_to_ndimage)
        img_to_ndimage = self.img / np.max(self.img)
        img_to_ndimage = img_as_float(img_to_ndimage)

        img_proc = ndimage.morphology.binary_dilation(img_as_ubyte(img_to_ndimage), str_3D)
        
        self.add_image(img_proc, name='Dilation')

    def erode(self, viewer):
        logger.debug("Erode")

        str_3D = ndimage.morphology.generate_binary_structure(3, 1)
        big_str_3D = ndimage.morphology.iterate_structure(str_3D, 1)

        img_proc = ndimage.morphology.binary_opening(simple_norm(self.img), big_str_3D)
        self.add_image(img_as_ubyte(img_proc), name='Eroded')

    
    def median(self, viewer, median_size=2):
        logger.debug("Median, size {median_size}")
        img_proc = img_as_ubyte(ndimage.median_filter(simple_norm(self.img), size=median_size))
        self.add_image(img_proc, name='Median')

    
    def anno_to_svg(self, viewer):
        svg = viewer.to_svg()
        logger.debug(svg)

    