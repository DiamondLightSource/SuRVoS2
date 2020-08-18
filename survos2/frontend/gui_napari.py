
import glob
import os
import sys
import h5py
import ntpath
import scipy
import yaml
from scipy import ndimage
import matplotlib.pyplot as plt  # print(__doc__)
import numpy as np
import pandas as pd

from numba import jit
from collections import namedtuple
from skimage import img_as_ubyte, img_as_float
from skimage import io

from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.parametertree import Parameter, ParameterTree


#import qdarkstyle
import pyqtgraph.parametertree.parameterTypes as pTypes
from qtpy import QtWidgets
from qtpy.QtCore import QSize
from vispy import scene
from vispy.color import Colormap

import napari
from napari import Viewer as NapariViewer

from functools import partial

from immerframe import Proxy
from typing import Union, Any, List, Optional, cast
from typing import Callable, Iterator, Union, Optional, List
from typing import List, Set, Dict, Tuple, Optional
from typing import Mapping, MutableMapping, Sequence, Iterable, List, Set

#
# SuRVoS 2 imports
#

from survos2.improc import map_blocks
from survos2.improc.features import gaussian, tvdenoising3d
from survos2.improc.regions.rag import create_rag
from survos2.improc.regions.slic import slic3d
from survos2.improc.segmentation import _qpbo as qpbo
from survos2.improc.segmentation.appearance import train, predict, refine, invrmap
from survos2.improc.segmentation.mappings import rmeans
from survos2.io import dataset_from_uri
from survos2.model import Workspace, Dataset
from survos2.utils import decode_numpy, encode_numpy
from survos2.utils import logger

from survos2.entity.anno.geom import prepare_points3d
from survos2.helpers import AttrDict, simple_norm
from survos2.server.config import appState
scfg = appState.scfg

import survos2.server.workspace as ws
import survos2.server.segmentation as sseg

#from survos2.frontend.model import SegSubject


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

        # client filter ops
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

    
    def median(self, viewer):
        median_size = scfg.marker_size
        logger.debug("Median, size {median_size}")
        img_proc = img_as_ubyte(ndimage.median_filter(simple_norm(self.img), size=median_size))
        self.add_image(img_proc, name='Median')

    
    def anno_to_svg(self, viewer):
        svg = viewer.to_svg()
        logger.debug(svg)

    