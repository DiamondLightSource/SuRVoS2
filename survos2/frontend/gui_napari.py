
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
        #self.supervoxel_features = supervoxel_features
        #self.supervoxel_rag = supervoxel_rag

        logger.debug(f"Shape of loaded supervoxels: {supervoxel_vol.shape}")

        # blank volume 
        if img.shape[0] == 1:
            img = np.zeros((30, 300, 300), dtype=int)
        self.img = img

        self.seg = np.zeros(self.img.shape, dtype=int)
        if anno.shape[0] == 1:
            anno = np.zeros(self.img.shape, dtype=int)

        self.anno = anno

        # seg ops
        #self.bind_key('Shift-C', self.commit)
        #self.bind_key('Shift-S', self.segment)
        #self.bind_key('Shift-R', self.refine)
        #self.bind_key('Shift-A', self.add_anno)

        # client filter ops
        self.bind_key('Shift-E', self.erode)
        self.bind_key('Shift-M', self.median)
        self.bind_key('Shift-D', self.dilate)

        # entity ops
        #self.bind_key('Shift-U', self.update_object_table)
        #self.bind_key('Shift-B', self.shape_to_labelim)
        #self.bind_key('Shift-O', self.anno_to_svg)
        #self.bind_key('Shift-T', self.generate_test_shapes)

        #
        # Attach 
        #

        #segSubject = SegSubject()
        #scfg.segSubject = segSubject
        #scfg.segSubject += partial(self.segment,scfg.widget) 


#    def update_object_table(self, viewer):
 #       logger.debug("Updating object table with dataframe of shape []")
  #      
   #     table_tuples = [(str(i), row[0]*scfg.mscale, 
    #                    row[1]*scfg.mscale, 
     #                   row[2]*scfg.mscale) for i,row in enumerate(scfg.cluster_rois)]
      #  
      #  tabledata = np.array(table_tuples, dtype=[('Object', object), ('z', float), ('x', float),('y', float)])
      #  
       # scfg.object_table.set_data(tabledata)
       # 
       # size = np.array([10, 10, 10])
       # viewer.add_points(scfg.cluster_rois, size=size)

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

    #def commit(self, viewer):
     #   logger.debug(f"Top layer is selected: {viewer.layers[-1].selected}")
      #  self.anno = viewer.layers[-1].data

    #def commit_filter(self, viewer):
     #   logger.debug(f"Top layer is selected: {viewer.layers[-1].selected}")
      #  self.img = viewer.layers[-1].data

    def median(self, viewer):
        median_size = scfg.marker_size
        logger.debug("Median, size {median_size}")
        img_proc = img_as_ubyte(ndimage.median_filter(simple_norm(self.img), size=median_size))
        self.add_image(img_proc, name='Median')

    #def add_anno(self, viewer):
     #   try:
      #      logger.debug("Add anno")
       #     #self.anno[anno_in == 0] += anno_in[anno_in == 0]
#
 #       except NameError as err:
  #          logger.debug("No input annotation provided: {}".format(err))

    def anno_to_svg(self, viewer):
        svg = viewer.to_svg()
        logger.debug(svg)

    #def generate_test_shapes(self, viewer):
     #   
      #  points3d = prepare_points3d(self.img.shape)
#
 #       logger.debug(self.img.shape, scfg.marker_size)
#
 #       max_points = np.min(scfg.marker_size, points3d.shape[0])
#
 #       for polypts in points3d[0:max_points]:
  #          viewer.add_shapes(
   #             polypts,
    #            shape_type='polygon',
     #           edge_width=2,
      #          edge_color='coral',
       ###         face_color='purple',
          #      opacity=0.75,
           # )
        # v#iewer.refresh()
"""
    def segment(self, viewer):
        logger.debug("Seg")

        try:


            prob_map, probs, pred_map, conf_map, P = sseg.process_anno_and_predict(self.features_stack, self.anno,
                                                            #viewer.layers['Active Annotation'].data,
                                                            self.supervoxel_vol, scfg.predict_params)
            

            #prob_map, probs, pred_map, conf_map = 
             
            self.P = probs

            logger.debug(f"Predicted a volume of shape: {prob_map.shape}")  #, Predicted features {P_dict}")
            
            self.prob_map = prob_map


            self.add_labels(self.prob_map, name='Prediction Labels')

            
            # post-process prediction (allows isosurfaces etc.)
            predicted = prob_map.copy()
            predicted -= np.min(prob_map)
            predicted += 1
            predicted = img_as_ubyte(predicted / np.max(predicted))
            predicted = img_as_ubyte(ndimage.median_filter(predicted, size=4))
            
            self.predicted = predicted
            self.add_image(self.predicted, name='Normalised Prediction ')
            
            self.add_image(pred_map, name='Prediction')
            #self.add_image(conf_map, name='Confidence')
            #print(f"Conf map: {conf_map.shape} {conf_map}")
       
        except Exception as err:
            logger.debug(f"Exception at segmentation prediction: {err}")

    def refine(self, viewer):

        logger.debug("Refine")

        try:
            Rp_ref = sseg.mrf_refinement(self.P,
                                    self.supervoxel_features,
                                    self.supervoxel_rag,
                                    self.features_stack)
            logger.debug(f"Refinement result shape: {Rp_ref.shape}")

            refined = img_as_ubyte(Rp_ref / np.max(Rp_ref))
            refined = img_as_ubyte(ndimage.median_filter(refined, size=3))

            self.add_image(refined, name='Refinement')

        except Exception as err:
            logger.error(f"Exception at refinement prediction: {err}")



"""