"""
An entity is Labeled geometric/vector data is stored in a dataframe

The most basic Entity Dataframe has 'z','x','y','class_code'


An Entity 
    Has a ROI
    Has Label(s)
    Has Optional Features
    Has Optional Measurements
        Simple measurement: Single "grade" for ROI
        Set of measurements:
    
Entity Collection
    (scene or complex object)
    DataFrame of Entities or MeasuredEntities
    Minimal: location and class (implied ROI based on class)
    Normal: location, roi, class

want to support running the clusterer
then making assignments
then using this as the new 'entities' in the gui

sampler.py contains functions that generate a table of entities from, e.g., a list of points

also
running the detector
using the detections as a new set of entities

anno.mask
supports converting entities into label volumes

anno.crowd
supports importing of data from zooniverse


"""

import itertools
import hdbscan
from collections import Counter
from statistics import mode, StatisticsError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

import time
import glob

import collections
import numpy as np
import pandas as pd
from typing import NamedTuple
import itertools


from scipy import ndimage
import torch.utils.data as data
from typing import List

import skimage
from skimage.morphology import thin
from skimage.io import imread, imread_collection
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from numpy.lib.stride_tricks import as_strided as ast
from numpy.random import permutation
from numpy import linalg

from survos2.frontend.nb_utils import summary_stats
from numpy.linalg import LinAlgError

#warnings.filterwarnings("ignore")
#warnings.filterwarnings(action='once')

from survos2.entity.anno.geom import centroid_3d, rescale_3d
from dataclasses import dataclass


def offset_points(pts, patch_pos):
    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]
    
    print(f"Offset: {offset_x}, {offset_y}, {offset_z}")
   
    z = pts[:,0].copy() - offset_z
    x = pts[:,1].copy() - offset_x
    y = pts[:,2].copy() - offset_y
    
    c = pts[:,3].copy() 
    
    
    offset_pts = np.stack([z,x,y, c], axis=1)
    
    return offset_pts
    

def make_entity_df(pts, flipxy=True):
    if flipxy:
        entities_df = pd.DataFrame({'z': pts[:, 0], 
                                'x': pts[:, 2],
                                'y': pts[:, 1],
                                'class_code' : pts[:,3]})
    else:
        entities_df = pd.DataFrame({'z': pts[:, 0], 
                                'x': pts[:, 1],
                                'y': pts[:, 2],
                                'class_code' : pts[:,3]})
    
    entities_df = entities_df.astype({'x': 'int32', 
                                  'y': 'int32', 
                                  'z':'int32', 
                                  'class_code': 'int32'})
    return entities_df




def make_entity_feats_df(pts, flipxy=True):
    if flipxy:
        entities_df = pd.DataFrame({'z': pts[:, 0], 
                                'x': pts[:, 2],
                                'y': pts[:, 1],
                                'class_code' : pts[:,3]})
    else:
        entities_df = pd.DataFrame({'z': pts[:, 0], 
                                'x': pts[:, 1],
                                'y': pts[:, 2],
                                'class_code' : pts[:,3]})
    
    entities_df = entities_df.astype({'x': 'int32', 
                                  'y': 'int32', 
                                  'z':'int32', 
                                  'class_code': 'int32'})
    return entity_feats_df




def make_entity_df2(pts):
    entities_df = pd.DataFrame({'z': pts[:, 0], 
                             'x': pts[:, 2],
                             'y': pts[:, 1],
                            'class_code' : pts[:,3]})


    entities_df = entities_df.astype({'x': 'float32', 
                                  'y': 'float32', 
                                  'z':'int32', 
                                  'class_code': 'int32'})
        
    return entities_df



