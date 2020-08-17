"""
Utilities for notebooks, e.g. for popping up napari to view different types of data.    

"""
import os
import argparse
import glob
import json
import math
import sys
import time
import ast
import numpy as np
from numpy import nonzero, zeros_like, zeros
from numpy.random import permutation
from napari import gui_qt
from napari import Viewer as NapariViewer
import napari

from IPython.display import Image
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from survos2.frontend.utils import quick_norm


def view_volume(imgvol, name=""):

    with napari.gui_qt():
    
        viewer = napari.Viewer()
        viewer.add_image(imgvol, name=name)
        
    return viewer
    
def view_volumes(imgvols, name=""):

    with napari.gui_qt():
        viewer = napari.Viewer()
        names = [str(i) for i in range(len(imgvols))]

        for i in range(len(imgvols)):
            viewer.add_image(imgvols[i], name=names[i])
            
    return viewer
 
def view_label(label_vol, name="Label"):
    with napari.gui_qt():
        
        viewer = napari.Viewer()
        
        viewer.add_labels(label_vol, name=name)


        return viewer

def view_labels(img_vols, label_vol, name=""):
    with napari.gui_qt():
        
        viewer = napari.Viewer()
        
        #names = [str(i) for i in range(len(img_vols))]
        #for i in range(len(img_vols)):
        #    viewer.add_image(quick_norm(img_vols[i]), name=names[i])
        #viewer.add_labels(label_vol, name="Label image")
        # add the labels
            
        label_layer = viewer.add_labels(
        label_vol,
        name='segmentation',
        #properties=label_properties,
        #color=color,
        )
        
        return viewer



def view_vols_labels(img_vols, label_vol, name=""):
    with napari.gui_qt():
        
        viewer = napari.Viewer()
        
        names = [str(i) for i in range(len(img_vols))]
            
        label_layer = viewer.add_labels(
        label_vol,
        name='Labels',
        #properties=label_properties,
        #color=color,
        )
        

        for i in range(len(img_vols)):
            viewer.add_image(quick_norm(img_vols[i]), name=names[i])
        

        return viewer

def view_vols_points(img_vols, entities, names=None, flipxy=True):

    with napari.gui_qt():
        
        viewer = napari.Viewer()
        
        if names is None:
            names = [str(i) for i in range(len(img_vols))]
        

        for i in range(len(img_vols)):
            viewer.add_image(img_vols[i], name=names[i])    



        sel_start, sel_end = 0, len(entities)
        
        if flipxy:
            centers = np.array([ [np.int(np.float(entities.iloc[i]['z'])), 
                                    np.int(np.float(entities.iloc[i]['y'])), 
                                    np.int(np.float(entities.iloc[i]['x']))] 
                                    for i in range(sel_start, sel_end)])

        else:
            centers = np.array([ [np.int(np.float(entities.iloc[i]['z'])), 
                        np.int(np.float(entities.iloc[i]['x'])), 
                        np.int(np.float(entities.iloc[i]['y']))] 
                        for i in range(sel_start, sel_end)])

        num_classes = len(np.unique(entities["class_code"])) + 5
        
        print(f"Number of entity classes {num_classes}")
        
        palette = np.array(sns.color_palette("hls", num_classes) )# num_classes))        
        
        #class_codes = [ entities['class_zode'].iloc[i] for i in range(sel_start, sel_end)]
        face_color_list = [palette[class_code] for class_code in entities["class_code"]]

        viewer.add_points(centers, size=[10] * len(centers), face_color=face_color_list, n_dimensional=True)



        return viewer





class NumpyEncoder(json.JSONEncoder):
    """
    usage:
    j=json.dumps(results,cls=NumpyEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def isstring(s):
    # if we use Python 3
    if (sys.version_info[0] >= 3):
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)

def summary_stats(arr):
    return np.max(arr), np.min(arr), np.mean(arr), np.median(arr), np.std(arr), arr.shape


def make_directories(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
                    

def show_images(images,titles=None, figsize=(12,12)):
    n_images = len(images)
    
    if titles is None: titles = [f'{im.shape} {str(im.dtype)}' for im in images]
    
    fig = plt.figure(figsize=figsize)
    
    for n, (image,title) in enumerate(zip(images,titles)):
        a = fig.add_subplot(1,n_images,n+1) 
    
        if image.ndim == 2: 
            plt.gray() 
    
        plt.imshow(image)
        a.set_title(title)
    
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    
    plt.show()
            