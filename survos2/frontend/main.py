"""

Init involves pointing survos to an image volume an a list of entities (which can be blank)

TODO: remove all the custom project stuff, leaving just init_ws and init_proj

"""
import os
import sys
import numpy as np
from loguru import logger
import argparse
import os
import h5py
import ast
import json
import time
import pandas as pd
import sys
import math

from matplotlib import patches, patheffects
from dataclasses import dataclass
from typing import List, Dict
from pprint import pprint
from pathlib import Path
from survos2.frontend.nb_utils import show_images
from attrdict import AttrDict

import skimage
from skimage import img_as_float, img_as_ubyte

from napari import layers

from survos2.frontend.frontend import frontend
from survos2.server.model import SRFeatures   #,  SegData
from survos2.server.features import prepare_prediction_features
from survos2.server.config import appState

from survos2.entity.entities import make_entity_df
from survos2 import survos
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.frontend.control import DataModel
from survos2.improc.utils import DatasetManager
from survos2.frontend.model import ClientData
from survos2.entity.sampler import crop_vol_and_pts_centered


def preprocess(img_volume):
    img_volume=np.array(img_volume).astype(np.float32)
    img_volume = np.nan_to_num(img_volume)

    img_volume = img_volume - np.min(img_volume)
    img_volume = img_volume / np.max(img_volume)
    return img_volume

def init_ws(project_file, ws_name):
    logger.info(f"Initialising workspace {ws_name} with image volume specified in project file {project_file}")
    
    
    with open(project_file) as project_file:    
            wparams = json.load(project_file)
            wparams = AttrDict(wparams)
    
    dataset_name = wparams['dataset_name']
    datasets_dir = wparams['datasets_dir']
    fname = wparams['vol_fname'] 

    logger.info(f"Loading h5 file {os.path.join(datasets_dir, fname)}")
    original_data = h5py.File(os.path.join(datasets_dir, fname), 'r')
    ds = original_data[dataset_name]
    img_volume = ds  #[dataset_name]
    logger.info(f"Loaded vol of size {img_volume.shape}")

    img_volume=preprocess(img_volume)
    tmpvol_fullpath = "out\\tmpvol.h5"

    with h5py.File(tmpvol_fullpath,  'w') as hf:
        hf.create_dataset("data",  data=img_volume)
    
    survos.run_command("workspace", "create", uri=None, workspace=ws_name)    
    logger.info(f"Created workspace {ws_name}")
    
    survos.run_command('workspace', 'add_data', uri=None, workspace=ws_name,
                data_fname=tmpvol_fullpath,
                dtype='float32')
    logger.info(f"Added data to workspace from {os.path.join(datasets_dir, fname)}")
    survos.run_command("workspace", "add_dataset", uri=None, workspace=ws_name, 
        dataset_name=dataset_name, dtype='float32')


def init_proj(wparams, precrop=False):    

    DataModel.g.current_workspace = wparams.workspace
    logger.debug(f"Set current workspace to {DataModel.g.current_workspace}")

    entities_df = pd.read_csv(os.path.join(wparams.project_dir, wparams.entities_relpath))
    entities_df.drop(entities_df.columns[entities_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    entities_df = make_entity_df(np.array(entities_df), flipxy=wparams.flipxy)

    logger.debug(f"Loaded entities {entities_df.shape}")

    survos.init_api()

    src = DataModel.g.dataset_uri('__data__')
    with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
    
    logger.debug(f"DatasetManager loaded volume of shape {img_volume.shape}")


    # view a ROI from a big volume by creating a temp dataset from a crop.
    if precrop:    
        precrop_coord = wparams.precrop_coord
        precrop_vol_size = wparams.precrop_vol_size
        logger.info(f"Preprocess cropping at {precrop_coord} to {precrop_vol_size}")

        img_volume, precropped_pts = crop_vol_and_pts_centered(img_volume,np.array(entities_df),
                                location=precrop_coord,
                                patch_size=precrop_vol_size,
                                debug_verbose=True,
                                offset=True)

        entities_df = make_entity_df(precropped_pts, flipxy=False)

        tmp_ws_name = DataModel.g.current_workspace + "_tmp" 
        
        result = survos.run_command("workspace", "get", uri=None, workspace=tmp_ws_name)

        if not type(result[0]) == dict:
            logger.debug("Creating temp workspace")
            survos.run_command("workspace", "create", uri=None, workspace=tmp_ws_name) 
        else:
            logger.debug("tmp exists, deleting and recreating")
            survos.run_command("workspace", "delete", uri=None, workspace=tmp_ws_name) 
            logger.debug("workspace deleted")
            survos.run_command("workspace", "create", uri=None, workspace=tmp_ws_name) 
            logger.debug("workspace recreated")

        import h5py
        tmpvol_fullpath = "out\\tmpvol.h5"
        with h5py.File(tmpvol_fullpath,  'w') as hf:
            hf.create_dataset("data",  data=img_volume)

        survos.run_command('workspace', 'add_data', uri=None, workspace=tmp_ws_name,
                    data_fname=tmpvol_fullpath,
                    dtype='float32')
        DataModel.g.current_workspace = tmp_ws_name    
    
    # Prepare clientData (what gets loaded into napari)
    # TODO: replacing all this with a pure workspace-and-api approach

    filtered_layers = [np.array(img_volume).astype(np.float32)]
    
    #vol_anno = np.ones_like(filtered_layers[0])
    #dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    #feats = SRFeatures(filtered_layers, dataset_feats, features_stack)
    #vol_supervoxels = np.ones_like(filtered_layers[0]) #superregions.supervoxel_vol.astype(np.uint32)    
    
    layer_names = ['Main',]

    opacities = [1.0,]

    clientData = ClientData(filtered_layers, layer_names, opacities, 
                            entities_df, wparams['class_names'])
    
    clientData.wparams = wparams


    return clientData


def setup_ws(project_file=None):    
    with open(project_file) as project_file:    
        wparams = json.load(project_file)
        wparams = AttrDict(wparams)
        clientData = init_proj(wparams)

    return clientData

def startup(project_file):
    clientData = setup_ws(project_file)
    viewer = frontend(clientData)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project_file', default='./projects/brain/ws_brain.json')
    parser.add_argument('-i', '--init_name', default=None)
    args = parser.parse_args()
    
    if not args.init_name:
        startup(args.project_file)
    else: 
        init_ws(args.project_file, args.init_name)


if __name__ == "__main__":
    main()


