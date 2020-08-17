"""

Init involves pointing survos to an image volume an a list of entities (which can be blank)



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
from UtilityCluster import show_images
from attrdict import AttrDict
import h5py
import skimage
from skimage import img_as_float, img_as_ubyte

from survos2.frontend.frontend import ClientData
from survos2.frontend.frontend import frontend
from survos2.server.model import Features, SegData
from survos2.server.filtering import prepare_prediction_features
from survos2.server.config import appState

from survos2.entity.entities import make_entity_df

import argparse

import numpy as np
from survos2 import survos
from napari import layers
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.frontend.control import DataModel
from survos2.improc.utils import DatasetManager


def clientData_to_workspace(clientData):
    """
    Make a new workspace 
    add a dataset with the default session
    add data to the dataset

    cleanup and close workspace
    """

    pass


    

def setup_clientData2(workspace):
    img_vol = img_as_float(workspace.get_data()[...])  
    vol_anno = workspace.get_dataset('annotations/level1')[...]
    vol_anno = np.zeros_like(vol_anno)
    vol_anno = vol_anno.astype(np.uint32)        
    vol_supervoxels = workspace.get_dataset('supervoxels/s1')[...]
    vol_supervoxels = vol_supervoxels.astype(np.uint32)

    return clientData


def setup_client_data(workspace, segdata):
    img_vol = img_as_float(workspace.get_data()[...])  
    vol_anno = workspace.get_dataset('annotations/level1')[...]
    vol_anno = np.zeros_like(vol_anno)
    vol_anno = vol_anno.astype(np.uint32)
        
    vol_supervoxels = workspace.get_dataset('supervoxels/s1')[...]
    vol_supervoxels = vol_supervoxels.astype(np.uint32)

    logger.debug(f"Supervox vol shape:{vol_supervoxels.shape}")
    filtered_layers = []

    num_feature_layers = workspace.get_dataset('features/f0').get_metadata()['num_features']

    for i in range(num_feature_layers):
        feature_layer_name = 'features/f' + str(i)
        filtered_layers.append(img_as_float(workspace.get_dataset(feature_layer_name)[...]))    
        
    logger.debug(f"Available datasets {workspace.available_datasets()}")
    logger.debug(f"{num_feature_layers}")
    
    img_vol -= np.min(img_vol)
    img_vol = img_vol / np.max(img_vol)
    
    #anno_in = np.zeros_like(anno_in)  #blank anno
    vol_stack = [img_vol,]
    vol_stack.extend(filtered_layers)
    layer_names = ['feature_' + str(i) for i, l in enumerate(vol_stack)]

    vol_stack.extend([vol_anno,vol_supervoxels])
    layer_names.extend(['Previous Annotation', 'Supervoxels'])
    opacities = [1.0 * len(layer_names)]

    logger.debug(f"Anno in: {vol_anno.shape}")
    logger.debug(f"num_vol_stack: {len(vol_stack)}, num_layer_names: {len(layer_names)}")

    clientData = ClientData(vol_stack, vol_anno, vol_supervoxels, segdata.feats, layer_names, opacities)

    return clientData

def init_old(scfg):
    survos_workspace, survos_segdata = stest.prepare_workspace(scfg)
    logger.info(f"Workspace: {survos_workspace}")
    logger.info(f"Filtered stack shape: {survos_segdata.feats.features_stack.shape}")
    return setup_client_data(survos_workspace, survos_segdata)


import os
def init_hunt(wparams):  
    data = h5py.File(os.path.join(wparams.datasets_dir,"mcd_s10_Nuc_Cyt_r1.h5"), 'r')#this is image slices file
    
    entities_df = pd.read_csv(wparams.entities_relpath)
    entities_df.drop(entities_df.columns[entities_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    entities_df = make_entity_df(np.array(entities_df), flipxy=True)
    print(f"Loaded entities {entities_df.shape}")
    img_volume = data[wparams['dataset_name']]#this is orginal data

    from survos2.entity.sampler import crop_vol_and_pts
    
    main_coords = [(32,800,800),(16,450,450),(32,700,700),(32,850,550),(32,500,500),(32,1450,1450),]
    precropped_vol_size = 80,224,224
    overall_offset = main_coords[1]

    img_volume, precropped_pts = crop_vol_and_pts(img_volume,np.array(entities_df),
                            location=overall_offset,
                            patch_size=precropped_vol_size,
                            debug_verbose=True,
                            offset=True)
    
    entities_df = make_entity_df(precropped_pts, flipxy=True)


    print(f"Loaded volume of shape {img_volume.shape}")
    scale_alpha = 0.1
    PATCH_DIM = 32
    gen_big = False
    patch_size=(20,224,224)
    wide_patch_pos = (80,1000,1000)

    vol_shape_x = img_volume[0].shape[0]
    vol_shape_y = img_volume[0].shape[1]
    vol_shape_z = len(img_volume)

    scfg = appState.scfg
    scfg['flipxy'] = True
    vol_stack = [np.array(img_volume).astype(np.float32)]
    vol_anno =  np.ones_like(vol_stack[0])
    vol_supervoxels = np.ones_like(vol_stack[0])
    filtered_layers = vol_stack

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)

    feats = Features(filtered_layers, dataset_feats, features_stack)
    vol_supervoxels = np.ones_like(vol_stack[0]) #superregions.supervoxel_vol.astype(np.uint32)
    segdata = SegData(vol_stack, feats, vol_anno, vol_supervoxels)
    layer_names = ['One',]
    opacities = [1.0,]

    clientData = ClientData(vol_stack, vol_anno, vol_supervoxels, 
                            segdata.feats, layer_names, opacities, 
                            entities_df, wparams['class_names'])

    clientData.wparams = wparams
    
    return clientData


def init_vf(wparams):
    dataset_name = wparams['dataset_name']
    datasets_dir = wparams['datasets_dir']

    entities_df = pd.read_csv(wparams.entities_relpath)
    entities_df.drop(entities_df.columns[entities_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    #entities_df = make_entity_df(np.array(entities_df), flipxy=True)

    scale_alpha = 0.1
    PATCH_DIM = 32
    gen_big = False
    patch_size=(20,224,224)
    wide_patch_pos = (80,1000,1000)

    #workspace_params = {}
    #workspace_params['scale_alpha'] = scale_alpha
    #workspace_params['PATCH_DIM']=PATCH_DIM
    #workspace_params['patch_size']=patch_size
    #workspace_params['wide_patch_pos']=wide_patch_pos
    #workspace_params['dataset_name'] = dataset_name
    
    fname = wparams['vol_fname'] #"combined_zooniverse_data.h5"    
    original_data = h5py.File(os.path.join(datasets_dir, fname), 'r')
    ds = original_data['dataset']
    img_volume = ds[dataset_name]


    from survos2.entity.sampler import crop_vol_and_pts
    
    main_coords = [(32,800,800),(16,450,450),(32,700,700),(32,850,550),(32,500,500),(32,1450,1450),]
    precropped_vol_size = 80,224,224 #48,448,448#224,224
    overall_offset = main_coords[2]

    img_volume, precropped_pts = crop_vol_and_pts(img_volume,np.array(entities_df),
                            location=overall_offset,
                            patch_size=precropped_vol_size,
                            debug_verbose=True,
                            offset=True)

    entities_df = make_entity_df(precropped_pts, flipxy=True)
    #wf1 = ds.get('workflow_1')  # <HDF5 dataset "workflow1_tv_denoised_padded": shape (165, 2112, 2160), type "<f4">
    
    vol_shape_x = img_volume[0].shape[0]
    vol_shape_y = img_volume[0].shape[1]
    vol_shape_z = len(img_volume)
    
    wparams['vol_shape_x'] = vol_shape_x
    wparams['vol_shape_y'] = vol_shape_y
    wparams['vol_shape_z'] = vol_shape_z
    
    #vol_stack = [np.array(np.random.random((100,500,500))),]
    vol_stack = [np.array(img_volume).astype(np.float32)]
    #vol_stack = [np.array(wf2),]
    #vol_anno = output_tensor.detach().numpy()
    vol_anno =  np.ones_like(vol_stack[0])
    
    
    vol_supervoxels = np.ones_like(vol_stack[0])

    blobs = skimage.data.binary_blobs(length=vol_stack[0].shape[1], n_dim=3, volume_fraction=0.3)

    
    from skimage.measure import label
    labeled_blobs = label(blobs)
    vol_supervoxels[0:vol_stack[0].shape[0],
                    0:vol_stack[0].shape[1],
                    0:vol_stack[0].shape[2]] = labeled_blobs[0:vol_stack[0].shape[0],
                    0:vol_stack[0].shape[1],
                    0:vol_stack[0].shape[2]] 
    


    filtered_layers = vol_stack

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)

    #vol_anno = prob_map
    feats = Features(filtered_layers, dataset_feats, features_stack)
    segdata = SegData(vol_stack, feats, vol_anno, vol_supervoxels)
    layer_names = ['One',]
    opacities = [1.0,]

    clientData = ClientData(vol_stack, vol_anno, vol_supervoxels, 
                            segdata.feats, layer_names, opacities, 
                            entities_df, wparams['class_names'])

    clientData.wparams = wparams
    
    return clientData

def init_brain(wparams):
    data = h5py.File(os.path.join(wparams.datasets_dir,"data.h5"), 'r')#this is image slices file
    
    entities_df = pd.read_csv(wparams.entities_relpath)
    entities_df.drop(entities_df.columns[entities_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
    #entities_pts = np.zeros((entities_df.shape[0],4))
    #entities_pts[:, 0:3] = np.array(entities_df)[:,0:3]
    #cc = [0] * len(entities_pts)
    #entities_pts[:, 3] =  cc
    
    #logger.debug(f"Added class code to bare pts producing {entities_pts.shape}")
    
    entities_df = make_entity_df(np.array(entities_df), flipxy=False)
    print(f"Loaded entities {entities_df.shape}")
    img_volume = data[wparams['dataset_name']]#[32:96,64:128,64:128]#this is orginal data

    print(f"Loaded volume of shape {img_volume.shape}")
    
    scale_alpha = 0.1
    PATCH_DIM = 32
    gen_big = False
    patch_size=(20,224,224)
    wide_patch_pos = (80,1000,1000)

    vol_shape_x = img_volume[0].shape[0]
    vol_shape_y = img_volume[0].shape[1]
    vol_shape_z = len(img_volume)

    filtered_layers = [np.array(img_volume).astype(np.float32)]
    vol_anno = np.ones_like(filtered_layers[0])
    filtered_layers = vol_stack
    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    feats = Features(filtered_layers, dataset_feats, features_stack)
    vol_supervoxels = np.ones_like(filtered_layers[0]) #superregions.supervoxel_vol.astype(np.uint32)
    segdata = SegData(filtered_layers, feats, vol_anno, vol_supervoxels)
    layer_names = ['One',]
    opacities = [1.0,]

    clientData = ClientData(filtered_layers, vol_anno, vol_supervoxels, 
                            segdata.feats, layer_names, opacities, 
                            entities_df, wparams['class_names'])
    clientData.wparams = wparams
    
    return clientData

def init_trypanosoma(wparams):
    dataset_dir = "D:/datasets/JSB_Trypanosoma/"
    sdata_rootpath = os.path.join(dataset_dir, "/survos_workspace/")
    data_fullpath = sdata_rootpath + "/RawData.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'



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
    
    #survos.run_command("workspace", "add_dataset", uri=None, workspace=ws_name, dataset_name='mainvol', dtype='float32')


def init_proj(wparams, precrop=False):
    
    DataModel.g.current_workspace = wparams.workspace

    entities_df = pd.read_csv(os.path.join(wparams.project_dir, wparams.entities_relpath))
    entities_df.drop(entities_df.columns[entities_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    entities_df = make_entity_df(np.array(entities_df), flipxy=wparams.flipxy)
    print(f"Loaded entities {entities_df.shape}")

    #data = h5py.File(os.path.join(wparams.datasets_dir,"data.h5"), 'r')#this is image slices file
    #img_volume = data[wparams['dataset_name']]#[32:96,64:128,64:128]#this is orginal data

    survos.init_api()
    src = DataModel.g.dataset_uri('__data__')
    with DatasetManager(src, out=None, dtype='float32', fillvalue=0) as DM:
        print(DM.sources[0].shape)
        src_dataset = DM.sources[0]
        img_volume = src_dataset[:]
    
    
    print(f"Loaded volume of shape {img_volume.shape}")
    vol_shape_x = img_volume[0].shape[0]
    vol_shape_y = img_volume[0].shape[1]
    vol_shape_z = len(img_volume)
        
    if precrop:
        from survos2.entity.sampler import crop_vol_and_pts
        
        precrop_coord = wparams.precrop_coord
        precrop_vol_size = wparams.precrop_vol_size
        logger.info(f"Precropping at {precrop_coord} to {precrop_vol_size}")

        img_volume, precropped_pts = crop_vol_and_pts(img_volume,np.array(entities_df),
                                location=precrop_coord,
                                patch_size=precrop_vol_size,
                                debug_verbose=True,
                                offset=True)

        entities_df = make_entity_df(precropped_pts, flipxy=False)

        tmp_ws_name = DataModel.g.current_workspace + "_tmp" 
        
        result = survos.run_command("workspace", "get", uri=None, workspace=tmp_ws_name)

        if not type(result[0]) == dict:
            print("Creating workspace")
            survos.run_command("workspace", "create", uri=None, workspace=tmp_ws_name) 
        else:
            print("tmp exists, deleting and recreating")
            survos.run_command("workspace", "delete", uri=None, workspace=tmp_ws_name) 
            print("workspace deleted")
            survos.run_command("workspace", "create", uri=None, workspace=tmp_ws_name) 
            print("workspace recreated")

        import h5py
        tmpvol_fullpath = "out\\tmpvol.h5"
        with h5py.File(tmpvol_fullpath,  'w') as hf:
            hf.create_dataset("data",  data=img_volume)

        survos.run_command('workspace', 'add_data', uri=None, workspace=tmp_ws_name,
                    data_fname=tmpvol_fullpath,
                    dtype='float32')
        DataModel.g.current_workspace = tmp_ws_name    
    
    # Prepare clientData
    filtered_layers = [np.array(img_volume).astype(np.float32)]
    vol_anno = np.ones_like(filtered_layers[0])
    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    feats = Features(filtered_layers, dataset_feats, features_stack)
    vol_supervoxels = np.ones_like(filtered_layers[0]) #superregions.supervoxel_vol.astype(np.uint32)
    segdata = SegData(filtered_layers, feats, vol_anno, vol_supervoxels)
    layer_names = ['One',]
    opacities = [1.0,]

    clientData = ClientData(filtered_layers, vol_anno, vol_supervoxels, 
                            segdata.feats, layer_names, opacities, 
                            entities_df, wparams['class_names'])
    clientData.wparams = wparams


    return clientData




def setup_ws(name='vf', project_file=None):    
    if project_file:
        with open(project_file) as project_file:    
            wparams = json.load(project_file)
            wparams = AttrDict(wparams)
            #wparams.datasets_dir = "/dls/science/groups/das/zooniverse/virus_factory/data"
            clientData = init_proj(wparams)
    else:
        if name=='hunt':
            with open('./projects/hunt1/ws_hunt1.json') as project_file:    
                wparams = json.load(project_file)
                wparams = AttrDict(wparams)
                print(wparams.patch_size[0])
                clientData = init_hunt(wparams)

        elif name=='vf':
            with open('./projects/vf1/ws_vf1.json') as project_file:    
                    wparams = json.load(project_file)
                    wparams = AttrDict(wparams)
            print(wparams)
            clientData = init_vf(wparams)

        elif name=='brain':
            with open('./projects/brain/ws_brain.json') as project_file:    
                wparams = json.load(project_file)
                wparams = AttrDict(wparams)
                print(wparams.patch_size[0])
                clientData = init_brain(wparams)

        elif name =='trypanosoma':
            with open('./projects/tryp1/ws_trypanosoma.json') as project_file:    
                wparams = json.load(project_file)
                wparams = AttrDict(wparams)
                print(wparams.patch_size[0])
                clientData = init_trypanosoma(wparams)
    
    return clientData


def startup(name, project_file):
    #clientData_to_workspace(clientData)
    clientData = setup_ws(name, project_file)
    viewer = frontend(clientData)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='brain')
    parser.add_argument('-p', '--project_file', default='./projects/brain/ws_brain.json')
    parser.add_argument('-i', '--init_name', default=None)
    args = parser.parse_args()
    
    if not args.init_name:
        startup(args.name, args.project_file)
    else: 
        init_ws(args.project_file, args.init_name)
if __name__ == "__main__":
    main()


    """
#old loader

if proj == 'vf':
    sdata_rootpath = "D:/datasets/survos_brain/ws3"
    sdata_rootpath = "D:/datasets/epfl_EM_dataset"
    sdata_rootpath = "D:/datasets/survos_hunt"
    sdata_rootpath = "D:/datasets/survos_brain/ws1"
    sdata_rootpath = "D:/datasets/VF_S1"  # survos data root

    data_fullpath = sdata_rootpath + "/data.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'
    output = sdata_rootpath
    output = "D:/datasets/VF_S1/output"
    features_path = sdata_rootpath + "/channels/*.h5"
    # features_path = sdata_rootpath + "/features/*.h5"
    annotations_fullpath = sdata_rootpath + "/annotations/*.h5"  # File/folder pattern or list of annotations
    print(features_path)
    # f.close()
elif proj == 'placenta':
    datasets_dir = "D:\\datasets\\survos_hunt"
    tiff_folder = "D:\\datasets\\placenta_dataset"
    final = []
    for fname in os.listdir(tiff_folder):
        im = Image.open(os.path.join(tiff_folder, fname))
        imarray = np.array(im)
        final.append(imarray)

    dt = np.asarray(final)
    #placeholder
    sdata_rootpath = datasets_dir
    data_fullpath = sdata_rootpath + "/data.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'
    features_path = sdata_rootpath + "/channels/*.h5"
    # features_path = sdata_rootpath + "/features/*.h5"
    annotations_fullpath = sdata_rootpath + "/annotations/*.h5"  # File/folder pattern or list of annotations

elif proj == 'trypanosoma':
    dataset_dir = "D:/datasets/JSB_Trypanosoma/"
    sdata_rootpath = os.path.join(dataset_dir, "/survos_workspace/")
    data_fullpath = sdata_rootpath + "/RawData.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'

elif proj == 'hunt':
    datasets_dir = "D:\\datasets\\survos_hunt"
    sdata_rootpath = datasets_dir
    data_fullpath = sdata_rootpath + "/data.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'
    features_path = sdata_rootpath + "/channels/*.h5"
    # features_path = sdata_rootpath + "/features/*.h5"
    annotations_fullpath = sdata_rootpath + "/annotations/*.h5"  # File/folder pattern or list of annotations
elif proj == 'bacteria':
    sdata_rootpath = "D:\\datasets\\patrick\\"
    data_fullpath = sdata_rootpath
    tif_fullpath = os.path.join(sdata_rootpath,"cropforAvery-1.tif")
    dt = np.array(io.imread(tif_fullpath))
    dataset_dir = "D:\\datasets\\survos_hunt"
    sdata_rootpath = dataset_dir
    data_fullpath = sdata_rootpath + "/data.h5"  # File/folder pattern or list of feature volumes in HDF5, MRC or SuRVoS format.'
    features_path = sdata_rootpath + "/channels/*.h5"
    # features_path = sdata_rootpath + "/features/*.h5"
    annotations_fullpath = sdata_rootpath + "/annotations/*.h5"  # File/folder pattern or list of annotations
# print (list(data.keys()))


if proj == 'trypanosoma':
    dt = data['t0']
    dt = dt['channel0']
    print("Volume shape : {}".format(dt.shape))
elif proj=='hunt':
    dt = data['data']

    print("Volume shape : {}".format(dt.shape))

# # Cropping and resampling

# roi_crop = (30,60, 300,500, 100,400)
# roi_crop = (0,165, 0, 768,0, 1024)
roi_crop = select_region(dt)
print(proj)
if proj == 'hunt':
    roi_crop = (0, 350, 350, 800, 350, 800)
    roi_crop = (0, 350, 0, 900, 0, 900)
    resample_amt = 0.5
elif proj == 'vf':
    roi_crop = (0, 90, 200, 1200, 200, 1200)
    resample_amt = 0.5
elif proj == 'bacteria':
    resample_amt =  1.0
    roi_crop = (0, 25, 200, 750, 200, 750)
print("Cropping to: {}".format(roi_crop))


    """