"""
Pipeline ops

"""
import os
import numpy as np
from typing import List, Optional
import itertools

from dataclasses import dataclass

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from matplotlib import offsetbox
from skimage import img_as_ubyte
from skimage import exposure
from skimage.segmentation import mark_boundaries
from scipy import ndimage

import seaborn as sns
import morphsnakes as ms
from skimage import img_as_ubyte



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

import torchio as tio
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator

from loguru import logger

from survos2.entity.entities import make_entity_df
from survos2.server.prediction import make_prediction, calc_feats
from survos2.entity.anno.masks import generate_sphere_masks_fast
from survos2.server.supervoxels import generate_supervoxels
from survos2.server.filtering import generate_features
from survos2.server.config import appState
from survos2.server.filtering import feature_factory
from survos2.server.filtering import generate_features, simple_laplacian
from survos2.server.model import Superregions, SRPrediction, Features
from survos2.entity.sampler import crop_vol_and_pts
from survos2.helpers import AttrDict
from survos2.entity.anno import geom
from survos2.utils import logger
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
from survos2.entity.detect.backbone import convblock, ResNetUNet, convblock2
from survos2.entity.detect.trainer import  (loss_dice, loss_calc, log_metrics,  
                                            train_model_cbs, TrainerCallback)
from survos2.entity.saliency import filter_proposal_mask
from survos2.entity.saliency import measure_big_blobs
from survos2.entity.sampler import centroid_to_bvol, viz_bvols

#from survos2.entity.detect.backbone import convblock, convblock2
#from survos2.frontend.nb_utils import view_vols_points, view_label, view_volumes, view_volume
from survos2.server.model import Features
from survos2.frontend.nb_utils import show_images


@dataclass
class PipelinePayload():
    main_volume : np.ndarray  # one channel volume, Z,X,Y
    layers : List[np.ndarray] # masks and other non-feature layers
    entities : np.ndarray
    features : Optional[Features]
    superregions : Optional[Superregions]
    prediction : Optional[SRPrediction]
    models : dict
    params : dict


def sphere_masks(p : PipelinePayload, mask_radius):
    total_mask = generate_sphere_masks_fast(p.layers['in_array'], 
                                            p.entities, radius=mask_radius)
    #core_mask = generate_sphere_masks_fast(p.layers['in_array'], 
    #                                       p.entities, radius=10)
    
    #show_images([total_mask[total_mask.shape[0]//2,:]], 
    #             ['Sphere mask, radius: ' + str(mask_radius),])
                 
    p.layers['generated'] = total_mask
                 
    return p

def make_masks(p : PipelinePayload):
    total_mask = generate_sphere_masks_fast(p.layers['in_array'], 
                                            p.entities, radius=20)
    core_mask = generate_sphere_masks_fast(p.layers['in_array'], 
                                           p.entities, radius=10)
    #show_images([total_mask[total_mask.shape[0]//2,:], 
    #             core_mask[core_mask.shape[0] // 2,:]])

    p.layers['total_mask'] = total_mask
    p.layers['core_mask'] = core_mask
    
    return p


def make_features(p : PipelinePayload, feature_params):

    # FEATURES (R -> List[R])
    logger.info("Calculating features")
    
    cropped_vol = p.main_volume    
    #feats = calc_feats(cropped_vol)
    #I_out = feats.filtered_layers[3] +  0.5 * feats.filtered_layers[2]
    #I_out = exposure.adjust_sigmoid(I_out,cutoff=0.7)
    roi_crop = [0,cropped_vol.shape[0],0,cropped_vol.shape[1], 0,cropped_vol.shape[2]]
    
    feats = generate_features(cropped_vol, feature_params, roi_crop, 1.0)
    #feats_idx = 0     
    
    p.features = feats
        
    return p


def acwe(p: PipelinePayload):    
    edge_map =1.0 - p.features.filtered_layers[0]
    edge_map = exposure.adjust_sigmoid(edge_map,
                                    cutoff=1.0)
    
    logger.info("Calculating ACWE")    
    seg1 = ms.morphological_geodesic_active_contour(edge_map,
                                                    iterations=3, 
                                                    init_level_set=p.layers['total_mask'],
                                                    smoothing=1,
                                                    threshold=0.1,
                                                    balloon=1.1)
    #show_images([seg1[seg1.shape[0] // 2,:]])

    outer_mask = ((seg1 * 1.0) > 0) * 2.0

    #inner_mask = ((seg2 * 1.0) > 0) * 1.0
    #outer_mask = outer_mask * (1.0 - inner_mask)
    #anno = outer_mask + inner_mask
        
    p.layers['acwe'] = outer_mask
    
    return p

    
def make_sr(p: PipelinePayload,  slic_params, feats_idx = -1):
    logger.info('Making superregions')
    
    #filt_layers2 = [p.main_volume,#I_out, 
    #                p.features.filtered_layers[0],
    #                p.features.filtered_layers[1], 
    #                p.features.filtered_layers[2], 
    #                p.features.filtered_layers[3]]
    
    feats2 = feature_factory(p.features.filtered_layers)
    
    p.superregions = generate_supervoxels(np.array(p.features.dataset_feats), 
                                        p.features.features_stack, 
                                        feats_idx, slic_params)

    #middle_slice = superregions.supervoxel_vol.shape[0] // 2
    #plt.figure(figsize=(12,12))
    #plt.imshow(mark_boundaries(superregions.supervoxel_vol[middle_slice,:], 
    #                           superregions.supervoxel_vol[middle_slice,:]))

    logger.debug(p.superregions)
    
    return p

            
def predict_sr(p : PipelinePayload, anno_name):
    logger.info("Predicting superregions")
    srprediction = make_prediction(p.features.features_stack,
                                   p.layers[anno_name],
                                   p.superregions,
                                   p.params.predict_params)

    predicted = srprediction.prob_map.copy()
    predicted -= np.min(srprediction.prob_map)
    predicted += 1
    predicted = img_as_ubyte(predicted / np.max(predicted))

    predicted = predicted - np.min(predicted)
    predicted = predicted / np.max(predicted)
    
    # clean prediction
    struct2 = ndimage.generate_binary_structure(3, 2)
    predicted_cl = (predicted > 0.0) * 1.0
    predicted_cl = ndimage.binary_closing(predicted_cl, structure=struct2)
    predicted_cl = ndimage.binary_opening(predicted_cl, structure=struct2)
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))

    #show_images([predicted[predicted.shape[0] // 2,:,]], figsize=(12,12))
    p.layers['prediction'] = predicted_cl
    
    return p    


def setup_cnn_model():
    #cb = TrainerCallback()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_class = 1
    model = ResNetUNet(num_class, convblock2).to(device)

    
    checkpoint_directory = appState.scfg.torch_models_fullpath #torch_models_fullpath
    file_path = os.path.join(checkpoint_directory, 'resnetUnet_model_0511_b.pt')
    load_mod = True

    if load_mod:
        def load_model_parameters(full_path):
            checkpoint = torch.load(full_path)
            return checkpoint

        checkpoint_file = 'model.pt'
        full_path = os.path.join(checkpoint_directory, checkpoint_file) 
        checkpoint = load_model_parameters(file_path)
        model.load_state_dict(checkpoint['model_state'])
        #optimizer_ft.load_state_dict(checkpoint['model_optimizer'])
        model.eval()
    return model


def saliency_pipeline(p: PipelinePayload):
    cropped_vol = p.main_volume    
    output_tensor = predict_and_agg(p.models['saliency_model'], cropped_vol, patch_size=(1,224,224), 
                                patch_overlap=(0,0,0), batch_size=1,
                                stacked_3chan=True)
    p.layers['result'] = output_tensor.detach().squeeze(0).numpy()

    proposal_mask = output_tensor.detach().squeeze(0).numpy()
    holdout = filter_proposal_mask(output_tensor.detach().squeeze(0).numpy(), 
        thresh=0.5, num_erosions=3, num_dilations=3)
    images =  [holdout,]
    filtered_tables = measure_big_blobs(images)
    logger.debug(filtered_tables)
    cols = ['z', 'x','y', 'class_code']
    pred_cents = np.array(filtered_tables[0][cols])
    #cents = cents[:,[0,2,1,3]]
    #target_cents = np.array(cropped_pts_df)[:,0:4]

    preds = centroid_to_bvol(pred_cents)
    saliency_bb = viz_bvols(p.main_volume, preds)
    logger.info(f"Produced bb mask of shape {saliency_bb.shape}")
    #targs = centroid_to_bvol(target_cents)
    p.layers['saliency_bb'] = saliency_bb

    return p



    
def rasterize_bb(p : PipelinePayload):
    holdout = filter_proposal_mask(proposal_mask, thresh=0.5, num_erosions=3, num_dilations=3)
    filtered_tables = measure_big_blobs(images)
    
    cols = ['z', 'x','y', 'class_code']
    pred_cents = np.array(filtered_tables[0][cols])
    #cents = cents[:,[0,2,1,3]]
    target_cents = np.array(cropped_pts_df)[:,0:4]

    preds = centroid_to_bvol(pred_cents)
    targs = centroid_to_bvol(target_cents)

    p.layers['preds'] = preds
    p.layers['targs'] = targs
    
    return p


def cnn_predict_2d_3chan(p: PipelinePayload, model):
    inputs_t = torch.FloatTensor(p.main_volume)

    #inputs_t = input_tensor.squeeze(1)
    inputs_t = inputs_t.to(device)
    stacked_t = torch.stack([inputs_t[:,0,:,:],
                             inputs_t[:,0,:,:],
                             inputs_t[:,0,:,:]], axis=1)
    logger.info(f"inputs_t: {inputs_t.shape}")
    pred = model(stacked_t)
    #pred, feats = model.forward_(stacked_t)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu()
    output = pred.unsqueeze(1)
    print(f"pred: {pred.shape}")

    p.layers['prediction'] = predicted_cl
    
    return p


def mask_pipeline(p : PipelinePayload):
    p = make_masks(p)
    p.layers['result'] = p.layers['total_mask']
    
    return p


def superregion_pipeline(p : PipelinePayload):
    p = make_masks(p)
    p = make_features(p, appState.scfg.feature_params['simple_gaussian'])
    p = make_sr(p, p.params.slic_params)

    p.layers['result'] = p.superregions.supervoxel_vol
    
    return p

def prediction_pipeline(p : PipelinePayload):
    # RASTERIZATION (V->R)
    p = make_masks(p)

    # FEATURES (R -> List[R])
    p = make_features(p, appState.scfg.feature_params['vf2'])
        
    # SUPERREGIONS (R->R)
    p = make_sr(p, p.params.slic_params)
    
    # ACTIVE CONTOUR (List[R] -> R)
    #p = acwe(p)

    # SR_PREDICTION 
    p = predict_sr(p, anno_name='total_mask')
    
    p.layers['result'] = p.layers['prediction']
    
    return p


def predict_and_agg(model, input_array, patch_size=(1,224,224), 
    patch_overlap=(0,16,16), batch_size=1, stacked_3chan=False, extra_unsqueeze=True):
    """
    Stacked CNN Predict and Aggregate
    Uses torchio
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tens = torch.FloatTensor(input_array)
    print(img_tens.shape)

    one_subject = tio.Subject(
            img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
            label=tio.Image(tensor=img_tens, label=tio.LABEL)
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(grid_sampler)

    input_tensors = []
    output_tensors = []

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            
            #input_tensor = patches_batch[IMAGE] 
            #locations = patches_batch[LOCATION]
            
            input_tensor = patches_batch['img']['data'] 
            location = patches_batch[LOCATION]
        
            print(f"Input tensor {input_tensor.shape}")
            
            inputs_t = input_tensor.squeeze(1)
            inputs_t = inputs_t.to(device)

            if stacked_3chan:
                inputs_t = torch.stack([inputs_t[:,0,:,:],inputs_t[:,0,:,:],inputs_t[:,0,:,:]], axis=1)
            else:
                inputs_t = inputs_t[:,0:1,:,:]
    
            print(f"inputs_t: {inputs_t.shape}")
            
            pred = model(inputs_t)
            #pred = torch.sigmoid(pred[0])
            pred = F.sigmoid(pred)
        
            #pred = pred.data.cpu()
            print(f"pred: {pred.shape}")

            if extra_unsqueeze:
                output = pred.unsqueeze(1)
            #output = pred.squeeze(0)
            
            input_tensors.append(input_tensor)
            output_tensors.append(output)

            print(output.shape, location)
            aggregator.add_batch(output, location)
            
    output_tensor = aggregator.get_output_tensor()
    print(input_tensor.shape,output_tensor.shape)

    return output_tensor


    
# todo: simple way to add multiple aggregators and bundle results
def tio_pipeline(input_array, entity_pts, scfg, patch_size=(64,64,64), patch_overlap=(0,0,0), batch_size=1):
    print("Starting up gridsampler pipeline...")
    
    input_tensors = []
    output_tensors = []

    entity_pts = entity_pts.astype(np.int32)

    img_tens = torch.FloatTensor(input_array)

    one_subject = tio.Subject(
            img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
            label=tio.Image(tensor=img_tens, label=tio.LABEL)
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator1 = GridAggregator(grid_sampler)
    aggregator2 = GridAggregator(grid_sampler)

    # use MarkedVols?
    payloads = []
    with torch.no_grad():        
        for patches_batch in patch_loader:
            #print(patches_batch)
            #input_tensor = patches_batch['img']['data'] 
            #print(f"Input tensor {input_tensor.shape}")
            
            locations = patches_batch[LOCATION]
            
            loc_arr = np.array(locations[0])
            loc = (loc_arr[0], loc_arr[1], loc_arr[2])
            print(f"Location: {loc}")
            
            # PREP_REGION DATA(IMG (R, Raster) AND GEOMETRY (V, Vector))
            cropped_vol, cropped_pts = crop_vol_and_pts(input_array,
                                                        entity_pts,
                                                        location=loc,
                                                        patch_size=patch_size,
                                                        offset=True,
                                                        debug_verbose=True)
            
            print(cropped_vol.shape, cropped_pts.shape)
            offset_pts = cropped_pts
            #offset_pts = prepare_point_data(cropped_pts, loc)

            plt.figure(figsize=(12,12))
            plt.imshow(cropped_vol[cropped_vol.shape[0]//2,:], cmap='gray')
            plt.scatter(cropped_pts[:,1], cropped_pts[:,2])
            print(f"Number of offset_pts: {offset_pts.shape}")
            print(f"Allocating memory for no. voxels: {cropped_vol.shape[0] * cropped_vol.shape[1] * cropped_vol.shape[2]}")

            payload = PipelinePayload(cropped_vol,
                                    {'in_array': cropped_vol},
                                    offset_pts,
                                    None,
                                    None,
                                    None,
                                    scfg)
            # Pipeline
            process_pipeline(payload)            

            # AGGREGATION (Output: large image)
            output_tensor = torch.FloatTensor(payload.layers['total_mask']).unsqueeze(0).unsqueeze(1)
            print(f"Aggregating output tensor of shape: {output_tensor.shape}")          
            aggregator1.add_batch(output_tensor, locations)
            
            output_tensor = torch.FloatTensor(payload.layers['prediction']).unsqueeze(0).unsqueeze(1)
            print(f"Aggregating output tensor of shape: {output_tensor.shape}")
            aggregator2.add_batch(output_tensor, locations)
            payloads.append(payload)


    output_tensor1 = aggregator1.get_output_tensor()
    print(output_tensor1.shape)    
    output_arr1 = np.array(output_tensor1.squeeze(0))

    output_tensor2 = aggregator2.get_output_tensor()
    print(output_tensor2.shape)    
    output_arr2 = np.array(output_tensor2.squeeze(0))

    return [output_tensor1,output_tensor2], payloads
