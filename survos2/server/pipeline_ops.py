from survos2.entity.entities import make_entity_df
from survos2.server.superseg import make_prediction, calc_feats
from survos2.entity.anno.masks import generate_sphere_masks_fast
from survos2.server.supervoxels import generate_supervoxels
from survos2.server.features import features_factory, generate_features, simple_laplacian
from survos2.server.model import SRData, SRPrediction, SRFeatures
from survos2.entity.sampler import crop_vol_and_pts_centered
from survos2.helpers import AttrDict
from survos2.entity.anno import geom
from survos2.frontend.model import ClientData
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
from survos2.entity.saliency import filter_proposal_mask
from survos2.entity.saliency import measure_big_blobs
from survos2.entity.sampler import centroid_to_bvol, viz_bvols
from survos2.server.pipeline import Patch



def sphere_masks(patch: Patch, params: dict):
    total_mask = generate_sphere_masks_fast(patch.image_layers['Main'], 
                                            patch.geometry_layers['Entities'], 
                                            radius=params.pipeline.mask_radius)
    #show_images([total_mask[total_mask.shape[0]//2,:]], 
    #             ['Sphere mask, radius: ' + str(mask_radius),])
                
    patch.image_layers['generated'] = total_mask
                
    return patch


def make_masks(patch: Patch, params:dict):
    """
    Rasterize point geometry to a mask (V->R)
    
    ((Array of 3d points) -> Float layer)

    """
    total_mask = generate_sphere_masks_fast(patch.image_layers['Main'], 
                                            patch.geometry_layers['Entities'], 
                                            radius=params.pipeline.mask_radius)
    
    core_mask = generate_sphere_masks_fast(patch.image_layers['Main'], 
                                           patch.geometry_layers['Entities'], 
                                           radius=params.pipeline.core_radius)
    #show_images([total_mask[total_mask.shape[0]//2,:], 
    #             core_mask[core_mask.shape[0] // 2,:]])

    patch.image_layers['total_mask'] = total_mask
    patch.image_layers['core_mask'] = core_mask
    
    return patch


def make_features(patch: Patch, params : dict):
    """
    Features (Float layer -> List[Float layer])
    
    """
    feature_params = params.feature_params['simple_gaussian']
    logger.info("Calculating features")
    
    cropped_vol =  patch.image_layers['Main Volume']

    roi_crop = [0,cropped_vol.shape[0],
                0,cropped_vol.shape[1], 
                0,cropped_vol.shape[2]]

    features = generate_features(cropped_vol, feature_params, roi_crop, 1.0)
    
    #TODOadd features to payload

    return patch

def acwe(patch: Patch, params : dict):
    """
    Active Contour
    
    (Float layer -> Float layer)
    
    """
    edge_map =1.0 - features.filtered_layers[0]
    edge_map = exposure.adjust_sigmoid(edge_map,
                                    cutoff=1.0)        
    logger.debug("Calculating ACWE")    

    seg1 = ms.morphological_geodesic_active_contour(edge_map,
                                                    iterations=3, 
                                                    init_level_set=patch.image_layers['total_mask'],
                                                    smoothing=1,
                                                    threshold=0.1,
                                                    balloon=1.1)
    
    outer_mask = ((seg1 * 1.0) > 0) * 2.0

    #inner_mask = ((seg2 * 1.0) > 0) * 1.0
    #outer_mask = outer_mask * (1.0 - inner_mask)
    #anno = outer_mask + inner_mask
        
    patch.image_layers['acwe'] = outer_mask
    
    return patch

    
def make_sr(patch: Patch, params : dict):
    """
    # SRData 
    #    Supervoxels (Float layer->Float layer)       
    #  
    #    Also generates feature vectors
    
    SRData contains the supervoxel image as well as the feature vectors made
    from features and the supervoxels 

    """
    logger.debug('Making superregions for sample')
        
    superregions = generate_supervoxels(np.array(features.dataset_feats), 
                                        features.features_stack, 
                                        params.slic_feat_idx, params.slic_params)
    logger.debug(superregions)
    
    return patch

            
def predict_sr(patch: Patch, params: dict):
    """
    (SRFeatures, Annotation, SRData -> Float layer prediction)
    
    """
    logger.debug("Predicting superregions")

    srprediction = _sr_prediction(features.features_stack,
                                patch.image_layers['Annotation'],
                                superregions,
                                params.predict_params)

    predicted = srprediction.prob_map.copy()
    predicted -= np.min(srprediction.prob_map)
    predicted += 1
    predicted = img_as_ubyte(predicted / np.max(predicted))
    predicted = predicted - np.min(predicted)
    predicted = predicted / np.max(predicted)
    
    patch.image_layers['segmentation'] = predicted

    return patch


def clean_segmentation(patch: Patch, params: params)

    predicted = patch.image_layers['segmentation']
    # clean prediction
    struct2 = ndimage.generate_binary_structure(3, 2)
    predicted_cl = (predicted > 0.0) * 1.0
    predicted_cl = ndimage.binary_closing(predicted_cl, structure=struct2)
    predicted_cl = ndimage.binary_opening(predicted_cl, structure=struct2)
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))

    patch.image_layers['prediction_cleaned'] = predicted_cl
    
    return patch  


def saliency_pipeline(patch: Patch, params: dict):
    """
    Use CNN to predict a saliency map from an image volume

    Post process the map into bounding boxes.
    """
    output_tensor = predict_and_agg(seg_models['saliency_model'], 
                                patch.image_layers['Main'], 
                                patch_size=(1,224,224), 
                                patch_overlap=(0,0,0), batch_size=1,
                                stacked_3chan=True)
    
    patch.image_layers['saliency_map'] = output_tensor.detach().squeeze(0).numpy()
    return patch
    # POST
    
def make_bb(patch: Patch, params: dict):
    holdout = filter_proposal_mask(patch.image_layers['saliency_map'], 
        thresh=0.5, num_erosions=3, num_dilations=3)
    images =  [holdout,]
    filtered_tables = measure_big_blobs(images)
    logger.debug(filtered_tables)
    cols = ['z', 'x','y', 'class_code']
    pred_cents = np.array(filtered_tables[0][cols])
    preds = centroid_to_bvol(pred_cents)
    saliency_bb = viz_bvols(patch.image_layers['Main Volume'], preds)
    logger.info(f"Produced bb mask of shape {saliency_bb.shape}")
    patch.image_layers['saliency_bb'] = saliency_bb
    
    return patch

    
def rasterize_bb(patch: Patch, params : dict):
    holdout = filter_proposal_mask(patch.image_layers['proposal_mask'], 
                            thresh=0.5, num_erosions=3, num_dilations=3)
    filtered_tables = measure_big_blobs(images)
    
    cols = ['z', 'x','y', 'class_code']
    pred_cents = np.array(filtered_tables[0][cols])
    #cents = cents[:,[0,2,1,3]]
    target_cents = np.array(cropped_pts_df)[:,0:4]

    preds = centroid_to_bvol(pred_cents)
    targs = centroid_to_bvol(target_cents)

    patch.image_layers['preds'] = preds
    patch.image_layers['targs'] = targs
    
    return patch


def cnn_predict_2d_3chan(patch : Patch, params : dict):
    model = seg_models['cnn']
    inputs_t = torch.FloatTensor(patch.image_layers['Main'])
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
    logger.debug(f"pred: {pred.shape}")

    patch.image_layers['cnn_prediction'] = predicted_cl
    
    return patch