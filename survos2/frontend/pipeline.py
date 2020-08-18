
import numpy as np
from magicgui import magicgui
from napari import layers
#from .semantic import fit, predict
#from .util import Featurizers, norm_entropy
from survos2.server.supervoxels import generate_supervoxels
from survos2.server.filtering import prepare_prediction_features, generate_features
from survos2.server.config import appState
from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
from survos2.server.filtering import simple_laplacian, spatial_gradient_3d, gaussian_blur3d
from survos2.server.model import Superregions, Features
from survos2.server.filtering import feature_factory
from survos2.server.supervoxels import superregion_factory
from survos2.server.prediction import make_prediction, calc_feats

import enum
from skimage import img_as_ubyte
from survos2.server.pipeline import PipelinePayload, mask_pipeline, saliency_pipeline, prediction_pipeline, survos_pipeline


class PipelineOp(enum.Enum):
    saliency_pipeline = saliency_pipeline
    mask_pipeline = mask_pipeline
    superprediction_pipeline = prediction_pipeline
    survos_pipeline = survos_pipeline

class FilterOption(enum.Enum):
    gaussian = [gaussian, appState.scfg.filter1['gauss_params'] ]
    laplacian = [simple_laplacian, appState.scfg.filter4['laplacian_params'] ]
    tv = [tvdenoising3d, appState.scfg.filter3['tvdenoising3d_params'] ]
    gradient = [spatial_gradient_3d, appState.scfg.filter5['gradient_params']]
    gblur = [gaussian_blur3d, appState.scfg.filter5['gradient_params']]

class Operation(enum.Enum):
    mask_pipeline = 'mask_pipeline'
    saliency_pipeline = 'saliency_pipeline'
    superprediction_pipeline = 'prediction_pipeline'
    survos_pipeline = 'survos_pipeline'


@magicgui(auto_call=True) # call_button="Set Pipeline")
def pipeline_gui(pipeline_option : Operation):
    appState.scfg['pipeline_option'] = pipeline_option.value


@magicgui(call_button="Assign ROI", layout='horizontal', 
          x_st={"maximum": 2000}, y_st={"maximum": 2000}, z_st={"maximum": 2000},
          x_end={"maximum": 2000}, y_end={"maximum": 2000}, z_end={"maximum": 2000})
def roi_gui(z_st : int, z_end: int,  x_st:int, x_end:int, y_st:int, y_end: int):  
    print(z_st, z_end, x_st, x_end, y_st, y_end)
    appState.scfg['roi_crop'] = (z_st, z_end, x_st, x_end, y_st, y_end)
    

@magicgui(call_button="To Workspace", layout='vertical')
def workspace_gui(base_image: layers.Image, filter_option : str):    
    img_vol = base_image.data   
           
    logger.debug(f"Selected layer name: {base_image.name} and shape: {img_vol.shape} ") 
    logger.debug(f"filter_option: {filter_option}")
    

@magicgui(call_button="Calc Features", layout='vertical')
def features_gui(base_image: layers.Image, filter_option : FilterOption) -> layers.Image:    
    cropped_vol = base_image.data   
    
    feature_params = [ filter_option.value]
               
    logger.debug(f"Selected layer name: {base_image.name}") 
    logger.debug(f"Feature params: {feature_params}")
    
    roi_crop = [0,cropped_vol.shape[0],0,cropped_vol.shape[1], 0,cropped_vol.shape[2]]
    feats = generate_features(cropped_vol, feature_params, roi_crop, 1.0)

    logger.info(f"Generated {len(feats.filtered_layers)} features.")
    
    return feats.filtered_layers[0]


@magicgui(call_button="Calculate SR", layout='vertical')
def sv_gui(base_image: layers.Image) -> layers.Labels:     #initial_labels: layers.Labels) -> layers.Labels:
            #           featurizer:Featurizers) -> layers.Labels:
    
    dataset_feats, filtered_stack = prepare_prediction_features([np.array(base_image.data),])
    superregions = generate_supervoxels(dataset_feats,  filtered_stack, 
                                        appState.scfg.feats_idx, appState.scfg.slic_params)

    print(f"Calculated superergions  {superregions.supervoxel_vol.shape}")  

    return superregions.supervoxel_vol #np.squeeze(data)

@magicgui(call_button="Predict", layout='vertical')
def prediction_gui(feature_1: layers.Image, feature_2: layers.Image, supervoxel_image: layers.Image, anno_image: layers.Image) -> layers.Image:   
    #dataset_feats, filtered_stack = prepare_prediction_features([np.array(base_image.data),])
    #superregions = generate_supervoxels(dataset_feats,  filtered_stack, 
    #                                    appState.scfg.feats_idx, appState.scfg.slic_params)

    #svol = ((supervoxel_image.data > 0.4) * 1).astype(np.uint32, copy= True)
    feats = feature_factory([feature_1.data, feature_2.data ])
    logger.info(feats.features_stack[0].shape)
    logger.info(len(feats.features_stack))
    sr = superregion_factory(supervoxel_image.data.astype(np.uint32), feats.features_stack)


    #p = PipelinePayload(base_image,
    #                    {'in_array': base_image.data},
    #                    np.array([0,0,0,0]),
    #                    feats,
    #                    sr,
    #                    None,
    #                    appState.scfg)

    #print(feats)
    print(sr)

    #print(feats.features_stack)

    #srprediction = make_prediction(feats.features_stack,
    #                                anno_image.data.astype(np.uint16),
    #                                sr,
    #                                appState.scfg.predict_params)
    
    from survos2.server.prediction import process_anno_and_predict

    prob_map, probs, pred_map, conf_map, P = process_anno_and_predict(feats.filtered_layers, anno_image.data.astype(np.uint16), supervoxel_image.data.astype(np.uint32),appState.scfg.predict_params)

    predicted = prob_map.copy()
    #predicted = srprediction.prob_map.copy()
    #predicted -= np.min(srprediction.prob_map)
    #predicted += 1
    #predicted = img_as_ubyte(predicted / np.max(predicted))
    #predicted = predicted - np.min(predicted)
    #predicted = predicted / np.max(predicted) 
  
    logger.debug(f"Predicted a volume of shape: {predicted.shape}")  #, Predicted features {P_dict}")
    print("Predicted superregions")  

    return predicted #np.squeeze(data)

    
def segment(p : PipelinePayload):
    logger.debug(f"Segmenting pipeline payload {p}")
    try:
        #prob_map, probs, pred_map, conf_map, P = sseg.process_anno_and_predict(self.features_stack, self.anno,
        #                                                #viewer.layers['Active Annotation'].data,
        #                                                self.supervoxel_vol, scfg.predict_params)
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

    
        logger.debug(f"Predicted a volume of shape: {predicted.shape}")  #, Predicted features {P_dict}")
        
        #self.prob_map = prob_map
        #self.add_labels(self.prob_map, name='Prediction Labels')
        # post-process prediction (allows isosurfaces etc.)
        #predicted = prob_map.copy()g
        #predicted -= np.min(prob_map)
        #predicted += 1
        #predicted = img_as_ubyte(predicted / np.max(predicted))
        #predicted = img_as_ubyte(ndimage.median_filter(predicted, size=4))
        #self.predicted = predicted
        #self.add_image(self.predicted, name='Normalised Prediction ')
        
        #viewer.add_image(predicted, name='Prediction')
        #self.add_image(conf_map, name='Confidence')
        #print(f"Conf map: {conf_map.shape} {conf_map}")

        return p

    except Exception as err:
        logger.debug(f"Exception at segmentation prediction: {err}")


def filter(viewer):
    ## RASTERIZATION (V->R)
    #payload = make_masks(payload)

    cropped_vol = viewer.layers['Original Image'].data

    payload = PipelinePayload(cropped_vol,
                        {'in_array': cropped_vol},
                        np.array([0,0,0,0]),
                        None,
                        None,
                        None,
                        appState.scfg)

    feature_params = [ [gaussian, appState.scfg.filter1['gauss_params']]]

     #   [gaussian, scfg.filter2['gauss_params'] ],
     #   [simple_laplacian, scfg.filter4['laplacian_params'] ],
     #   [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]]
    # feature_params = [ 
    # [gaussian, scfg.filter2['gauss_params'] ]]
    # FEATURES (R -> List[R])

    payload = make_features(payload, feature_params)
    logger.info(f"Filtered layer to produce {payload.features.filtered_layers}")
    
    return payload.features.features_stack[0]
