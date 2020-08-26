
import numpy as np
from magicgui import magicgui
from napari import layers
import enum
from skimage import img_as_ubyte
from loguru import logger   

from survos2.server.features import prepare_prediction_features, generate_features, feature_factory
from survos2.server.config import appState
#from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
#from survos2.server.filtering import simple_laplacian, spatial_gradient_3d, gaussian_blur
from survos2.server.model import SRData, SRFeatures
#from survos2.server.superseg import sr_prediction
#from survos2.server.supervoxels import superregion_factory,  generate_supervoxels



#class FilterOption(enum.Enum):
#    gaussian = [gaussian, appState.scfg.filter1['gauss_params'] ]
#    laplacian = [simple_laplacian, appState.scfg.filter4['laplacian_params'] ]
#    tv = [tvdenoising3d, appState.scfg.filter3['tvdenoising3d_params'] ]
#    gradient = [spatial_gradient_3d, appState.scfg.filter5['gradient_params']]
#    gblur = [gaussian_blur, appState.scfg.filter5['gradient_params']]

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
    appState.scfg['roi_crop'] = (z_st, z_end, x_st, x_end, y_st, y_end)
    
@magicgui(call_button="To Workspace", layout='vertical')
def workspace_gui(base_image: layers.Image, filter_option : str):    
    img_vol = base_image.data              
    logger.debug(f"Selected layer name: {base_image.name} and shape: {img_vol.shape} ") 
    logger.debug(f"filter_option: {filter_option}")
    

@magicgui(call_button="Predict", layout='vertical')
def prediction_gui(feature_1: layers.Image, feature_2: layers.Image, 
                    supervoxel_image: layers.Image, anno_image: layers.Image) -> layers.Image:   
    feats = feature_factory([feature_1.data, feature_2.data ])
    logger.info(feats.features_stack[0].shape)
    logger.info(len(feats.features_stack))
    
    sr = superregion_factory(supervoxel_image.data.astype(np.uint32), feats.features_stack)
    
    srprediction = sr_prediction(feats.features_stack,
                                    anno_image.data.astype(np.uint16),
                                    sr,
                                    appState.scfg.predict_params)
    
    
    
    return srprediction.prob_map

    
