
import numpy as np
from magicgui import magicgui
from napari import layers
import enum
from skimage import img_as_ubyte
from loguru import logger   

from survos2.server.features import prepare_prediction_features, generate_features, features_factory
from survos2.frontend.model import ClientData
#from survos2.improc.features import gaussian, tvdenoising3d, gaussian_norm
#from survos2.server.filtering import simple_laplacian, spatial_gradient_3d, gaussian_blur
from survos2.server.model import SRData, SRFeatures
#from survos2.server.superseg import sr_prediction
#from survos2.server.supervoxels import superregion_factory,  generate_supervoxels



#class FilterOption(enum.Enum):
#    gaussian = [gaussian, scfg.filter1['gauss_params'] ]
#    laplacian = [simple_laplacian, scfg.filter4['laplacian_params'] ]
#    tv = [tvdenoising3d, scfg.filter3['tvdenoising3d_params'] ]
#    gradient = [spatial_gradient_3d, scfg.filter5['gradient_params']]
#    gblur = [gaussian_blur, scfg.filter5['gradient_params']]

class Operation(enum.Enum):
    mask_pipeline = 'mask_pipeline'
    saliency_pipeline = 'saliency_pipeline'
    superprediction_pipeline = 'prediction_pipeline'
    survos_pipeline = 'survos_pipeline'

@magicgui(auto_call=True) # call_button="Set Pipeline")
def pipeline_gui(pipeline_option : Operation):
    scfg['pipeline_option'] = pipeline_option.value

@magicgui(call_button="Assign ROI", layout='horizontal', 
          x_st={"maximum": 2000}, y_st={"maximum": 2000}, z_st={"maximum": 2000},
          x_end={"maximum": 2000}, y_end={"maximum": 2000}, z_end={"maximum": 2000})
def roi_gui(z_st : int, z_end: int,  x_st:int, x_end:int, y_st:int, y_end: int):  
    scfg['roi_crop'] = (z_st, z_end, x_st, x_end, y_st, y_end)
    
@magicgui(call_button="To Workspace", layout='vertical')
def workspace_gui(base_image: layers.Image, filter_option : str):    
    img_vol = base_image.data              
    logger.debug(f"Selected layer name: {base_image.name} and shape: {img_vol.shape} ") 
    logger.debug(f"filter_option: {filter_option}")
    


    
