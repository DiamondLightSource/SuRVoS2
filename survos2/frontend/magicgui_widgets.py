import numpy as np
from magicgui import magicgui
from napari import layers
import enum
from skimage import img_as_ubyte
from loguru import logger   

from survos2.server.features import prepare_prediction_features, generate_features, features_factory
from survos2.server.model import SRData, SRFeatures
from survos2.server.config import scfg
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.frontend.control import Launcher
from survos2.frontend.model import ClientData

class Operation(enum.Enum):
    mask_pipeline = 'mask_pipeline'
    saliency_pipeline = 'saliency_pipeline'
    superprediction_pipeline = 'prediction_pipeline'
    survos_pipeline = 'survos_pipeline'


class TransferOperation(enum.Enum):
    features = 'features'
    regions = 'regions'
    annotations = 'annotations'

@magicgui(auto_call=True) # call_button="Set Pipeline")
def pipeline_gui(pipeline_option : Operation):
    scfg['pipeline_option'] = pipeline_option.value

@magicgui(call_button="Assign ROI", layout='horizontal', 
          x_st={"maximum": 5000}, y_st={"maximum": 5000}, z_st={"maximum": 5000},
          x_end={"maximum": 5000}, y_end={"maximum": 5000}, z_end={"maximum": 5000})
def roi_gui(z_st : int, z_end: int,  x_st:int, x_end:int, y_st:int, y_end: int):  
    scfg['roi_crop'] = (z_st, z_end, x_st, x_end, y_st, y_end)
    

@magicgui(call_button="Update annotation", layout='vertical')
def workspace_gui(Layer: layers.Image): #, Group : TransferOperation):            
    logger.debug(f"Selected layer name: {Layer.name} and shape: {Layer.data.shape} ") 
    params = dict(feature_type='viewer', workspace=True)
    
    #if Group.name=='features':
    #    result = Launcher.g.run('features', 'create', **params)
    #elif Group.name=='annotations':
    #    result = Launcher.g.run('annotations', 'add_level', **params)
    #elif Group.name=='regions':
    #    result = Launcher.g.run('regions', 'create', **params)

    #result = Launcher.g.run('annotations', 'add_level', **params)

    params = dict(level=Layer.name, workspace=True)
    result = Launcher.g.run('annotations', 'get_levels', **params)[0]
    print(result)

    if result:
        fid = result['id']
        ftype = result['kind']
        fname = result['name']
        logger.debug(f"Transferred to workspace {fid}, {ftype}, {fname}")
    
    dst = DataModel.g.dataset_uri(fid, group='annotations')     

    with DatasetManager(dst, out=dst, dtype='uint16', fillvalue=0) as DM:
        DM.out[:] = Layer.data

    #if Group.name=='features':
    #    with DatasetManager(dst, out=dst, dtype='float32', fillvalue=0) as DM:
    #        DM.out[:] = Layer.data
    #elif Group.name == 'annotations':
    #    with DatasetManager(dst, out=dst, dtype='uint16', fillvalue=0) as DM:
    #        DM.out[:] = Layer.data
    #elif Group.name == 'regions':
    #    with DatasetManager(dst, out=dst, dtype='uint32', fillvalue=0) as DM:
    #        DM.out[:] = Layer.data
            
    scfg.ppw.clientEvent.emit({'source': 'workspace_gui', 'data':'refresh', 'value': None})
    
