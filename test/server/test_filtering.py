import os
import pytest
import h5py
import numpy as np 

from survos2 import survos
from survos2.improc.utils import DatasetManager
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.frontend.control import DataModel

from loguru import logger

from torch.testing import assert_allclose

@pytest.fixture(scope="session")
def datamodel():
    test_datadir = "/dls/science/groups/das/SuRVoS/s2/tests"
    
    # make test vol
    map_fullpath = os.path.join(test_datadir,"testvol_4x4x4b.h5")
    
    testvol = np.array([[[0.1761602 , 0.6701295 , 0.13151232, 0.95726678],
        [0.4795476 , 0.48114134, 0.0410548 , 0.29893265],
        [0.49127266, 0.70298447, 0.42751211, 0.08101552],
        [0.73805652, 0.83111601, 0.36852477, 0.38732476]],

       [[0.2847222 , 0.96054574, 0.25430756, 0.35403861],
        [0.54439093, 0.65897414, 0.1959487 , 0.90714872],
        [0.84462152, 0.90754182, 0.02455657, 0.26180662],
        [0.1711208 , 0.40122666, 0.54562598, 0.01419861]],

       [[0.59280376, 0.42706895, 0.86637913, 0.87831645],
        [0.57991401, 0.31989204, 0.85869799, 0.6333411 ],
        [0.21539274, 0.63780214, 0.64204493, 0.74425482],
        [0.1903691 , 0.81962537, 0.31774673, 0.34812628]],

       [[0.40880077, 0.595773  , 0.28856063, 0.19316746],
        [0.03195766, 0.62475541, 0.50762591, 0.34700798],
        [0.98913461, 0.07883111, 0.96534233, 0.57697606],
        [0.71496714, 0.70764578, 0.92294417, 0.91300531]]])
    
    
    with h5py.File(map_fullpath,  'w') as hf:
        hf.create_dataset("data",  data=testvol)

    tmp_ws_name = "testworkspace_tmp" 
    
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
 
    # add data to workspace
    survos.run_command('workspace', 'add_data', uri=None, workspace=tmp_ws_name,
                    data_fname=map_fullpath,
                    dtype='float32')

   
    DataModel.g.current_workspace =  tmp_ws_name

    return DataModel

class Tests(object):
    def test_feature_shape(self, datamodel):
        DataModel = datamodel
        src = DataModel.g.dataset_uri('__data__', None)
        dst = DataModel.g.dataset_uri('001_gaussian_blur', group='features')

        survos.run_command('features', 'gaussian_blur', uri=None, 
                        src=src,
                        dst=dst)

        with DatasetManager(src, out=dst, dtype='float32', fillvalue=0) as DM:
            print(DM.sources[0].shape)
            src_dataset = DM.sources[0]
            dst_dataset = DM.out
            src_arr = src_dataset[:]
            dst_arr = dst_dataset[:]

        assert dst_arr.shape == (4,4,4)
        
        assert_allclose(src_arr, np.array([[[0.16612536, 0.67279379, 0.12032965, 0.9673129 ],
        [0.47731235, 0.47894706, 0.02754662, 0.29205408],
        [0.48933884, 0.70649341, 0.42393911, 0.06853466],
        [0.74246711, 0.837919  , 0.3634353 , 0.38271861]],

       [[0.27747831, 0.97067616, 0.24628176, 0.34857673],
        [0.54382269, 0.66135165, 0.18642259, 0.91590639],
        [0.85177172, 0.91630959, 0.01062425, 0.2539736 ],
        [0.1609564 , 0.3969779 , 0.54508949, 0.        ]],

       [[0.59348014, 0.42348456, 0.87408868, 0.88633289],
        [0.58025901, 0.3135523 , 0.86621007, 0.63505962],
        [0.2063665 , 0.63963535, 0.64398722, 0.74882475],
        [0.18069954, 0.82613296, 0.31135184, 0.3425124 ]],

       [[0.40474673, 0.59652571, 0.28141542, 0.18356984],
        [0.01821561, 0.62625321, 0.5061125 , 0.34136535],
        [1.        , 0.0662941 , 0.97559606, 0.57724553],
        [0.71878414, 0.71127456, 0.93210791, 0.92191354]]]))

        assert_allclose(dst_arr, np.array([[[0.1334253 , 0.18309763, 0.18128583, 0.13541691],
        [0.19046707, 0.25114629, 0.2388469 , 0.16883893],
        [0.19799377, 0.25374037, 0.23332517, 0.15631635],
        [0.14958453, 0.18756688, 0.16748451, 0.10621871]],

       [[0.18743832, 0.26002762, 0.26079947, 0.1948524 ],
        [0.26292312, 0.35610881, 0.34901825, 0.25268969],
        [0.26981699, 0.3587358 , 0.34435031, 0.24155034],
        [0.20132653, 0.2641314 , 0.24896044, 0.16910219]],

       [[0.1889769 , 0.26528233, 0.26963302, 0.20190974],
        [0.26367947, 0.36606592, 0.36840445, 0.27200189],
        [0.27032515, 0.3715018 , 0.36991027, 0.26886454],
        [0.20220719, 0.27610379, 0.27227667, 0.19469661]],

       [[0.13751902, 0.19435027, 0.19884795, 0.14815024],
        [0.19330847, 0.2737053 , 0.28118965, 0.21023472],
        [0.20052321, 0.28329667, 0.2909368 , 0.21732429],
        [0.15249521, 0.21546596, 0.22105172, 0.16474919]]]))
       
if __name__ == '__main__':
    pytest.main()

