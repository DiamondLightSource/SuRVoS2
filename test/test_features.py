import os
import pytest
import h5py
import numpy as np 

from survos2 import survos
from survos2.improc.utils import DatasetManager
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.frontend.control import DataModel


from torch.testing import assert_allclose

@pytest.fixture(scope="session")
def datamodel():
    test_datadir = "D:\\datasets"
    test_workspace_name = "testvol_29"  # need to set a new workspace name...
    
    # make test vol
    map_fullpath = os.path.join(test_datadir,"testvol_4x4x4b.h5")
    #testvol = np.random.random((4,4,4))# astype(np.uint8)
    #with h5py.File(map_fullpath,  'w') as hf:
    #    hf.create_dataset("data",  data=testvol)

    # create new workspace
    survos.run_command("workspace", "create", uri=None, workspace=test_workspace_name)
        
    # add data to workspace
    survos.run_command('workspace', 'add_data', uri=None, workspace=test_workspace_name,
                    data_fname=map_fullpath,
                    dtype='float32')

    ## add dataset to workspace
    # run gaussian_blur    

    DataModel.g.current_workspace = test_workspace_name

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
        
        assert_allclose(src_arr, np.array([[[1.        , 0.82068106, 0.42073764, 0.51587235],
        [0.85844268, 0.093271  , 0.81512804, 0.1409235 ],
        [0.40732418, 0.49330192, 0.30259749, 0.80359946],
        [0.68093999, 0.74056308, 0.96625073, 0.48458495]],

       [[0.45676518, 0.1774164 , 0.24397241, 0.95134383],
        [0.41796174, 0.23618559, 0.13389243, 0.04459196],
        [0.98537844, 0.69071347, 0.28030243, 0.72146098],
        [0.00538105, 0.32259491, 0.80230278, 0.75218765]],

       [[0.05712534, 0.93959021, 0.23600161, 0.77016731],
        [0.36986052, 0.9944051 , 0.90327656, 0.71696671],
        [0.22365138, 0.48614203, 0.3829062 , 0.74210897],
        [0.18544236, 0.43265719, 0.78338101, 0.71556586]],

       [[0.54682799, 0.3201304 , 0.25386172, 0.93151697],
        [0.66178861, 0.02124417, 0.21574268, 0.93328711],
        [0.        , 0.00671551, 0.06041111, 0.14144617],
        [0.06327323, 0.04650178, 0.92045145, 0.36577849]]]))

        assert_allclose(dst_arr, np.array([[[0.1490646 , 0.19287831, 0.18641028, 0.13451761],
        [0.19899806, 0.26474053, 0.26180401, 0.19193436],
        [0.19645271, 0.27019641, 0.27425525, 0.20536835],
        [0.14229402, 0.2043739 , 0.21399249, 0.16382854]],

       [[0.18283825, 0.2482031 , 0.25040337, 0.19064094],
        [0.24286833, 0.33724022, 0.34605712, 0.2650333 ],
        [0.23648183, 0.3385703 , 0.35530755, 0.27574381],
        [0.16772471, 0.25130859, 0.27207845, 0.21492796]],

       [[0.16764127, 0.2381117 , 0.24947754, 0.19824161],
        [0.21886861, 0.31741875, 0.33751971, 0.26857039],
        [0.20801128, 0.31110197, 0.33789775, 0.27112889],
        [0.14358498, 0.22562897, 0.25300926, 0.20584027]],

       [[0.11026709, 0.16355111, 0.17833549, 0.14834332],
        [0.13923284, 0.2115462 , 0.23430192, 0.19469883],
        [0.12705612, 0.20041816, 0.22716931, 0.18964946],
        [0.08399598, 0.14100567, 0.16565712, 0.13964762]]]))
       
if __name__ == '__main__':
    pytest.main()

