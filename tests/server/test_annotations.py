import os

import h5py
import numpy as np
import pytest
from torch.testing import assert_allclose
from loguru import logger
from skimage.data import binary_blobs


import survos
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.entity.pipeline import Patch
import survos2.frontend.control
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.entity.pipeline import run_workflow
from survos2.server.state import cfg
from survos2.server.superseg import sr_predict
from survos2.api.superregions import supervoxels
from survos2.server.superseg import sr_predict
from survos2.frontend.nb_utils import view_dataset

tmp_ws_name = "testworkspace_tmp2"
@pytest.fixture(scope="session")
def datamodel():
    # make test vol
    map_fullpath = os.path.join("./tmp/testvol_4x4x4b.h5")
    testvol = np.array(
        [
            [
                [0.1761602, 0.6701295, 0.13151232, 0.95726678],
                [0.4795476, 0.48114134, 0.0410548, 0.29893265],
                [0.49127266, 0.70298447, 0.42751211, 0.08101552],
                [0.73805652, 0.83111601, 0.36852477, 0.38732476],
            ],
            [
                [0.2847222, 0.96054574, 0.25430756, 0.35403861],
                [0.54439093, 0.65897414, 0.1959487, 0.90714872],
                [0.84462152, 0.90754182, 0.02455657, 0.26180662],
                [0.1711208, 0.40122666, 0.54562598, 0.01419861],
            ],
            [
                [0.59280376, 0.42706895, 0.86637913, 0.87831645],
                [0.57991401, 0.31989204, 0.85869799, 0.6333411],
                [0.21539274, 0.63780214, 0.64204493, 0.74425482],
                [0.1903691, 0.81962537, 0.31774673, 0.34812628],
            ],
            [
                [0.40880077, 0.595773, 0.28856063, 0.19316746],
                [0.03195766, 0.62475541, 0.50762591, 0.34700798],
                [0.98913461, 0.07883111, 0.96534233, 0.57697606],
                [0.71496714, 0.70764578, 0.92294417, 0.91300531],
            ],
        ]
    )
    # testvol = np.ones((4,4,4)).astype(np.float32) / 2.0
    # print(testvol)

    with h5py.File(map_fullpath, "w") as hf:
        hf.create_dataset("data", data=testvol)

    print(DataModel.g.CHROOT)
    
    test_anno = np.array([[[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]]], dtype=np.uint32)

    result = survos.run_command("workspace", "get", uri=None, workspace=tmp_ws_name)

    # create temp workspace
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
    survos.run_command(
        "workspace",
        "add_data",
        uri=None,
        workspace=tmp_ws_name,
        data_fname=map_fullpath,
        dtype="float32",
    )

    survos.run_command(
        "features", "create", uri=None, workspace=tmp_ws_name, feature_type="raw"
    )

    DataModel.g.current_workspace = tmp_ws_name

    # add level to workspace and add a red label
    result = survos.run_command('annotations', 'add_level', uri=None, workspace=tmp_ws_name)
    params = dict(level="001_level")
    result = survos.run_command('annotations', 'add_label', uri=None, workspace=tmp_ws_name, **params)
    label = dict(
                idx=2,
                name="Labelname",
                color="#FF0000",
                visible=True,
            )
    result = survos.run_command('annotations', 'update_label', uri=None, workspace=tmp_ws_name, **params, **label)
    
    # set the array to test_anno above
    src = DataModel.g.dataset_uri('001_level', group="annotations")
    survos.run_command('annotations', 'set_volume', uri=None, workspace=tmp_ws_name, src=src, vol_array=test_anno)

    return DataModel


class Tests(object):
    def test_setup(self, datamodel):
        result = survos.run_command('annotations', 'get_levels', uri=None, workspace=tmp_ws_name)
        assert result[0][0]['kind'] == 'level'
        assert result[0][0]['name'] == '001 Level'
        assert result[0][0]['labels']['2']['idx'] == 2

        result = survos.run_command('features', 'existing', uri=None, workspace=tmp_ws_name,
                    dtype='float32')
        assert result[0]['001_raw']['id'] == '001_raw'

    def test_annotations(self, datamodel):
        result = survos.run_command('annotations', 'add_level', uri=None, workspace=tmp_ws_name)
        assert 'name' in result[0]
        result = survos.run_command('annotations', 'get_levels', uri=None, workspace=tmp_ws_name)
        assert 'name' in result[0][0]

if __name__ == "__main__":
    pytest.main()

