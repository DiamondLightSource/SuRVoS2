from survos2.model import Workspace
from survos2.io import dataset_from_uri
from survos2.api.regions import get_slice
import os
import pytest
import h5py
import numpy as np

from survos2 import survos
from survos2.improc.utils import DatasetManager
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.model import DataModel

from loguru import logger
from torch.testing import assert_allclose


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

    with h5py.File(map_fullpath, "w") as hf:
        hf.create_dataset("data", data=testvol)

    tmp_ws_name = "testworkspace_tmp1"
    print(DataModel.g.CHROOT)

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
    survos.run_command(
        "workspace",
        "add_data",
        uri=None,
        workspace=tmp_ws_name,
        data_fname=map_fullpath,
        dtype="float32",
    )

    DataModel.g.current_workspace = tmp_ws_name

    return DataModel


class Tests(object):
    def test_feature_shape(self, datamodel):
        DataModel = datamodel
        src = DataModel.g.dataset_uri("__data__", None)
        dst = DataModel.g.dataset_uri("001_gaussian_blur", group="features")

        survos.run_command("features", "gaussian_blur", uri=None, src=src, dst=dst)

        with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
            print(DM.sources[0].shape)
            src_dataset = DM.sources[0]
            dst_dataset = DM.out
            src_arr = src_dataset[:]
            dst_arr = dst_dataset[:]

        assert dst_arr.shape == (4, 4, 4)
        assert np.max(dst_arr) <= 1.0
        assert np.min(dst_arr) >= 0.0


if __name__ == "__main__":
    pytest.main()
