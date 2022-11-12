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
    DataModel.g.current_workspace = tmp_ws_name
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

    test_anno = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
        ],
        dtype=np.uint32,
    )

    testvol = np.random.random((32, 32, 32))  # astype(np.uint8)

    test_anno = np.zeros_like(testvol)
    test_anno[8:12, 8:12, 8:12] = 1
    test_anno[8:12, 24:28, 24:28] = 1
    test_anno[8:12, 16:18, 16:18] = 1
    test_anno = test_anno.astype(np.uint)

    with h5py.File(map_fullpath, "w") as hf:
        hf.create_dataset("data", data=testvol)

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

    survos.run_command("features", "create", uri=None, workspace=tmp_ws_name, feature_type="raw")

    # add level to workspace and add a red label
    result = survos.run_command("annotations", "add_level", uri=None, workspace=tmp_ws_name)
    params = dict(level="001_level")
    result = survos.run_command(
        "annotations", "add_label", uri=None, workspace=tmp_ws_name, **params
    )
    label = dict(
        idx=1,
        name="Labelname",
        color="#FF0000",
        visible=True,
    )
    result = survos.run_command(
        "annotations", "update_label", uri=None, workspace=tmp_ws_name, **params, **label
    )

    # set the array to test_anno above
    src = DataModel.g.dataset_uri("001_level", group="annotations")
    survos.run_command(
        "annotations", "set_volume", uri=None, workspace=tmp_ws_name, src=src, vol_array=test_anno
    )

    return DataModel


class Tests(object):
    def test_label_splitter(self, datamodel):
        DataModel = datamodel
        src = DataModel.g.dataset_uri("001_level", group="annotations")
        r1, r2 = survos.run_command(
            "analyzer",
            "label_splitter",
            src=src,
            dst=src,
            workspace=tmp_ws_name,
            mode="3",
            pipelines_id="None",
            analyzers_id="None",
            annotations_id="001_level",
            feature_id="001_raw",
            split_ops={1: {"split_feature_index": 4, "split_op": 1, "split_threshold": 0.5}},
            background_label=0,
        )
        result_features, features_array, bvols = r1
        assert len(features_array[0]) == 20

    def test_point_generator(self, datamodel):
        src = DataModel.g.dataset_uri("001_level", group="annotations")

        r1, r2 = survos.run_command(
            "analyzer",
            "point_generator",
            src=src,
            dst=src,
            workspace=tmp_ws_name,
            bg_mask_id="None",
            num_before_masking=10,
        )
        assert len(r1) == 10

    def test_find_connected_components(self, datamodel):
        src = DataModel.g.dataset_uri("001_level", group="annotations")
        r1, r2 = survos.run_command(
            "analyzer",
            "find_connected_components",
            src=src,
            dst=src,
            workspace=tmp_ws_name,
            label_index=1,
            area_min=0,
            area_max=1e16,
            mode=3,
            pipelines_id="None",
            analyzers_id="None",
            annotations_id="001_level",
        )
        assert r1 == "bob"


if __name__ == "__main__":
    pytest.main()
