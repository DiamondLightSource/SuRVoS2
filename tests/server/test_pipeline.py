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

from survos2.api.regions import supervoxels
from survos2.server.superseg import sr_predict
from survos2.frontend.nb_utils import view_dataset


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

    print(DataModel.g.CHROOT)
    tmp_ws_name = "testworkspace_tmp1"

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
    def test_superregions_shape(self, datamodel):
        DataModel = datamodel
        src = DataModel.g.dataset_uri("__data__", None)
        dst = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
        result = survos.run_command(
            "features", "gaussian_blur", uri=None, src=src, dst=dst
        )
        result = survos.run_command("regions", "create", uri=None, workspace=DataModel.g.current_workspace)
        assert len(result)==2
        assert result[0]['kind'] == 'supervoxels'
        features_src = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
        dst = DataModel.g.dataset_uri("001_regions", group="regions")

        n_segments=8
        result = supervoxels(
            features_src,
            dst,
            n_segments=n_segments,
            compactness=1,
            spacing=[1, 1, 1],
            multichannel=False,
            enforce_connectivity=False,
        )

        assert result['n_segments'] == n_segments
    

        with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            dst_dataset = DM.out
            src_arr = src_dataset[:]
            dst_arr = dst_dataset[:]

        assert dst_arr.shape == src_arr.shape
        assert len(np.unique(dst_arr)) == n_segments

    def test_sr_predict_shape(self, datamodel):
        DataModel = datamodel
        src = DataModel.g.dataset_uri("__data__", None)
        dst = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
        result = survos.run_command(
            "features", "gaussian_blur", uri=None, src=src, dst=dst
        )

        with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            dst_dataset = DM.out
            src_arr = src_dataset[:]
            gblur_arr = dst_dataset[:]

        result = survos.run_command("regions", "create", uri=None)
        features_src = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
        dst = DataModel.g.dataset_uri("001_regions", group="regions")

        result = supervoxels(
            features_src,
            dst,
            n_segments=8,
            compactness=0.5,
            spacing=[1, 1, 1],
            multichannel=False,
            enforce_connectivity=False,
        )
        with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            dst_dataset = DM.out
            src_arr = src_dataset[:]
            dst_arr = dst_dataset[:]
        
        superseg_cfg = cfg.pipeline
        superseg_cfg["type"] = "rf"
        superseg_cfg["predict_params"]["clf"] = "Ensemble"

        refine = False
        lam = (1.0,)

        anno_arr = np.ones_like(dst_arr)
        anno_arr[2:4,2:4,2:4] = 2
        feature_arr = view_dataset("001_gaussian_blur", "features", 3)
        segmentation = sr_predict(
            dst_arr,
            anno_arr,
            [feature_arr, gblur_arr],
            None,
            superseg_cfg,
            refine,
            lam,
        )
        
        print(segmentation)
        print(len(np.unique(segmentation)))
        assert segmentation.shape == dst_arr.shape
        #assert len(np.unique(segmentation)) == 2

    def test_feature_generation(self, datamodel):
        DataModel = datamodel

        src = DataModel.g.dataset_uri("__data__", None)
        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            raw_arr = src_dataset[:]

        random_blobs = binary_blobs(length=max(raw_arr.shape), n_dim=3)
        random_blobs_anno = np.zeros_like(raw_arr)
        random_blobs_anno[
            0 : raw_arr.shape[0], 0 : raw_arr.shape[1], 0 : raw_arr.shape[2]
        ] = random_blobs[
            0 : raw_arr.shape[0], 0 : raw_arr.shape[1], 0 : raw_arr.shape[2]
        ]

        result = survos.run_command(
            "annotations",
            "add_level",
            uri=None,
            workspace=DataModel.g.current_workspace,
        )
        assert "id" in result[0]

        level_id = result[0]["id"]
        label_values = np.unique(random_blobs_anno)

        for v in label_values:
            params = dict(
                level=level_id,
                idx=int(v),
                name=str(v),
                color="#11FF11",
                workspace=DataModel.g.current_workspace,
            )
        label_result = survos.run_command("annotations", "add_label", **params)

        dst = DataModel.g.dataset_uri(level_id, group="annotations")
        with DatasetManager(dst, out=dst, dtype="uint32", fillvalue=0) as DM:
            DM.out[:] = random_blobs_anno

    def test_objects(self, datamodel):
        DataModel = datamodel

        # add data to workspace
        result = survos.run_command(
            "objects",
            "create",
            uri=None,
            workspace=DataModel.g.current_workspace,
            fullname="test.csv",
        )
        assert result[0]["id"] == "001_points"

        result = survos.run_command(
            "objects",
            "create",
            uri=None,
            workspace=DataModel.g.current_workspace,
            fullname="test.csv",
        )

        assert result[0]["id"] == "002_points"

        result = survos.run_command('objects', 'existing', uri=None, workspace=DataModel.g.current_workspace,dtype='float32')
        
        assert len(result[0]) == 2

    def test_analyzers(self, datamodel):
        DataModel = datamodel

        # add data to workspace
        result = survos.run_command(
            "analyzer", "create", uri=None, workspace=DataModel.g.current_workspace
        )
        assert result[0]["id"] == "001_image_stats"

        result = survos.run_command('analyzer', 'create', uri=None, workspace=DataModel.g.current_workspace, order=5)
        assert result[0]["id"] == "002_label_splitter"

        result = survos.run_command('analyzer', 'existing', uri=None, workspace=DataModel.g.current_workspace,
                  dtype='float32')

        print(result)
        assert len(result[0])==2


if __name__ == "__main__":
    pytest.main()
