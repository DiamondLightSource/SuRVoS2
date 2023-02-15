import pytest
from httpx import AsyncClient
from main import app
import os
import h5py
import numpy as np
from loguru import logger

from survos2.model import DataModel
from survos2.api.workspace import create as create_workspace
from survos2.api.workspace import add_dataset, add_data, delete, get
from survos2.improc.utils import DatasetManager

tmp_ws_name = "testworkspace_tmp1"


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

    result = get(workspace=tmp_ws_name)
    if not isinstance(result, bool):
        logger.debug("tmp exists, deleting and recreating")
        delete(workspace=tmp_ws_name)
        logger.debug("workspace deleted")

    create_workspace(workspace=tmp_ws_name)
    logger.debug("workspace recreated")
    add_data(workspace=tmp_ws_name, data_fname=map_fullpath)
    DataModel.g.current_workspace = tmp_ws_name

    return DataModel


class Tests(object):
    @pytest.mark.asyncio
    async def test_superregions(self, datamodel):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            DataModel = datamodel
            src = DataModel.g.dataset_uri("__data__", None)
            dst = DataModel.g.dataset_uri("001_gaussian_blur", group="features")

            params = {"workspace": "testworkspace_tmp1", "src": src, "dst": dst, "sigma": 1}
            response = await ac.get("/features/gaussian_blur", params=params)
            assert (
                response.text
                == '{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"001_gaussian_blur"}'
            )

            dst = DataModel.g.dataset_uri("002_gaussian_blur", group="features")
            params = {"workspace": "testworkspace_tmp1", "src": src, "dst": dst, "sigma": 1}
            response = await ac.get("/features/gaussian_blur", params=params)
            assert (
                response.text
                == '{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"002_gaussian_blur"}'
            )

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/features/existing", params=params)
            assert (
                response.text
                == '{"001_gaussian_blur":{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"001_gaussian_blur"},"002_gaussian_blur":{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"002_gaussian_blur"}}'
            )

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/superregions/create", params=params)
            assert (
                response.text
                == '{"name":"001 Supervoxels","kind":"supervoxels","id":"001_supervoxels"}'
            )

            src = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
            dst = DataModel.g.dataset_uri("001_supervoxels", group="supervoxels")

            params = {
                "src": src,
                "dst": dst,
                "mask_id": "None",
                "n_segments": 10,
                "compactness": 20,
                "spacing": [1, 1, 1],
                "multichannel": False,
                "enforce_connectivity": False,
                "out_dtype": "int",
                "zero_parameter": False,
                "max_num_iter": 10,
            }

            response = await ac.get("/superregions/supervoxels", params=params)
            assert (
                response.text
                == '{"kind":"supervoxels","name":"supervoxels","mask_id":"None","n_segments":10,"compactness":20.0,"multichannel":false,"enforce_connectivity":false,"out_dtype":"int","zero_parameter":false,"max_num_iter":10,"spacing":[1,1,1],"source":"001_gaussian_blur","id":"001_supervoxels"}'
            )

            with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
                src_dataset = DM.sources[0]
                dst_dataset = DM.out
                src_arr = src_dataset[:]
                supervoxel_arr = dst_dataset[:]

            assert src_arr.shape == supervoxel_arr.shape
            assert (
                np.max(supervoxel_arr).astype(np.int) == 8
            )  # 1 less than max num of segments, indexed from 0

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/superregions/create", params=params)
            assert (
                response.text
                == '{"name":"002 Supervoxels","kind":"supervoxels","id":"002_supervoxels"}'
            )

            src = DataModel.g.dataset_uri("002_gaussian_blur", group="features")
            dst = DataModel.g.dataset_uri("002_supervoxels", group="supervoxels")

            params = {
                "src": src,
                "dst": dst,
                "mask_id": "None",
                "n_segments": 10,
                "compactness": 20,
                "spacing": [1, 1, 1],
                "multichannel": False,
                "enforce_connectivity": False,
                "out_dtype": "int",
                "zero_parameter": False,
                "max_num_iter": 10,
            }

            response = await ac.get("/superregions/supervoxels", params=params)
            assert (
                response.text
                == '{"kind":"supervoxels","name":"supervoxels","mask_id":"None","n_segments":10,"compactness":20.0,"multichannel":false,"enforce_connectivity":false,"out_dtype":"int","zero_parameter":false,"max_num_iter":10,"spacing":[1,1,1],"source":"002_gaussian_blur","id":"002_supervoxels"}'
            )

            with DatasetManager(src, out=dst, dtype="float32", fillvalue=0) as DM:
                dst_dataset = DM.out
                supervoxel_arr2 = dst_dataset[:]

            assert np.allclose(supervoxel_arr, supervoxel_arr2)

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/superregions/existing", params=params)
            assert (
                response.text
                == '{"001_supervoxels":{"kind":"supervoxels","name":"001 Supervoxels","id":"001_supervoxels"},"002_supervoxels":{"kind":"supervoxels","name":"002 Supervoxels","id":"002_supervoxels"}}'
            )

            # CLEANUP
            params = {"workspace": "testworkspace_tmp1", "feature_id": "001_gaussian_blur"}
            response = await ac.get("/features/remove", params=params)

            params = {"workspace": "testworkspace_tmp1", "feature_id": "002_gaussian_blur"}
            response = await ac.get("/features/remove", params=params)

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/features/existing", params=params)
            assert response.text == "{}"

            params = {"workspace": "testworkspace_tmp1", "region_id": "001_supervoxels"}
            response = await ac.get("/superregions/remove", params=params)

            params = {"workspace": "testworkspace_tmp1", "region_id": "002_supervoxels"}
            response = await ac.get("/superregions/remove", params=params)

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/superregions/existing", params=params)
            assert response.text == "{}"

    @pytest.mark.asyncio
    async def test_superregion_segment(self, datamodel):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            DataModel = datamodel

            # Make levels
            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/annotations/get_levels", params=params)
            assert response.text == "[]"

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/annotations/add_level", params=params)

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/annotations/get_levels", params=params)
            assert (
                response.text
                == '[{"kind":"level","modified":[0],"name":"001 Level","id":"001_level"}]'
            )

            # Make features

            src = DataModel.g.dataset_uri("__data__", None)
            dst = DataModel.g.dataset_uri("001_gaussian_blur", group="features")

            params = {"workspace": "testworkspace_tmp1", "src": src, "dst": dst, "sigma": 1}
            response = await ac.get("/features/gaussian_blur", params=params)
            assert (
                response.text
                == '{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"001_gaussian_blur"}'
            )

            dst = DataModel.g.dataset_uri("002_gaussian_blur", group="features")
            params = {"workspace": "testworkspace_tmp1", "src": src, "dst": dst, "sigma": 1}
            response = await ac.get("/features/gaussian_blur", params=params)
            assert (
                response.text
                == '{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"002_gaussian_blur"}'
            )

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/features/existing", params=params)
            assert (
                response.text
                == '{"001_gaussian_blur":{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"001_gaussian_blur"},"002_gaussian_blur":{"kind":"gaussian_blur","name":"gaussian_blur","sigma":[1],"source":"__data__","id":"002_gaussian_blur"}}'
            )

            # Make superregions

            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/superregions/create", params=params)
            assert (
                response.text
                == '{"name":"001 Supervoxels","kind":"supervoxels","id":"001_supervoxels"}'
            )

            src = DataModel.g.dataset_uri("001_gaussian_blur", group="features")
            dst = DataModel.g.dataset_uri("001_supervoxels", group="supervoxels")

            params = {
                "src": src,
                "dst": dst,
                "mask_id": "None",
                "n_segments": 10,
                "compactness": 20,
                "spacing": [1, 1, 1],
                "multichannel": False,
                "enforce_connectivity": False,
                "out_dtype": "int",
                "zero_parameter": False,
                "max_num_iter": 10,
            }

            response = await ac.get("/superregions/supervoxels", params=params)
            assert (
                response.text
                == '{"compactness":20.0,"enforce_connectivity":false,"kind":"supervoxels","mask_id":"None","max_num_iter":10,"multichannel":false,"n_segments":10,"name":"supervoxels","out_dtype":"int","source":"001_gaussian_blur","spacing":[1,1,1],"zero_parameter":false,"id":"001_supervoxels"}'
            )

            params = {"workspace": "testworkspace_tmp1", "pipeline_type": "superegion_segment"}
            response = await ac.get("/pipelines/create", params=params)
            assert (
                response.text
                == '{"name":"001 Superegion Segment","kind":"superegion_segment","id":"001_superegion_segment"}'
            )

            src = DataModel.g.dataset_uri("001_superegion_segment", group="pipelines")
            dst = DataModel.g.dataset_uri("001_superegion_segment", group="pipelines")

            classifier_params = {
                "clf": "ensemble",
                "type": 0,
                "n_estimators": 100,
                "max_depth": 20,
                "learning_rate": 1.0,
                "subsample": 1.0,
                "n_jobs": 10,
            }

            body = {
                "src": src,
                "dst": dst,
                "workspace": "testworkspace_tmp1",
                "anno_id": "001_level",
                "constrain_mask": "None",
                "region_id": "001_supervoxels",
                "lam": 1,
                "refine": False,
                "classifier_type": "Ensemble",
                "projection_type": "None",
                "confidence": False,
                "classifier_params": classifier_params,
                "feature_ids": ["001_gaussian_blur"],
            }

            # response = await ac.post("/pipelines/superregion_segment", **body)
            # assert response.text == '{}'
            # Not working ... 'Method Not Allowed'

            # CLEANUP
            params = {"workspace": "testworkspace_tmp1", "feature_id": "001_gaussian_blur"}
            response = await ac.get("/features/remove", params=params)

            params = {"workspace": "testworkspace_tmp1", "feature_id": "002_gaussian_blur"}
            response = await ac.get("/features/remove", params=params)

    @pytest.mark.asyncio
    async def test_rasterize_points(self, datamodel):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            DataModel = datamodel

            params = {"workspace": "testworkspace_tmp1", "fullname": ".//tests//server//test.csv"}
            response = await ac.get("/objects/create", params=params)
            assert (
                response.text
                == '{"name":"001 Points","kind":"points","fullname":".//tests//server//test.csv","basename":"test.csv","id":"001_points"}'
            )

            dst = DataModel.g.dataset_uri("001_points", group="objects")
            params = {
                "workspace": "testworkspace_tmp1",
                "fullname": ".//tests//server//test.csv",
                "dst": dst,
                "scale": 1.0,
            }

            response = await ac.get("/objects/points", params=params)

            import ast

            response_dict = ast.literal_eval(response.text)
            assert response_dict["basename"] == "test.csv"

            params = {"workspace": "testworkspace_tmp1", "feature_type": "raw"}
            response = await ac.get("/features/create", params=params)

            params = {"workspace": "testworkspace_tmp1", "pipeline_type": "rasterize_points"}
            response = await ac.get("/pipelines/create", params=params)
            assert (
                response.text
                == '{"name":"002 Rasterize Points","kind":"rasterize_points","id":"002_rasterize_points"}'
            )

            dst = DataModel.g.dataset_uri("002_rasterize_points", group="pipelines")
            params = {
                "src": dst,
                "dst": dst,
                "acwe": False,
                "balloon": 0.0,
                "feature_id": "001_raw",
                "iterations": 0,
                "object_id": "objects/001_points",
                "selected_class": 0,
                "size": [5.0, 5.0, 5.0],
                "smoothing": 0,
                "threshold": 0.0,
                "workspace": "huntcrop2",
            }

            response = await ac.get("/pipelines/rasterize_points", params=params)
            assert (
                response.text
                == '{"kind":"rasterize_points","name":"002 Rasterize Points","workspace":"huntcrop2","feature_id":"001_raw","object_id":"objects/001_points","acwe":false,"balloon":0.0,"threshold":0.0,"iterations":0,"smoothing":0,"selected_class":0,"size":[5.0,5.0,5.0],"source":"002_rasterize_points","id":"002_rasterize_points"}'
            )

    @pytest.mark.asyncio
    async def test_features(self, datamodel):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            params = {"workspace": "testworkspace_tmp1"}
            response = await ac.get("/features/existing", params=params)
            assert response.text == '{"001_raw":{"kind":"raw","name":"001 Raw","id":"001_raw"}}'


if __name__ == "__main__":
    pytest.main()
