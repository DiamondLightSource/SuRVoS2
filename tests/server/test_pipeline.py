import os
import h5py
import numpy as np
import pytest
from torch.testing import assert_allclose
import survos2.frontend.control
from survos2.frontend.control import Launcher
from survos2.server.pipeline import Patch

import survos2.frontend.control
from survos2.frontend.control import Launcher
import survos2.frontend.control
from survos2.model import DataModel
from survos2.improc.utils import DatasetManager
from survos2.server.pipeline import run_workflow
from survos2.server.state import cfg
from survos2.server.superseg import sr_predict


@pytest.mark.skip(reason="todo")
def test_sr_predict():
    workspace_name = "test_workspace_"
    DataModel.g.current_workspace = workspace_name
    DataModel.g.current_session = "default"

    # get anno
    src = DataModel.g.dataset_uri(anno_id, group="annotations")
    with DatasetManager(src, out=None, dtype="uint16", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        anno_image = src_dataset[:] & 15

    # get superregions
    src = DataModel.g.dataset_uri(region_id, group="regions")
    with DatasetManager(src, out=None, dtype="uint32", fillvalue=0) as DM:
        src_dataset = DM.sources[0]
        supervoxel_image = src_dataset[:]

    # get features
    features = []

    for feature_id in feature_ids:
        src = DataModel.g.dataset_uri(feature_id, group="features")

        with DatasetManager(src, out=None, dtype="float32", fillvalue=0) as DM:
            src_dataset = DM.sources[0]
            features.append(src_dataset[:])

    superseg_cfg = cfg.pipeline
    superseg_cfg["type"] = classifier_type
    superseg_cfg["predict_params"]["proj"] = projection_type

    anno_id = "001_level"
    region_id = "001_supervoxels"
    feature_ids = ["001_gaussian_blur", "001_raw"]
    classifier_type = "rf"
    projection_type = None
    refine = False
    lam = (1.0,)
    num_components = 0

    segmentation = sr_predict(
        supervoxel_image,
        anno_image,
        features,
        superseg_cfg,
        refine,
        lam,
        num_components,
    )
    assert segmentation.shape == anno_image.shape


@pytest.mark.skip(reason="todo")
def test_mask_pipeline(p: Patch):
    p = make_masks(p)
    p.image_layers["result"] = p.image_layers["total_mask"]
    return p


@pytest.mark.skip(reason="todo")
def test_survos_pipeline(self, p: Patch):
    p = predict_sr(s)
    return p


@pytest.mark.skip(reason="todo")
def test_superregion_pipeline(self, p: Patch):
    p = make_masks(s)
    p = make_features(s)
    p = make_sr(s)

    p.image_layers["result"] = s.superregions.supervoxel_vol
    return p


@pytest.mark.skip(reason="todo")
def test_prediction_pipeline(self, p: Patch):
    p = make_masks(s)
    p = make_features(s)
    p = make_sr(s)
    # p = do_acwe(p)
    p = predict_sr(s)
    p.image_layers["result"] = p.image_layers["prediction"]
    return p
