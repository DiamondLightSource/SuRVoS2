
from survos2.server.filtering.blur import tvdenoise_kornia
import numpy as np
from torch.testing import assert_allclose
import pytest
import numpy as np
from survos2.entity.entities import (make_bounding_vols, make_entity_mask, make_entity_df, offset_points, uncrop_pad)
from survos2.entity.sampler import centroid_to_bvol
import pandas as pd

def test_make_entity_df():
    points = np.array([[10,10,10,0],[10,20,20,0],[10,30,30,0],[10,40,40,0],[10,50,50,0]])
    result = make_entity_df(points)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5,4)
    

def test_offset_points():
    points = np.array([[10,10,10,0],[10,20,20,0],[10,30,30,0],[10,40,40,0],[10,50,50,0]])
    result = offset_points(points, (10,10,10))
    assert result[0][0] == points[0][0] - 10


def test_uncrop_pad():
    img_vol = np.ones((64,64,64))
    result = uncrop_pad(img_vol, (96,96,96), (16,80,16,80,16,80))
    assert result.shape == (96,96,96)
    assert result[0][0][0] == 0.0
    assert result[32][32][32] == 1.0

def test_make_entity_mask():
    img_vol = np.ones((128,128,128))
    points = np.array([[32,32,32,0],[32,42,42,0],[32,52,52,0],[32,62,62,0],[32,72,72,0]])
    result = centroid_to_bvol(points)
    result = make_entity_mask(img_vol, points,bvol_dim=(4,4,4))
    # returns a padded volume
    assert result[0].shape == (136,136,136)
    