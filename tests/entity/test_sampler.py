from survos2.server.filtering.blur import tvdenoise_kornia
import numpy as np
from torch.testing import assert_allclose
import pytest
import numpy as np
from survos2.entity.sampler import (sample_bvol, generate_random_points_in_volume, centroid_to_bvol, offset_points, grid_of_points, sample_marked_patches)

def test_sample_bvol():        
    img_vol = np.ones((10,10,10))
    result = sample_bvol(img_vol, (2,8,2,8,2,8)) 
    result.shape
    assert(result.shape == (6,6,6))

def test_generate_random_points_in_volume():
    img_vol = np.ones((10,10,10))
    result = generate_random_points_in_volume(img_vol, 10, border=(0,0,0))
    assert result.shape[0] == 10 and result.shape[1] == 4

def test_centroid_to_bvol():
    points = np.array([[10,10,10,0],[10,20,20,0],[10,30,30,0],[10,40,40,0],[10,50,50,0]])
    result = centroid_to_bvol(points)
    assert result.shape == (5,6)

def test_offset_points():
    points = np.array([[10,10,10,0],[10,20,20,0],[10,30,30,0],[10,40,40,0],[10,50,50,0]])
    result = offset_points(points, (10,10,10))
    assert result[0][0] == points[0][0] + 10

def test_grid_of_points():
    img_vol = np.ones((32,32,32))
    result = grid_of_points(img_vol, (4,4,4), (2,2,2))
    assert result[0][0] == 4
    assert result.shape[0] == 8
    img_vol = np.ones((32,32,32))
    result = grid_of_points(img_vol, (4,4,4), (4,4,4))
    assert result.shape[0] == 64


def test_sample_marked_patches():
    img_vol = np.ones((64,64,64))
    pts = grid_of_points(img_vol, (4,4,4), (32,32,32))
    img_volume = np.random.random((64,64,64))
    #padded_anno = (np.random((32,32,32)) > 0.5) * 1.0
    locs = np.array([[10,10,10,0],[10,20,20,0],[10,30,30,0],[10,40,40,0],[10,50,50,0]])
    result = sample_marked_patches(img_volume, locs, pts, patch_size=(4, 4, 4))
    assert result.vols.shape == (5,4,4,4)
    result.vols_pts.shape[0] == 5
    result.vols_pts[0][0] == [1,1,1,0]