
import numpy as np
from survos2.entity.components import measure_components, filter_proposal_mask, filter_small_components, measure_regions


def test_measure_components():
    img = np.zeros((32,32,32))
    img[8:12,8:12,8:12] = 1
    img[8:12, 24:28,24:28] = 1
    result = measure_components(img)
    assert result.shape == (2,11)


def test_filter_proposal_mask():
    img = np.zeros((32,32,32))
    img[8:12,8:12,8:12] = 1
    img[8:12, 24:28,24:28] = 1
    result = filter_proposal_mask(img, num_erosions=0, num_dilations=3, num_medians=0)
    assert result.shape == (32,32,32)
    assert np.sum(result) > np.sum(img)

def test_filter_small_components():
    img = np.zeros((32,32,32))
    img[8:12,8:12,8:12] = 1
    img[8:12, 24:28,24:28] = 1
    img[8:12,16:18,16:18] = 1
    result= filter_small_components([img], min_component_size=16)[0]
    assert np.sum(result) < np.sum(img)

def test_measure_regions():
    img = np.zeros((32,32,32))
    img[8:12,8:12,8:12] = 1
    result = measure_regions([img.astype(np.uint32)])
    assert result[0].shape == (1,11)