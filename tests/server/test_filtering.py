import numpy as np
from torch.testing import assert_allclose
import pytest

from survos2.server.config import cfg
from survos2.server.filtering import (
    gaussian_blur_kornia,
    simple_invert,
    laplacian,
    spatial_gradient_3d,
)
from survos2.improc.features import tvdenoising3d
from survos2.server.features import generate_features
from skimage.data import binary_blobs


def test_feature_generation():

    img_vol = np.zeros((5, 5, 5))
    img_vol[2, 2, 2] = 1.0

    filter_cfg = {
        "filter1": {
            "plugin": "features",
            "feature": "gaussian",
            "params": {"sigma": (1,1,1)},
        },
        "filter2": {
            "plugin": "features",
            "feature": "gaussian",
            "params": {"sigma": (2,2,2)},
        },
        "filter3": {
            "plugin": "features",
            "feature": "tvdenoising3d",
            "params": {"lamda": 3.0},
        },
        "filter4": {
            "plugin": "features",
            "feature": "laplacian",
            "params": {"kernel_size": (3,3,3)},
        },
        "filter5": {
            "plugin": "features",
            "feature": "gradient",
            "params": {"sigma": (3,3,3)},
        },
    }

    feature_params = [
        [gaussian_blur_kornia, filter_cfg["filter1"]["params"]],
        [gaussian_blur_kornia, filter_cfg["filter2"]["params"]],
        [laplacian, filter_cfg["filter4"]["params"]],
    ]

    num_feature_params = len(feature_params)

    roi_crop = [0, img_vol.shape[0], 0, img_vol.shape[1], 0, img_vol.shape[2]]

    sr_feat = generate_features(img_vol, feature_params, roi_crop, 1.0)

    num_generated_features = len(sr_feat.filtered_layers)

    assert num_feature_params == num_generated_features

    generated_feature_shapes = [layer.shape for layer in sr_feat.filtered_layers]

    assert generated_feature_shapes == [
        img_vol.shape for i in range(num_generated_features)
    ]


if __name__ == "__main__":
    pytest.main()
