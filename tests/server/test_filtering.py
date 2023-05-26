from survos2.server.filtering.blur import tvdenoise_kornia
import numpy as np
import pytest

from survos2.server.filtering import (
    gaussian_blur_kornia,
    simple_invert,
    laplacian,
    spatial_gradient_3d,
    ndimage_laplacian,
    hessian_eigvals_image,
    simple_invert,
    median,
    gamma_adjust,
    threshold,
    invert_threshold,
)

from survos2.server.features import generate_features


def test_feature_generation():
    img_vol = np.random.random((5, 5, 5))
    img_vol[2, 2, 2] = 1.0

    filter_cfg = {
        "gaussian1": {
            "plugin": "features",
            "feature": "gaussian",
            "params": {"sigma": (1, 1, 1)},
        },
        "gaussian2": {
            "plugin": "features",
            "feature": "gaussian",
            "params": {"sigma": (2, 2, 2)},
        },
        "tvdenoise": {
            "plugin": "features",
            "feature": "tvdenoising3d",
            "params": {"regularization_amount": 0.001, "max_iter": 50},
        },
        "laplacian": {
            "plugin": "features",
            "feature": "laplacian",
            "params": {"kernel_size": 3},
        },
        "gradient": {
            "plugin": "features",
            "feature": "gradient",
            "params": {},
        },
        "simple_invert": {
            "plugin": "features",
            "feature": "",
            "params": {},
        },
        "median": {
            "plugin": "features",
            "feature": "",
            "params": {"median_size": 2, "num_iter": 1},
        },
        "ndimage_laplacian": {
            "plugin": "features",
            "feature": "",
            "params": {"kernel_size": 2},
        },
        "hessian_eigvals": {
            "plugin": "features",
            "feature": "",
            "params": {"sigma": 2},
        },
        "gamma_adjust": {
            "plugin": "features",
            "feature": "",
            "params": {"gamma": 2},
        },
        "threshold": {
            "plugin": "features",
            "feature": "",
            "params": {"thresh": 0.7},
        },
        "invert_threshold": {
            "plugin": "features",
            "feature": "",
            "params": {"thresh": 0.7},
        },
    }

    feature_params = [
        [gaussian_blur_kornia, filter_cfg["gaussian1"]["params"]],
        [gaussian_blur_kornia, filter_cfg["gaussian2"]["params"]],
        [laplacian, filter_cfg["laplacian"]["params"]],
        [tvdenoise_kornia, filter_cfg["tvdenoise"]["params"]],
        [spatial_gradient_3d, filter_cfg["gradient"]["params"]],
        [simple_invert, filter_cfg["simple_invert"]["params"]],
        [median, filter_cfg["median"]["params"]],
        [ndimage_laplacian, filter_cfg["ndimage_laplacian"]["params"]],
        [hessian_eigvals_image, filter_cfg["hessian_eigvals"]["params"]],
        [gamma_adjust, filter_cfg["gamma_adjust"]["params"]],
        [threshold, filter_cfg["threshold"]["params"]],
        [invert_threshold, filter_cfg["invert_threshold"]["params"]],
    ]

    num_feature_params = len(feature_params)
    roi_crop = [0, img_vol.shape[0], 0, img_vol.shape[1], 0, img_vol.shape[2]]
    sr_feat = generate_features(img_vol, feature_params, roi_crop, 1.0)
    num_generated_features = len(sr_feat.filtered_layers)
    assert num_feature_params == num_generated_features

    generated_feature_shapes = [layer.shape for layer in sr_feat.filtered_layers]
    assert generated_feature_shapes == [img_vol.shape for i in range(num_generated_features)]


if __name__ == "__main__":
    pytest.main()
