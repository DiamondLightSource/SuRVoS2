import numpy as np
from survos2.server.model import SRFeatures
from loguru import logger
from survos2.io import dataset_from_uri
from survos2.improc.utils import map_blocks
import scipy
from survos2.improc.segmentation.mappings import rmeans, normalize


def prepare_prediction_features(filtered_layers):
    # reshaping for survos

    logger.debug(f"Preparing {len(filtered_layers)} features of shape {filtered_layers[0].shape}")

    dataset_feats_reshaped = [
        f.reshape(
            1,
            filtered_layers[0].shape[0],
            filtered_layers[0].shape[1],
            filtered_layers[0].shape[2],
        )
        for f in filtered_layers
    ]

    dataset_feats = np.vstack(dataset_feats_reshaped).astype(np.float32)

    features_stack = []

    for i, feature in enumerate(dataset_feats):
        features_stack.append(feature[...].ravel())

    features_stack = np.stack(features_stack, axis=1).astype(np.float32)

    return dataset_feats, features_stack


def features_factory(filtered_layers):
    logger.debug(f"Preparing SRFeatures with number of images {len(filtered_layers)}")
    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    features = SRFeatures(filtered_layers, dataset_feats, features_stack)

    return features


def prepare_features(features, roi_crop, resample_amt):
    """Calculate filters on image volume to generate features for survos segmentation

    Arguments:
        features {list of string} -- list of feature uri
        roi_crop {tuple of int} -- tuple defining a bounding box for cropping the image volume
        resample_amt {float} -- amount to scale the input volume

    Returns:
        features -- dataclass containing the processed image layers, and a stack made from them
    """
    # features_stack = []
    filtered_layers = []

    for i, feature in enumerate(features):

        logger.info(f"Loading feature number {i}: {os.path.basename(feature)}")

        data = dataset_from_uri(feature, mode="r")
        data = data[
            roi_crop[0] : roi_crop[1],
            roi_crop[2] : roi_crop[3],
            roi_crop[4] : roi_crop[5],
        ]
        data = scipy.ndimage.zoom(data, resample_amt, order=1)

        logger.info(f"Cropped and resampled feature shape: {data.shape}")
        filtered_layers.append(data)

        # features_stack.append(data[...].ravel())

    # features_stack = np.stack(features_stack, axis=1)

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    features = SRFeatures(filtered_layers, dataset_feats, features_stack)

    return features


def generate_features(img_vol, feature_params, roi_crop, resample_amt):
    def proc_layer(layer):
        layer_proc = layer[
            roi_crop[0] : roi_crop[1],
            roi_crop[2] : roi_crop[3],
            roi_crop[4] : roi_crop[5],
        ].astype(np.float32, copy=False)
        return layer_proc

    logger.info(f"From img vol of shape: {img_vol.shape}")
    logger.info(f"Generating features with params: {feature_params}")

    # map_blocks through Dask
    filtered_layers = [
        proc_layer(map_blocks(filter_fn, img_vol, **params_dict)) for filter_fn, params_dict in feature_params
    ]

    filtered_layers = np.array(filtered_layers).astype(np.float32)

    dataset_feats, features_stack = prepare_prediction_features(filtered_layers)
    logger.info(f"Shape of feature data: {dataset_feats.shape}")

    features = SRFeatures(filtered_layers, dataset_feats, features_stack)

    return features
