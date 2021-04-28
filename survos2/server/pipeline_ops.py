import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loguru import logger
from skimage import img_as_ubyte
from survos2.entity.anno import geom
from survos2.entity.anno.masks import generate_sphere_masks_fast
from survos2.entity.entities import make_entity_df
from survos2.entity.saliency import filter_proposal_mask, measure_big_blobs
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts_centered,
    viz_bvols,
)
from survos2.frontend.nb_utils import show_images
from survos2.helpers import AttrDict
from survos2.improc.features import gaussian, gaussian_norm, tvdenoising3d
from survos2.server.features import features_factory, generate_features
from survos2.server.model import SRData, SRFeatures, SRPrediction
from survos2.server.pipeline import Patch
from survos2.server.superseg import mrf_refinement, sr_predict
from survos2.server.supervoxels import generate_supervoxels, superregion_factory
from torch.utils.data import DataLoader

from tqdm import tqdm


def save_model(filename, model, optimizer, torch_models_fullpath):

    if not os.path.exists(torch_models_fullpath):
        os.makedirs(torch_models_fullpath)

    model_dictionary = {
        "model_state": model.state_dict(),
        "model_optimizer": optimizer.state_dict(),
    }

    checkpoint_directory = torch_models_fullpath
    file_path = os.path.join(checkpoint_directory, filename)

    torch.save(model_dictionary, file_path)


def make_noop(patch: Patch, params: dict):
    return patch


def sphere_masks(patch: Patch, params: dict):
    total_mask = generate_sphere_masks_fast(
        patch.image_layers["Main"],
        patch.geometry_layers["Entities"],
        radius=params.pipeline.mask_radius,
    )
    show_images(
        [total_mask[total_mask.shape[0] // 2, :]],
        ["Sphere mask, radius: " + str(mask_radius),],
    )

    patch.image_layers["generated"] = total_mask

    return patch


def make_masks(patch: Patch, params: dict):
    """
    Rasterize point geometry to a mask (V->R)

    ((Array of 3d points) -> Float layer)

    """
    padding = params["mask_params"]["padding"]
    geom = patch.geometry_layers["Points"].copy()

    mask_radius = params["mask_params"]["mask_radius"]

    geom[:, 0] = geom[:, 0] + padding[0]
    geom[:, 1] = geom[:, 1] + padding[1]
    geom[:, 2] = geom[:, 2] + padding[2]

    total_mask = generate_sphere_masks_fast(
        patch.image_layers["Main"], geom, radius=mask_radius[0],
    )

    core_mask = generate_sphere_masks_fast(
        patch.image_layers["Main"],
        geom,
        radius=params["mask_params"]["core_mask_radius"][0],
    )

    show_images(
        [total_mask[total_mask.shape[0] // 2, :], core_mask[core_mask.shape[0] // 2, :]],
        figsize=(4,4)
    )

    patch.image_layers["total_mask"] = total_mask
    patch.image_layers["core_mask"] = core_mask

    return patch


def make_features(patch: Patch, params: dict):
    """
    Features (Float layer -> List[Float layer])

    """

    logger.debug("Calculating features")

    cropped_vol = patch.image_layers["Main"]

    roi_crop = [
        0,
        cropped_vol.shape[0],
        0,
        cropped_vol.shape[1],
        0,
        cropped_vol.shape[2],
    ]
    logger.debug(f"Roi crop: {roi_crop}")

    for feature_name, v in params["feature_params"].items():
        logger.debug(feature_name, v)
        features = generate_features(cropped_vol, v, roi_crop, 1.0)

        for i, layer in enumerate(features.filtered_layers):
            patch.image_layers[feature_name] = layer

    return patch


def make_sr(patch: Patch, params: dict):
    """
    Takes the first feature defined in feature_params
    """
    # logger.debug("Making superregions for sample")

    filtered_layers = [patch.image_layers[k] for k in params["feature_params"].keys()]
    features = features_factory(filtered_layers)
    superregions = generate_supervoxels(
        np.array(features.dataset_feats),
        features.features_stack,
        0,
        params["filter_cfg"]["superregions1"]["params"],
    )

    patch.features = features
    patch.image_layers["SR"] = superregions.supervoxel_vol

    return patch


def predict_sr(patch: Patch, params: dict):
    """"""
    # logger.debug("Predicting superregions")

    sr = superregion_factory(
        patch.image_layers["SR"].astype(np.uint16), patch.features.features_stack
    )
    srprediction = _sr_prediction(
        patch.features.features_stack,
        patch.image_layers["total_mask"].astype(np.uint),
        sr,
        params["pipeline"]["predict_params"],
    )

    Rp_ref = mrf_refinement(
        srprediction.P,
        sr.supervoxel_vol,
        patch.features.features_stack,
        lam=10,
        gamma=False,
    )

    predicted = srprediction.prob_map.copy()
    predicted -= np.min(srprediction.prob_map)
    predicted += 1
    predicted = img_as_ubyte(predicted / np.max(predicted))
    predicted = predicted - np.min(predicted)
    predicted = predicted / np.max(predicted)

    patch.image_layers["segmentation"] = predicted
    patch.image_layers["refinement"] = Rp_ref

    return patch


def make_features2(patch: Patch, params: dict):
    """
    Features (Float layer -> List[Float layer])

    """

    logger.info("Calculating features")

    cropped_vol = patch.image_layers["Main"]
    print(cropped_vol.shape)

    roi_crop = [
        0,
        cropped_vol.shape[0],
        0,
        cropped_vol.shape[1],
        0,
        cropped_vol.shape[2],
    ]
    print(f"Roi crop: {roi_crop}")

    for feature_name, v in params["feature_params"].items():
        print(feature_name, v)
        features = generate_features(cropped_vol, v, roi_crop, 1.0)

        for i, layer in enumerate(features.filtered_layers):
            patch.image_layers[feature_name] = layer

    return patch


def make_acwe(patch: Patch, params: dict):
    """
    Active Contour

    (Float layer -> Float layer)

    """
    from skimage import exposure

    edge_map = 1.0 - patch.image_layers["Main"]
    edge_map = exposure.adjust_sigmoid(edge_map, cutoff=1.0)
    logger.debug("Calculating ACWE")
    import morphsnakes as ms

    seg1 = ms.morphological_geodesic_active_contour(
        edge_map,
        iterations=3,
        init_level_set=patch.image_layers["total_mask"],
        smoothing=1,
        threshold=0.1,
        balloon=1.1,
    )

    outer_mask = ((seg1 * 1.0) > 0) * 2.0

    # inner_mask = ((seg2 * 1.0) > 0) * 1.0
    # outer_mask = outer_mask * (1.0 - inner_mask)
    # anno = outer_mask + inner_mask

    patch.image_layers["acwe"] = outer_mask
    #show_images([outer_mask[outer_mask.shape[0] // 2, :]], figsize=(12, 12))

    return patch


def make_sr2(patch: Patch, params: dict):
    """
    # SRData
    #    Supervoxels (Float layer->Float layer)
    #
    #    Also generates feature vectors

    SRData contains the supervoxel image as well as the feature vectors made
    from features and the supervoxels

    """
    logger.debug("Making superregions for sample")

    filtered_layers = [patch.image_layers[k] for k in params["feature_params"].keys()]

    features = features_factory(filtered_layers)
    superregions = generate_supervoxels(
        np.array(features.dataset_feats),
        features.features_stack,
        params["pipeline"]["slic_feat_idx"],
        params["slic_params"],
    )

    patch.annotation_layers["SR"] = superregions.supervoxel_vol

    return patch


def predict_sr2(patch: Patch, params: dict):
    """
    (SRFeatures, Annotation, SRData -> Float layer prediction)

    """
    logger.debug("Predicting superregions")

    srprediction = _sr_prediction(
        features.features_stack,
        patch.image_layers["Annotation"],
        superregions,
        params.predict_params,
    )

    predicted = srprediction.prob_map.copy()
    predicted -= np.min(srprediction.prob_map)
    predicted += 1
    predicted = img_as_ubyte(predicted / np.max(predicted))
    predicted = predicted - np.min(predicted)
    predicted = predicted / np.max(predicted)
    patch.image_layers["segmentation"] = predicted

    return patch


def clean_segmentation(patch: Patch, params: dict):

    predicted = patch.image_layers["segmentation"]

    # clean prediction
    struct2 = ndimage.generate_binary_structure(3, 2)
    predicted_cl = (predicted > 0.0) * 1.0
    predicted_cl = ndimage.binary_closing(predicted_cl, structure=struct2)
    predicted_cl = ndimage.binary_opening(predicted_cl, structure=struct2)
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    predicted_cl = img_as_ubyte(ndimage.median_filter(predicted_cl, size=4))
    patch.image_layers["prediction_cleaned"] = predicted_cl

    return patch


def saliency_pipeline(patch: Patch, params: dict):
    """
    Use CNN to predict a saliency map from an image volume

    Post process the map into bounding boxes.
    """
    output_tensor = predict_and_agg(
        params["saliency_model"],
        patch.image_layers["Main"],
        patch_size=(1, 224, 224),
        patch_overlap=(0, 0, 0),
        batch_size=1,
        stacked_3chan=params["pipeline"]["stacked_3chan"],
    )

    patch.image_layers["saliency_map"] = output_tensor.detach().squeeze(0).numpy()
    return patch


def make_bb(patch: Patch, params: dict):
    holdout = filter_proposal_mask(
        patch.image_layers["saliency_map"], thresh=0.5, num_erosions=3, num_dilations=3
    )
    images = [
        holdout,
    ]
    filtered_tables = measure_big_blobs(images)
    logger.debug(filtered_tables)
    cols = ["z", "x", "y", "class_code"]
    pred_cents = np.array(filtered_tables[0][cols])
    preds = centroid_to_bvol(pred_cents)
    saliency_bb = viz_bvols(patch.image_layers["Main Volume"], preds)
    logger.info(f"Produced bb mask of shape {saliency_bb.shape}")
    patch.image_layers["saliency_bb"] = saliency_bb

    return patch


def rasterize_bb(patch: Patch, params: dict):
    holdout = filter_proposal_mask(
        patch.image_layers["proposal_mask"], thresh=0.5, num_erosions=3, num_dilations=3
    )
    filtered_tables = measure_big_blobs(images)

    cols = ["z", "x", "y", "class_code"]
    pred_cents = np.array(filtered_tables[0][cols])
    # cents = cents[:,[0,2,1,3]]
    target_cents = np.array(cropped_pts_df)[:, 0:4]

    preds = centroid_to_bvol(pred_cents)
    targs = centroid_to_bvol(target_cents)

    patch.image_layers["preds"] = preds
    patch.image_layers["targs"] = targs

    return patch


def cnn_predict_2d_3chan(patch: Patch, params: dict):
    model = seg_models["cnn"]
    inputs_t = torch.FloatTensor(patch.image_layers["Main"])
    # inputs_t = input_tensor.squeeze(1)
    inputs_t = inputs_t.to(device)
    stacked_t = torch.stack(
        [inputs_t[:, 0, :, :], inputs_t[:, 0, :, :], inputs_t[:, 0, :, :]], axis=1
    )
    logger.info(f"inputs_t: {inputs_t.shape}")
    pred = model(stacked_t)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu()
    output = pred.unsqueeze(1)
    logger.debug(f"pred: {pred.shape}")

    patch.image_layers["cnn_prediction"] = predicted_cl

    return patch


def predict_2d_3chan(model, input_tensor, device):
    device = torch.device(device)
    inputs_t = input_tensor.squeeze(1)
    inputs_t = inputs_t.to(device)
    stacked_t = torch.stack(
        [inputs_t[:, 0, :, :], inputs_t[:, 0, :, :], inputs_t[:, 0, :, :]], axis=1
    )
    print(f"inputs_t: {inputs_t.shape}")
    pred = model(stacked_t)
    # pred, feats = model.forward_(stacked_t)
    pred = F.sigmoid(pred)
    pred = pred.data.cpu()

    # output = pred.unsqueeze(1)
    print(f"pred: {pred.shape}")

    return pred


def predict_and_agg(
    model,
    input_array,
    patch_size=(1, 224, 224),
    patch_overlap=(0, 16, 16),
    batch_size=1,
    stacked_3chan=False,
    extra_unsqueeze=True,
    device=0,
):

    import torchio as tio
    from torchio import IMAGE, LOCATION
    from torchio.data.inference import GridAggregator, GridSampler
    from torchvision import datasets, models, transforms

    device = torch.device(device)
    img_tens = torch.FloatTensor(input_array)

    one_subject = tio.Subject(
        img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
        label=tio.Image(tensor=img_tens, label=tio.LABEL),
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(grid_sampler)

    input_tensors = []
    output_tensors = []

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):

            input_tensor = patches_batch["img"]["data"]
            location = patches_batch[LOCATION]

            # print(f"Input tensor {input_tensor.shape}")

            inputs_t = input_tensor.squeeze(1)
            inputs_t = inputs_t.to(device)

            if stacked_3chan:
                inputs_t = torch.stack(
                    [inputs_t[:, 0, :, :], inputs_t[:, 0, :, :], inputs_t[:, 0, :, :]],
                    axis=1,
                )
            else:
                inputs_t = inputs_t[:, 0:1, :, :]

            print(f"inputs_t: {inputs_t.shape}")

            output = predict_2d_3chan(model, input_tensor, device)

            # pred = model(inputs_t)
            # pred = torch.sigmoid(pred[0])
            # pred = torch.sigmoid(pred[0])
            # pred = pred.data.cpu()
            # print(f"pred: {pred.shape}")

            if extra_unsqueeze:
                output = output.unsqueeze(1)
            # output = pred.squeeze(0)

            input_tensors.append(input_tensor)
            output_tensors.append(output)

            # print(output.shape, location)
            aggregator.add_batch(output, location)

    output_tensor = aggregator.get_output_tensor()
    logger.debug(f"Predicted volume: {output_tensor.shape}")

    return output_tensor


def predict_and_agg2(
    model,
    input_array,
    patch_size=(1, 224, 224),
    patch_overlap=(0, 16, 16),
    batch_size=1,
    stacked_3chan=False,
    extra_unsqueeze=True,
):
    """
    Stacked CNN Predict and Aggregate
    Uses torchio
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tens = torch.FloatTensor(input_array)
    print(img_tens.shape)

    one_subject = tio.Subject(
        img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
        label=tio.Image(tensor=img_tens, label=tio.LABEL),
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(grid_sampler)

    input_tensors = []
    output_tensors = []

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):

            input_tensor = patches_batch["img"]["data"]
            location = patches_batch[LOCATION]

            print(f"Input tensor {input_tensor.shape}")

            inputs_t = input_tensor.squeeze(1)
            inputs_t = inputs_t.to(device)

            if stacked_3chan:
                inputs_t = torch.stack(
                    [inputs_t[:, 0, :, :], inputs_t[:, 0, :, :], inputs_t[:, 0, :, :]],
                    axis=1,
                )
            else:
                inputs_t = inputs_t[:, 0:1, :, :]

            print(f"inputs_t: {inputs_t.shape}")

            pred = model(inputs_t)
            # pred = torch.sigmoid(pred[0])
            pred = 1.0 - pred[0].data.cpu()
            print(f"pred: {pred.shape}")

            if extra_unsqueeze:
                output = pred.unsqueeze(1)
            # output = pred.squeeze(0)

            input_tensors.append(input_tensor)
            output_tensors.append(output)

            print(output.shape, location)
            aggregator.add_batch(output, location)

    output_tensor = aggregator.get_output_tensor()
    print(input_tensor.shape, output_tensor.shape)

    return output_tensor


def test_make_masks(p):
    print(f"Making masks ")
    p.image_layers["mask"] = np.array([0, 1, 2, 3])
    return p


def test_make_features(p):
    p.image_layers["features"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making features")
    return p


def test_make_sr(p):
    p.image_layers["sr"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making superregions")
    return p


def test_make_seg_sr(p):
    p.image_layers["seg_sr"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making segmentation with superregions")
    return p


def test_make_seg_cnn(p):
    p.image_layers["seg_cnn"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making segmentation with cnn")
    return p
