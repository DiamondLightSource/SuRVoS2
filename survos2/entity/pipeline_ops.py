import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from loguru import logger
from skimage import img_as_ubyte
from survos2.entity.anno import geom
from survos2.entity.anno.masks import generate_sphere_masks_fast
from survos2.entity.components import filter_proposal_mask, measure_big_blobs
from survos2.entity.sampler import (
    centroid_to_bvol,
    crop_vol_and_pts_centered,
    viz_bvols,
)
from survos2.frontend.nb_utils import show_images
from survos2.helpers import AttrDict
from survos2.server.features import features_factory, generate_features
from survos2.server.model import SRData, SRFeatures, SRPrediction
from survos2.entity.pipeline import Patch
from survos2.server.superseg import mrf_refinement, sr_predict
from survos2.server.supervoxels import generate_supervoxels, superregion_factory
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm


def load_model(detmod, file_path):
    def load_model_parameters(full_path):
        checkpoint = torch.load(full_path)
        return checkpoint

    checkpoint = load_model_parameters(file_path)
    detmod.load_state_dict(checkpoint["model_state"])
    detmod.eval()

    print(f"Loaded model from {file_path}")
    return detmod


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
        [
            "Sphere mask, radius: " + str(mask_radius),
        ],
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
        patch.image_layers["Main"],
        geom,
        radius=mask_radius[0],
    )

    core_mask = generate_sphere_masks_fast(
        patch.image_layers["Main"],
        geom,
        radius=params["mask_params"]["core_mask_radius"][0],
    )

    show_images(
        [
            total_mask[total_mask.shape[0] // 2, :],
            core_mask[core_mask.shape[0] // 2, :],
        ],
        figsize=(4, 4),
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


def predict_agg_3d(
    input_array,
    model3d,
    patch_size=(128, 224, 224),
    patch_overlap=(12, 12, 12),
    nb=True,
    device=0,
    debug_verbose=False,
    fpn=False,
    overlap_mode="crop",
):
    import torchio as tio
    from torchio import IMAGE, LOCATION
    from torchio.data.inference import GridAggregator, GridSampler

    img_tens = torch.FloatTensor(input_array).unsqueeze(0)
    print(f"Predict and aggregate on volume of {img_tens.shape}")

    one_subject = tio.Subject(
        img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
        label=tio.Image(tensor=img_tens, label=tio.LABEL),
    )

    img_dataset = tio.SubjectsDataset(
        [
            one_subject,
        ]
    )
    img_sample = img_dataset[-1]

    batch_size = 1

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator1 = GridAggregator(grid_sampler, overlap_mode=overlap_mode)

    input_tensors = []
    output_tensors = []

    if nb:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    with torch.no_grad():

        for patches_batch in tqdm(patch_loader):
            input_tensor = patches_batch["img"]["data"]
            locations = patches_batch[LOCATION]
            inputs_t = input_tensor
            inputs_t = inputs_t.to(device)

            if fpn:
                outputs = model3d(inputs_t)[0]
            else:
                outputs = model3d(inputs_t)
            if debug_verbose:
                print(f"inputs_t: {inputs_t.shape}")
                print(f"outputs: {outputs.shape}")

            output = outputs[:, 0:1, :]
            # output = torch.sigmoid(output)

            aggregator1.add_batch(output, locations)

    return aggregator1


def prepare_unet3d(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from unet import UNet2D, UNet3D

    device = torch.device(device)
    model3d = UNet3D(
        normalization="batch",
        preactivation=True,
        residual=True,
        num_encoding_blocks=3,
        upsampling_type="trilinear",
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} gpus.")
        model3d = nn.DataParallel(model3d).to(device).eval()
    else:
        model3d = model3d.to(device).eval()
    # optimizer = optim.Adam(model3d.parameters(), lr=initial_lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, num_epochs)

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler


def make_proposal(
    vol,
    model_fullname,
    model_type,
    nb=True,
    patch_size=(64, 64, 64),
    patch_overlap=(0, 0, 0),
    overlap_mode="crop",
    gpu_id=0,
):

    if model_type == "unet3d":
        model3d, optimizer, scheduler = prepare_unet3d(device=gpu_id)
    elif model_type == "fpn3d":
        model3d, optimizer, scheduler = prepare_fpn3d(gpu_id=gpu_id)
    print(f"Predicting segmentation on volume of shape {vol.shape}")

    if model_type == "unet3d":
        model3d = load_model(model3d, model_fullname)
        aggregator = predict_agg_3d(
            vol,
            model3d,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            device=gpu_id,
            fpn=False,
            overlap_mode=overlap_mode,
        )
        output_tensor1 = aggregator.get_output_tensor()
        print(f"Aggregated volume of {output_tensor1.shape}")
        seg_out = np.nan_to_num(output_tensor1.squeeze(0).numpy())

    elif model_type == "fpn3d":
        model3d = load_model(model3d, model_fullname)
        aggregator = predict_agg_3d(
            vol,
            model3d,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            device=gpu_id,
            fpn=False,
        )
        output_tensor1 = aggregator.get_output_tensor()
        print(f"Aggregated volume of {output_tensor1.shape}")
        seg_out = np.nan_to_num(output_tensor1.squeeze(0).numpy())

    return seg_out
