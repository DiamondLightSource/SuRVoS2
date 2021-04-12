from loguru import logger
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, models, transforms

import torchio as tio
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator

from survos2.entity.sampler import crop_vol_and_pts_centered
from survos2.server.pipeline import Pipeline, Patch


def make_masks(p):
    print(f"Making masks ")
    p.image_layers["mask"] = np.array([0, 1, 2, 3])
    return p


def make_features(p):
    p.image_layers["features"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making features")
    return p


def make_sr(p):
    p.image_layers["sr"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making superregions")
    return p


def make_seg_sr(p):
    p.image_layers["seg_sr"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making segmentation with superregions")
    return p


def make_seg_cnn(p):
    p.image_layers["seg_cnn"] = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"Making segmentation with cnn")
    return p


# todo: better way to add multiple aggregators and bundle results


def gridsampler_pipeline(
    input_array,
    entity_pts,
    patch_size=(64, 64, 64),
    patch_overlap=(0, 0, 0),
    batch_size=1,
):
    logger.debug("Starting up gridsampler pipeline...")
    input_tensors = []
    output_tensors = []

    entity_pts = entity_pts.astype(np.int32)
    img_tens = torch.FloatTensor(input_array)

    one_subject = tio.Subject(
        img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
        label=tio.Image(tensor=img_tens, label=tio.LABEL),
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator1 = GridAggregator(grid_sampler)
    aggregator2 = GridAggregator(grid_sampler)

    pipeline = Pipeline(
        {
            "p": 1,
            "ordered_ops": [
                make_masks,
                make_features,
                make_sr,
                make_seg_sr,
                make_seg_cnn,
            ],
        }
    )

    payloads = []

    with torch.no_grad():
        for patches_batch in patch_loader:
            locations = patches_batch[LOCATION]
            loc_arr = np.array(locations[0])
            loc = (loc_arr[0], loc_arr[1], loc_arr[2])
            logger.debug(f"Location: {loc}")

            # Prepare region data (IMG (Float Volume) AND GEOMETRY (3d Point))
            cropped_vol, offset_pts = crop_vol_and_pts_centered(
                input_array,
                entity_pts,
                location=loc,
                patch_size=patch_size,
                offset=True,
                debug_verbose=True,
            )

            plt.figure(figsize=(12, 12))
            plt.imshow(cropped_vol[cropped_vol.shape[0] // 2, :], cmap="gray")
            plt.scatter(offset_pts[:, 1], offset_pts[:, 2])

            logger.debug(f"Number of offset_pts: {offset_pts.shape}")
            logger.debug(
                f"Allocating memory for no. voxels: {cropped_vol.shape[0] * cropped_vol.shape[1] * cropped_vol.shape[2]}"
            )

            # payload = Patch(
            #    {"in_array": cropped_vol},
            #    offset_pts,
            #    None,
            # )

            payload = Patch(
                {"total_mask": np.random.random((4, 4),)},
                {"total_anno": np.random.random((4, 4),)},
                {"points": np.random.random((4, 3),)},
            )
            pipeline.init_payload(payload)

            for step in pipeline:
                print(step)

            # Aggregation (Output: large volume aggregated from many smaller volumes)
            output_tensor = (
                torch.FloatTensor(payload.annotation_layers["total_mask"])
                .unsqueeze(0)
                .unsqueeze(1)
            )
            logger.debug(f"Aggregating output tensor of shape: {output_tensor.shape}")
            aggregator1.add_batch(output_tensor, locations)

            output_tensor = (
                torch.FloatTensor(payload.annotation_layers["prediction"])
                .unsqueeze(0)
                .unsqueeze(1)
            )
            logger.debug(f"Aggregating output tensor of shape: {output_tensor.shape}")
            aggregator2.add_batch(output_tensor, locations)
            payloads.append(payload)

    output_tensor1 = aggregator1.get_output_tensor()
    logger.debug(output_tensor1.shape)
    output_arr1 = np.array(output_tensor1.squeeze(0))

    output_tensor2 = aggregator2.get_output_tensor()
    logger.debug(output_tensor2.shape)
    output_arr2 = np.array(output_tensor2.squeeze(0))

    return [output_tensor1, output_tensor2], payloads
