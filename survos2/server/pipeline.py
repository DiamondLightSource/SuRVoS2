"""
Pipeline ops

"""
import itertools
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
import yaml
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from survos2.entity.sampler import crop_vol_and_pts_centered
from survos2.frontend.control import Launcher
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.model import SRFeatures


@dataclass
class Patch:
    """A Patch is processed by a Pipeline

    3 layer dictionaries for the different types (float image, integer image, geometry)

    Pipeline functions need to agree on the names they
    use for layers.

    TODO Adapter
    """

    image_layers: Dict
    annotation_layers: Dict
    geometry_layers: Dict
    features: SRFeatures


class Pipeline:
    """
    A pipeline produces output such as a segmentation, and often has
    several different inputs of several different types as well as a
    dictionary of parameters (e.g. a superregion segmentation takes
    an annotation uint16, a supervoxel uint32 and multiple float32 images)

    A pipeline follows the iterator protocol. The caller creates an instance,
    providing the list of operations and a payload and then iterates through
    the pipeline. The payload may be changed with init_payload. The result Patch
    is obtained by calling output_result


    """

    def __init__(self, params, models=None):
        self.params = params
        self.ordered_ops = iter(params["ordered_ops"])
        self.payload = None

    def init_payload(self, patch):
        self.payload = patch

    def output_result(self):
        return self.payload

    def __iter__(self):
        return self

    def __next__(self):
        self.payload = next(self.ordered_ops)(self.payload)
        return self.payload


# todo: better way to add multiple aggregators and bundle results


def run_workflow(workflow_file):
    if not os.path.isabs(workflow_file):
        workflow_file = os.path.join(os.getcwd(), workflow_file)

        with open(workflow_file) as f:
            workflows = yaml.safe_load(f.read())

        num_workflow_steps = len(workflows.keys())
        minVal, maxVal = 0, num_workflow_steps

        print(workflows)

        for step_idx, k in enumerate(workflows):
            workflow = workflows[k]
            action = workflow.pop("action")
            plugin, command = action.split(".")
            params = workflow.pop("params")

            src_name = workflow.pop("src")
            dst_name = workflow.pop("dst")

            if "src_group" in workflow:
                plugin = workflow.pop("src_group")

            src = DataModel.g.dataset_uri(src_name, group=plugin)
            dst = DataModel.g.dataset_uri(dst_name, group=plugin)

            all_params = dict(src=src, dst=dst, modal=True)
            all_params.update(params)
            logger.info(f"Executing workflow {all_params}")

            print(
                f"+ Running {k}, with {plugin}, {command} on {src}\n to dst {dst} {all_params}\n"
            )

            import survos

            # Launcher.g.run(plugin, command, **all_params)
            survos.run_command(plugin, command, uri=None, src=src, dst=dst)

            # src_arr = view_dataset(dst_name, plugin, 10)

    else:
        print("Need input workflow YAML file")

    return all_params, params


def view_dataset(dataset_name, group, z):
    src = DataModel.g.dataset_uri(dataset_name, group=group)

    with DatasetManager(src, out=None, dtype="float32") as DM:
        src_dataset = DM.sources[0]
        src_arr = src_dataset[:]
    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(src_arr[z, :])

    return src_arr


def gridsampler_pipeline(
    input_array,
    entity_pts,
    patch_size=(64, 64, 64),
    patch_overlap=(0, 0, 0),
    batch_size=1,
):
    from torchio import IMAGE, LOCATION
    from torchio.data.inference import GridAggregator, GridSampler

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
                logger.debug(step)

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
