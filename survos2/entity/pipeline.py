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

import yaml
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from survos2.entity.sampler import crop_vol_and_pts_centered
from survos2.frontend.control import Launcher
from survos2.improc.utils import DatasetManager
from survos2.model import DataModel
from survos2.server.model import SRData, SRFeatures, SRPrediction


@dataclass
class Patch:
    image_layers: Dict
    annotation_layers: Dict
    geometry_layers: Dict
    features: SRFeatures


def run_workflow(workflow_file):
    print(workflow_file)
    fullpath = os.path.join(
        os.path.abspath(os.getcwd()), os.path.abspath(workflow_file)
    )
    if not os.path.isabs(workflow_file):
        workflow_file = fullpath

        with open(workflow_file) as f:
            workflows = yaml.safe_load(f.read())

        num_workflow_steps = len(workflows.keys())
        minVal, maxVal = 0, num_workflow_steps

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

    else:
        print("Need input workflow YAML file")

    return all_params, params
