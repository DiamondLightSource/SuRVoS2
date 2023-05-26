#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import numpy as np
import pandas as pd
import json
import os
import yaml
from loguru import logger
from matplotlib import pyplot as plt
from survos2.frontend.nb_utils import start_server
import papermill as pm


os.chdir("../tests")


# # Start Server

port = 8843
loss_criterion = "BCEDiceLoss"
encoder_type = "resnet50"
workspace_name = "vf_main_feb2023"


for method in [
    "U_NET",
]:  # ['U_NET','DEEPLABV3', 'U_NET_PLUS_PLUS', 'FPN', 'MA_NET']:
    logger.info("Running notebook for: {}".format(method))
    pm.execute_notebook(
        input_path="Seg_Basic_Multiaxis.ipynb",
        output_path="artifact_dir/notebooks/basic_{}.ipynb".format(method),
        parameters={
            "workspace_name": workspace_name,
            "port": port,
            "method": method,
            "loss_criterion": loss_criterion,
            "encoder_type": encoder_type,
        },
    )
