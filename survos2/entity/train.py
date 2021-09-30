import json
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import torch

from survos2.entity.pipeline_ops import save_model
from survos2.entity.pipeline_ops import save_model
from entityseg.training.patches import load_patch_vols, prepare_dataloaders
from entityseg.models.unet3d import prepare_unet3d,display_unet_pred # , train_unet3d
from survos2.entity.models.head_cnn import (
    display_fpn3d_pred,
    prepare_fpn3d,
    
)
from entityseg.training.trainer import (
    train_fpn3d,
)


def train_seg(
    train_v_class1,
    wf_params,
    model_type="fpn3d",
    gpu_id=0,
    load_saved_model=False,
    save_current_model=True,
    test_on_volume=False,
    model=None,
    num_epochs=1,
):

    train_params = {
        "train_vols": (train_v_class1[0], train_v_class1[1]),
        "model_type": model_type,
        "num_epochs": num_epochs,
        "gpu_id": gpu_id,
        "load_saved_model": load_saved_model,
        "save_current_model": save_current_model,
        "test_on_volume": test_on_volume,
        "display_plots": False,
        "torch_models_fullpath": wf_params["torch_models_fullpath"],
    }
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    
    return train_all(train_params, model)


def train_all(
    train_params,
    patch_size=(64, 64, 64),
    model=None,
    batch_size=1,
    bce_weight=0.7,
    initial_lr=0.001,
):
    model_type = train_params["model_type"]
    gpu_id = train_params["gpu_id"]
    save_current_model = train_params["save_current_model"]
    num_epochs = train_params["num_epochs"]
    torch_models_fullpath = train_params["torch_models_fullpath"]
    print(train_params["train_vols"])

    # prepare patch dataset
    img_vols, label_vols = load_patch_vols(train_params["train_vols"])
    dataloaders = prepare_dataloaders(
        img_vols, label_vols, train_params["model_type"], display_plots=False
    )

    if model_type == "fpn3d":
        detmod, optimizer, scheduler = prepare_fpn3d(gpu_id=train_params["gpu_id"])
    if model is not None:
        detmod = model

    if model_type == "fpn3d":
        detmod, outputs, losses = train_fpn3d(
            detmod,
            dataloaders,
            optimizer,
            dice_weight=1 - bce_weight,
            num_epochs=num_epochs,
            device=gpu_id,
        )
        # from functools import partial
        # from entityseg.training.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d
        # print(f"Training model with num_seg_classes: {detmod.cf.num_seg_classes}")

        # fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)

        # metricCallback = MetricCallback()
        # trainer = Trainer(
        #     detmod,
        #     optimizer,
        #     fpn3d_criterion,
        #     scheduler,
        #     dataloaders,
        #     metricCallback,
        #     prepare_labels=prepare_labels_fpn3d,
        #     num_epochs=num_epochs,
        #     initial_lr=0.001,
        #     num_out_channels=1,
        #     device=gpu_id,
        # )

        # training_loss, validation_loss, learning_rate = trainer.run()
        # plt.figure()
        # plt.plot(training_loss)
        # plt.figure()
        # plt.plot(validation_loss)

        # detmod = trainer.model
        # # #print(f"FPN with outputs of shape {outputs.shape}")
        # display_preds_fpn3d(outputs)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            save_model(
                "detmod_gtacwe_quarter" + dt_string + ".pt",
                detmod,
                optimizer,
                torch_models_fullpath,
            )

        if train_params["display_plots"]:
            display_fpn3d_pred(detmod, dataloaders, device=gpu_id)

    if model_type == "fpn3d":
        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "fpn3d_fullblob" + dt_string + ".pt"
            save_model(model_file, detmod, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")

    if model_type == "unet3d":
        model3d, optimizer, scheduler = prepare_unet3d(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )

        from functools import partial
        from entityseg.training.trainer import (
            Trainer,
            MetricCallback,
            unet3d_loss,
            prepare_labels_unet3d,
        )

        unet_criterion = partial(unet3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            unet_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_unet3d,
            num_out_channels=2,
            num_epochs=num_epochs,
            initial_lr=0.01,
            device=gpu_id,
        )
        training_loss, validation_loss, learning_rate = trainer.run()
        plt.plot(training_loss)
        plt.plot(validation_loss)

        model3d = trainer.model
        if train_params["display_plots"]:
            display_unet_pred(model3d, dataloaders, device=gpu_id)

    if model_type == "unet3d":
        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "unet3d_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")

    return model_file
