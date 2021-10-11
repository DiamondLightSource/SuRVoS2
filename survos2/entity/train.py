import json
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import torch

# from survos2.entity.models.fpn import Configs

from survos2.entity.pipeline_ops import save_model
from survos2.entity.patches import load_patch_vols, prepare_dataloaders
from entityseg.training.patches import load_patch_vols, prepare_dataloaders
from survos2.entity.models.unet3d import prepare_unet3d,display_unet_pred 
from survos2.entity.models.head_cnn import (
    display_fpn3d_pred,
    prepare_fpn3d,
    
)

from survos2.entity.trainer import (
    train_fpn3d,
)


def train_oneclass_detseg(
    train_v_class1,
    project_file,
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
        "load_saved_model": False,
        "save_current_model": True,
        "test_on_volume": True,
        "project_file": project_file,
        "display_plots": False,
        "torch_models_fullpath": wf_params["torch_models_fullpath"],
    }
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    train_params_file = "vf_train_params" + dt_string + ".json"
    with open(train_params_file, "w") as outfile:
        json.dump(train_params, outfile, indent=4, sort_keys=True)
        print(f"Wrote {train_params_file} with training settings")
    return train_all(train_params_file, model)


def train_twoclass_detseg(
    wf,
    training_vols,
    project_file,
    model_type="fpn3d",
    gpu_id=0,
    load_saved_model=False,
    save_current_model=True,
    test_on_volume=False,
    num_epochs=1,
):

    train_v_class1, train_v_class2 = training_vols
    print(f"Using training volumes: {training_vols}")

    train_params = {
        "train_vols": (train_v_class1[0], train_v_class1[1]),
        "model_type": model_type,
        "num_epochs": num_epochs,
        "gpu_id": gpu_id,
        "load_saved_model": False,
        "save_current_model": True,
        "test_on_volume": True,
        "project_file": project_file,
        "display_plots": False,
        "torch_models_fullpath": wf.params["torch_models_fullpath"],
    }

    train_params_file = "vf_train_params_twoclass.json"
    with open(train_params_file, "w") as outfile:
        json.dump(train_params, outfile, indent=4, sort_keys=True)
        print(f"Wrote {train_params_file} with training settings")

    class1_model_file = train_all(train_params_file)

    print(f"Using training volumes: {train_v_class2[0]}")
    model_type = "fpn3d"
    gpu_id = 0
    load_saved_model = False
    save_current_model = True
    test_on_volume = False

    train_params = {
        "train_vols": (train_v_class2[0], train_v_class2[1]),
        "model_type": model_type,
        "num_epochs": num_epochs,
        "gpu_id": gpu_id,
        "load_saved_model": False,
        "save_current_model": True,
        "test_on_volume": True,
        "project_file": project_file,
        "display_plots": False,
        "torch_models_fullpath": wf.params["torch_models_fullpath"],
    }

    train_params_file = "vf_train_params_twoclass.json"

    with open(train_params_file, "w") as outfile:
        json.dump(train_params, outfile, indent=4, sort_keys=True)
        print(f"Wrote {train_params_file} with training settings")

    class2_model_file = train_all(train_params_file)
    return [class1_model_file, class2_model_file]


def load_model(file_path):
    def load_model_parameters(full_path):
        return torch.load(full_path)


def plot_losses(training_loss, validation_loss):
    plt.figure()
    plt.plot(training_loss)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.figure()
    plt.plot(validation_loss)
    plt.title('Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

def train_all(
    train_param_file="vf_train_params_180221.json",
    patch_size=(64, 64, 64),
    model=None,
    batch_size=1,
    bce_weight=0.7,
    initial_lr=0.001,
):
    with open(train_param_file) as project_file:
        train_params = json.load(project_file)
    pprint(train_params)

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
        detmod, optimizer, scheduler = prepare_fpn3d(gpu_id=train_params["gpu_id"], lr=initial_lr)
    if model is not None:
        detmod = model

    if model_type == "fpn3d":
        # detmod, outputs, losses = train_fpn3d(
        #     detmod,
        #     dataloaders,
        #     optimizer,
        #     dice_weight=1 - bce_weight,
        #     num_epochs=num_epochs,
        #     device=gpu_id,
        # )
        from functools import partial
        from entityseg.training.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d
        
        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)

        metricCallback = MetricCallback()
        trainer = Trainer(
            detmod,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=1,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        plot_losses(training_loss, validation_loss)
        
        detmod = trainer.model
        #if train_params["display_plots"]:
        #    display_preds_fpn3d(outputs)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        # if save_current_model:
        #     save_model(
        #         "detmod_gtacwe_quarter" + dt_string + ".pt",
        #         detmod,
        #         optimizer,
        #         torch_models_fullpath,
        #     )

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

        # model3d = train_unet3d(
        #     model3d,
        #     optimizer,
        #     scheduler,
        #     dataloaders,
        #     num_epochs=num_epochs,
        #     initial_lr=0.001,
        #     bce_weight=0.5,
        #     device=gpu_id,
        #     patch_size=patch_size,
        # )

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
        plot_losses(training_loss, validation_loss)

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

def display_preds_fpn3d(inputs, outputs, z_slice=8):
    [
        show_images(
            [
                inputs[idx, 0, z_slice, :].cpu().detach().numpy(),
                outputs[idx, 0, z_slice, :].cpu().detach().numpy(),
            ],
            figsize=(4, 4),
        )
        for idx in range(outputs.shape[0])
    ]
