import json
from datetime import datetime
from pprint import pprint
import matplotlib.pyplot as plt
import torch
from survos2.entity.pipeline_ops import save_model
from survos2.entity.patches import load_patch_vols, prepare_dataloaders
from survos2.entity.models.head_cnn import (
    display_fpn3d_pred,
    prepare_fpn3d,
)


def train_oneclass_detseg(
    train_v_class1,
    project_file,
    wf_params,
    model_type="fpn3d",
    gpu_id=0,
    model=None,
    num_epochs=1,
    bce_weight=0.3,  #0.7
    initial_lr=0.01,
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
        "display_plots": True,
        "torch_models_fullpath": wf_params["torch_models_fullpath"],        
    }

    return train_all(train_params, model, bce_weight=bce_weight, initial_lr=initial_lr)


def train_twoclass_detseg(
    wf,
    training_vols,
    project_file,
    model_type="fpn3d",
    gpu_id=0,
    num_epochs=1,
    bce_weight=0.3,  #0.7
    initial_lr=0.01,
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
        "bce_weight" : bce_weight, 
        "initial_lr" :initial_lr
    }

    class1_model_file = train_all(train_params, bce_weight=bce_weight, initial_lr=initial_lr)

    print(f"Using training volumes: {train_v_class2[0]}")
    
    gpu_id = 0

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

    class2_model_file = train_all(train_params, bce_weight=bce_weight, initial_lr=initial_lr)
    return [class1_model_file, class2_model_file]


def load_model(file_path):
    def load_model_parameters(full_path):
        return torch.load(full_path)


def plot_losses(training_loss, validation_loss):
    plt.figure()
    plt.plot(training_loss)
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.figure()
    plt.plot(validation_loss)
    plt.title("Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")


def train_all(
    train_params,
    patch_size=(64, 64, 64),
    model=None,
    batch_size=1,
    bce_weight=0.3,
    initial_lr=0.01,
    display_plots=False,
):
    model_type = train_params["model_type"]
    gpu_id = train_params["gpu_id"]
    save_current_model = train_params["save_current_model"]
    num_epochs = train_params["num_epochs"]
    torch_models_fullpath = train_params["torch_models_fullpath"]

    # prepare patch dataset
    img_vols, label_vols = load_patch_vols(train_params["train_vols"])
    dataloaders = prepare_dataloaders(img_vols, label_vols, train_params["model_type"])

    if model_type == "fpn3d":
        model3d, optimizer, scheduler = prepare_fpn3d(gpu_id=train_params["gpu_id"])

    elif model_type == "unet3d":
        from survos2.entity.models.unet3d import prepare_unet3d, display_unet_pred

        model3d, optimizer, scheduler = prepare_unet3d(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )
    elif model_type == "vnet":
        from survos2.entity.models.vnet import prepare_vnet

        model3d, optimizer, scheduler = prepare_vnet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )
    elif model_type == "xunet":
        from survos2.entity.models.x_unet import prepare_xunet

        model3d, optimizer, scheduler = prepare_xunet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "unet":
        from survos2.entity.models.monai_nets import prepare_unet

        model3d, optimizer, scheduler = prepare_unet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "vnet_monai":
        from survos2.entity.models.monai_nets import prepare_vnet_monai

        model3d, optimizer, scheduler = prepare_vnet_monai(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "dynunet":
        from survos2.entity.models.monai_nets import prepare_dynunet

        model3d, optimizer, scheduler = prepare_dynunet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "attention_unet":
        from survos2.entity.models.monai_nets import prepare_attention_unet

        model3d, optimizer, scheduler = prepare_attention_unet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "swin_unetr":
        from survos2.entity.models.monai_nets import prepare_swin_unetr

        model3d, optimizer, scheduler = prepare_swin_unetr(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "unetplusplus":
        from survos2.entity.models.monai_nets import prepare_unetplusplus

        model3d, optimizer, scheduler = prepare_unetplusplus(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "segresnet":
        from survos2.entity.models.monai_nets import prepare_SegResNet

        model3d, optimizer, scheduler = prepare_SegResNet(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )
    elif model_type == "segresnetvae":
        from survos2.entity.models.monai_nets import prepare_SegResNetVAE

        model3d, optimizer, scheduler = prepare_SegResNetVAE(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
    )

    if model is not None:
        model3d = model

    if model_type == "xunet":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "xunet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "attention_unet" or model_type == "dynunet":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.001,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "attention_unet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "segresnet":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.001,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "segresnet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "segresnetvae":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.001,
            num_out_channels=2,
            device=gpu_id,
            model_output_as_list = True
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "segresnet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "unet":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "unet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "vnet_monai":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.005,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "vnet_monai_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "swin_unetr":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "swin_unetr_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "unetplusplus":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
            model_output_as_list = True
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "unetplusplus_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "vnet":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.001,
            num_out_channels=2,
            device=gpu_id,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            from survos2.entity.models.head_cnn import display_vnet_pred

            display_vnet_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "vnet_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "fpn3d":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
            model_output_as_list = False
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            display_fpn3d_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "fpn3d_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")
    elif model_type == "unet3d":
        from survos2.entity.models.unet3d import prepare_unet3d, display_unet_pred

        model3d, optimizer, scheduler = prepare_unet3d(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )

        from functools import partial
        from survos2.entity.trainer import (
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
        model3d = trainer.model

        if train_params["display_plots"]:
            display_unet_pred(model3d, dataloaders, device=gpu_id)
            plot_losses(training_loss, validation_loss)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "unet3d_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")

    return model_file


def train_all2(
    train_params,
    patch_size=(64, 64, 64),
    model=None,
    batch_size=1,
    bce_weight=0.3,
    initial_lr=0.01,
    display_plots=False,
):

    model_type = train_params["model_type"]
    gpu_id = train_params["gpu_id"]
    save_current_model = train_params["save_current_model"]
    num_epochs = train_params["num_epochs"]
    torch_models_fullpath = train_params["torch_models_fullpath"]

    # prepare patch dataset
    img_vols, label_vols = load_patch_vols(train_params["train_vols"])
    dataloaders = prepare_dataloaders(img_vols, label_vols, train_params["model_type"])

    if model_type == "fpn3d":
        model3d, optimizer, scheduler = prepare_fpn3d(gpu_id=train_params["gpu_id"])
    # if model is provided use that

    elif model_type == "unet3d":
        from survos2.entity.models.unet3d import prepare_unet3d, display_unet_pred

        model3d, optimizer, scheduler = prepare_unet3d(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )

    if model is not None:
        model3d = model

    if model_type == "fpn3d":
        from functools import partial
        from survos2.entity.trainer import Trainer, MetricCallback, fpn3d_loss, prepare_labels_fpn3d

        fpn3d_criterion = partial(fpn3d_loss, bce_weight=bce_weight)
        metricCallback = MetricCallback()
        trainer = Trainer(
            model3d,
            optimizer,
            fpn3d_criterion,
            scheduler,
            dataloaders,
            metricCallback,
            prepare_labels=prepare_labels_fpn3d,
            num_epochs=num_epochs,
            initial_lr=0.01,
            num_out_channels=2,
            device=gpu_id,
            model_output_as_list=True,
        )

        training_loss, validation_loss, learning_rate = trainer.run()
        model3d = trainer.model

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if display_plots:
            plot_losses(training_loss, validation_loss)
            display_fpn3d_pred(model3d, dataloaders, device=gpu_id)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "fpn3d_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")

    if model_type == "unet3d":
        from survos2.entity.models.unet3d import prepare_unet3d, display_unet_pred

        model3d, optimizer, scheduler = prepare_unet3d(
            existing_model_fname=None, device=gpu_id, initial_lr=initial_lr
        )

        from functools import partial
        from survos2.entity.trainer import (
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
        model3d = trainer.model

        if train_params["display_plots"]:
            display_unet_pred(model3d, dataloaders, device=gpu_id)
            plot_losses(training_loss, validation_loss)

        now = datetime.now()
        dt_string = now.strftime("%d%m_%H%M")

        if save_current_model:
            model_file = "unet3d_fullblob" + dt_string + ".pt"
            save_model(model_file, model3d, optimizer, torch_models_fullpath)
            print(f"Saved model {model_file}")

    return model_file
