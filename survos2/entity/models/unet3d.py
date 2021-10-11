import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm.notebook import tqdm
from unet import UNet3D
from torch import optim
from torch.optim import lr_scheduler
from entityseg.training.utils import load_model

def prepare_unet3d(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):

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


def prepare_unet3d2(num_epochs=2, initial_lr=0.001, device=0):
    device = torch.device(device)

    model3d = (
        UNet3D(
            normalization="batch",
            preactivation=True,
            residual=True,
            num_encoding_blocks=3,
            upsampling_type="trilinear",
        )
        .to(device)
        .eval()
    )

    # optimizer = optim.Adam(model3d.parameters(), lr=initial_lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, num_epochs)

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.51)

    return model3d, optimizer, scheduler


def display_unet_pred(model3d, dataloaders, max_samples_display=10, device=0):
    import torch.nn.functional as F
    from survos2.frontend.nb_utils import show_images, summary_stats

    count = 0
    for batch in tqdm(dataloaders["val"]):
        input_samples, labels = batch
        with torch.no_grad():
            var_input = input_samples.float().to(device).unsqueeze(1)
            preds = model3d(var_input)
            out = F.sigmoid(preds)
            out_arr = out.detach().cpu().numpy()

            print(var_input.shape)
            print(out_arr.shape)
            print(summary_stats(out_arr))

            out_arr_proc = out_arr.copy()
            show_images(
                [input_samples[0, i * 10, :, :].numpy() for i in range(1, 4)],
                figsize=(3, 3),
            )
            # show_images([1.0 - out_arr_proc[0,0,i * 8,:,:]  for i in range(1,4)])

        show_images(
            [1.0 - out_arr_proc[0, 1, i * 8, :, :] for i in range(1, 4)], figsize=(3, 3)
        )
        count += 1
        if count > max_samples_display:
            return
