import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from torch.optim import lr_scheduler


def prepare_vnet_monai(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import VNet

    device = torch.device(device)
    model3d = VNet(spatial_dims=3, in_channels=1, out_channels=1, act="elu")

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

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


def prepare_unet(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import Unet

    device = torch.device(device)
    model3d = Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler


def prepare_attention_unet(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import AttentionUnet

    device = torch.device(device)
    model3d = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler


def prepare_dynunet(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import DynUNet

    device = torch.device(device)

    strides = (1, 2, 2, 2, 2, 2)
    model3d = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 3, 3, 3, 3, 3),
        strides=strides,
        upsample_kernel_size=strides[1:],
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler


def prepare_swin_unetr(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import AttentionUnet, SwinUNETR, UNETR, BasicUNetPlusPlus

    device = torch.device(device)
    model3d = SwinUNETR(
        (64, 64, 64),
        in_channels=1,
        out_channels=1,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
    )

    # model3d = UNETR(in_channels=1, out_channels=1, img_size=(64,64,64), feature_size=32, norm_name='batch')

    # model3d = BasicUNetPlusPlus(spatial_dims=3,  in_channels=1, out_channels=1,
    #     features=(32, 32, 64, 128, 256, 32),
    #     deep_supervision=False,
    #     act=('LeakyReLU', {'negative_slope': 0.1, 'inplace': True}),
    #     norm=('instance', {'affine': True}),
    #     bias=True,
    #     dropout=0.0,
    #     upsample='deconv')

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

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


def prepare_unetplusplus(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import BasicUNetPlusPlus

    device = torch.device(device)

    model3d = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32),
        deep_supervision=False,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        # norm=('instance', {'affine': True}),
        norm={"batch", {"affine": True}},
        bias=True,
        dropout=0.0,
        upsample="deconv",
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

    optimizer = optim.AdamW(
        model3d.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.51)

    return model3d, optimizer, scheduler


def prepare_SegResNet(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import SegResNet

    device = torch.device(device)

    model3d = SegResNet(
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=1,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout_prob=None,
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

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


def prepare_SegResNetVAE(existing_model_fname=None, num_epochs=2, initial_lr=0.001, device=0):
    from survos2.entity.utils import load_model
    from monai.networks.nets import SegResNetVAE

    device = torch.device(device)

    model3d = SegResNetVAE(
        input_image_size=(64, 64, 64),
        spatial_dims=3,
        init_filters=8,
        in_channels=1,
        out_channels=1,
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        dropout_prob=None,
    )

    if existing_model_fname is not None:
        model3d = load_model(existing_model_fname, model3d)

    model3d = model3d.to(device).eval()

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
