import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm.notebook import tqdm
from unet import UNet3D

from survos2.entity.models.fpn import CNNConfigs
from survos2.entity.pipeline_ops import predict_agg_3d
from survos2.frontend.nb_utils import slice_plot, show_images
from torch import optim
from torch.optim import lr_scheduler
from survos2.entity.utils import load_model



class Head_TwoStage_Cls(nn.Module):
    def __init__(
        self,
        conv,
        n_input_channels,
        n_features,
        n_output_channels,
        n_classes,
        stride=1,
    ):
        super(Head_TwoStage_Cls, self).__init__()
        self.dim = conv.dim
        self.n_classes = n_classes
        self.relu = "relu"  # 'leaky_relu'
        self.norm = "batch_norm"  #'instance_norm'
        print(f"n_input_channels:{n_input_channels} ")
        print(
            f"n_output_channels: {n_output_channels} with number of classes: {n_classes}"
        )

        self.conv_1 = self._conv_block(n_input_channels, 16)
        self.conv_2 = self._conv_block(16, 32)
        self.conv_3 = self._conv_block(32, 64)
        self.conv_4 = self._conv_block(64, 128)

        self.conv_final = conv(
            n_features, n_output_channels, ks=2, stride=stride, pad=0, relu=None  # ks=3
        )

        self.linear_class = nn.Linear(750, self.n_classes)

    # ks=3, pad=1
    def _conv_block(self, in_c, out_c, stride=1, padding=0):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=(3, 3, 3), stride=stride, padding=padding
            ),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_1(x)
        # print(x.shape)
        x = self.conv_2(x)
        # print(x.shape)
        x = self.conv_3(x)
        # print(x.shape)
        x = self.conv_4(x)

        class_logits_raw = self.conv_final(x)
        axes = (0, 2, 3, 1) if self.dim == 2 else (0, 2, 3, 4, 1)
        class_logits = class_logits_raw.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(
            x.size()[0],
            -1,
        )
        return class_logits_raw, class_logits  # , linear_logits


def train_head(head_cls, dataloaders, num_epochs=10, batch_size=1, device=0):
    iters, losses = [], []
    iters_sub, train_acc, val_acc = [], [], []
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = optim.SGD(
        head_cls.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    n = 0

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        for img, label in dataloaders["train"]:
            input_feat_vol = img[0]
            input_feat_vol = (
                input_feat_vol.unsqueeze(0).float().to(device)
            )  # .unsqueeze(0)
            print(input_feat_vol.shape)
            class_logits_raw, class_logits = head_cls(input_feat_vol)

            loss = criterion(
                class_logits, torch.Tensor([label["labels"]]).long().to(device)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.detach().cpu().numpy() / batch_size)

            if n % 100 == 0:
                iters_sub.append(n)
                train_acc.append(calc_accuracy(head_cls, dataloaders["train"], device))
                val_acc.append(calc_accuracy(head_cls, dataloaders["val"], device))
            n += 1
        print(train_acc)

    return head_cls, losses, train_acc, val_acc


def make_classifications(model, loader, device):
    predictions = []
    correct, total = 0, 0

    logits = []
    for img, label in loader:
        var_input = img[0].float().to(device).unsqueeze(1)
        out = model.forward_pyr(var_input)
        class_logits, _ = model.Classifier(out[0])
        class_logits = class_logits.squeeze(2).squeeze(2).squeeze(2)
        target_label = torch.Tensor([label["labels"]]).long().to(device)
        logits.append(class_logits.cpu().detach().numpy())
        pred = torch.argmax(class_logits)
        correct += pred.eq(label["labels"].view_as(pred)).sum().item()
        total += int(label["labels"].shape[0])
        predictions.append(pred.cpu().numpy())
    predictions = np.array(predictions)
    return predictions, correct / total, logits



def setup_fpn_for_extraction(wf, checkpoint_file, gpu_id=0):
    gpu_id = 0
    batch_size = 1

    model3d, optimizer, lr_scheduler = prepare_fpn3d(gpu_id=gpu_id)
    
    if "torch_models_fullpath" in wf.params:
        full_path = os.path.join(wf.params["torch_models_fullpath"], checkpoint_file)
    else:
        full_path = checkpoint_file
        
    checkpoint = checkpoint = torch.load(full_path)
    model3d.load_state_dict(checkpoint["model_state"])
    #optimizer.load_state_dict(checkpoint["model_optimizer"])
    model3d = model3d.eval()

    return model3d


def setup_training(existing_model_file, torch_models_fullpath):
    model_fullpath = os.path.join(torch_models_fullpath, existing_model_file)
    print(f"Loading model: {model_fullpath}")
    detmod, optimizer, scheduler = prepare_fpn3d(
        existing_model_fname=model_fullpath, gpu_id=0
    )
    trainable_parameters = []

    for name, p in detmod.named_parameters():
        if "Fpn" not in name:
            trainable_parameters.append(p)

    len(trainable_parameters)
    optimizer = torch.optim.AdamW(
        params=trainable_parameters,
        lr=0.005,
    )
    return detmod, optimizer


def setup_training2(existing_model_file, wf):
    model_fullpath = os.path.join(
        wf.params["torch_models_fullpath"], existing_model_file
    )
    print(f" {model_fullpath}")
    detmod, optimizer, scheduler = prepare_fpn3d(
        existing_model_fname=model_fullpath, gpu_id=0
    )
    trainable_parameters = []

    # for name, p in detmod.named_parameters():
    #     if "Fpn" not in name:
    #         trainable_parameters.append(p)

    len(trainable_parameters)
    optimizer = torch.optim.AdamW(
        #params=trainable_parameters,
        lr=0.005,
    )
    return detmod, optimizer


def prepare_fpn_features(wf, checkpoint_file, dataloaders, gpu_id=0):
    model3d, optimizer, lr_scheduler = prepare_fpn3d(gpu_id=gpu_id)
    full_path = os.path.join(wf.params["torch_models_fullpath"], checkpoint_file)
    checkpoint = checkpoint = torch.load(full_path)
    model3d.load_state_dict(checkpoint["model_state"])
    #optimizer.load_state_dict(checkpoint["model_optimizer"])
    model3d.eval()
    feats = process_fpn3d_pred(model3d, dataloaders["train"], device=gpu_id)
    vec_mat = np.stack(feats)
    print(f"Feature matrix of shape {vec_mat.shape}")
    return vec_mat


def process_fpn3d_pred(model3d, dataloader, device=0, nb=True):
    from survos2.frontend.nb_utils import show_images
    progress_bar = tqdm
        
    f4 = []
    f3 = []
    f2 = []

    for batch in progress_bar(dataloader):
        input_all, labels = batch
        input_sample = input_all[0]
        print(input_sample.shape)
        #print(input_sample)

        input_sample_t = torch.FloatTensor(input_sample)
        with torch.no_grad():
            var_input = input_sample_t.float().to(device).unsqueeze(0).unsqueeze(0)
            out = model3d.forward_pyr(var_input)
            
            print(out[2].shape)
            print(out[3].shape)
            print(out[4].shape)
            f2.append(out[2].detach().cpu().numpy())
            f3.append(out[3].detach().cpu().numpy())
            f4.append(out[4].detach().cpu().numpy())

    #return f2,f3, f4
    f2_fts = [f[0, 0, :].reshape((8 * 8 * 32)) for f in f2]
    f3_fts = [f[0, 0, :].reshape((4 * 4 * 16)) for f in f3]
    f4_fts = [f[0, 0, :].reshape((2 * 2 * 8)) for f in f4]
    feats = np.array([np.hstack((a, b, c)) for a, b, c in zip(f2_fts, f3_fts, f4_fts)])

    return feats


def process_fpn3d_pred_(model3d, dataloader, device=0, nb=True):
    from survos2.frontend.nb_utils import show_images
    #pyr_feats = []
    progress_bar = tqdm
        
    f4 = []
    f3 = []
    f2 = []
    f1 = []
    f0 = []
    for batch in progress_bar(dataloader):
        input_all, labels = batch
        input_sample = input_all[0]

        input_sample_t = torch.FloatTensor(input_sample)
        with torch.no_grad():
            var_input = input_sample_t.float().to(device).unsqueeze(0).unsqueeze(0)
            out = model3d.forward_pyr(var_input)
            f0.append(out[0].detach().cpu().numpy())
            f1.append(out[1].detach().cpu().numpy())
            f2.append(out[2].detach().cpu().numpy())
            f3.append(out[3].detach().cpu().numpy())
            f4.append(out[4].detach().cpu().numpy())

    # return f0, f1, f2,f3, f4
    f2_fts = [f[0, 0, :].reshape((8 * 8 * 32)) for f in f2]
    f3_fts = [f[0, 0, :].reshape((4 * 4 * 16)) for f in f3]
    f4_fts = [f[0, 0, :].reshape((2 * 2 * 8)) for f in f4]
    feats = np.array([np.hstack((a, b, c)) for a, b, c in zip(f2_fts, f3_fts, f4_fts)])

    return feats


def prepare_fpn3d(existing_model_fname=None, gpu_id=0):
    cf = CNNConfigs("mymodel")
    #metrics = defaultdict(float)
    #epoch_samples = 0
    device = torch.device(gpu_id)
    print(f"Device {device}")
    print(f"Dim: {cf.dim} {cf.num_seg_classes}")
    from survos2.entity.models.detNet2 import detNet
    detmod = detNet(cf).to(device)

    if existing_model_fname is not None:
        print(existing_model_fname)
        detmod = load_model(detmod, str(existing_model_fname))

    detmod = detmod.train()

    # optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,
    #                                 detmod.parameters()), lr=1e-3)
    # exp_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, num_epochs)

    optimizer = optim.AdamW(
        detmod.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.21)

    return detmod, optimizer, scheduler



def display_fpn3d_pred(model3d, dataloaders, device=0):
    import torch.nn.functional as F
    from survos2.frontend.nb_utils import show_images, summary_stats

    for batch in tqdm(dataloaders["val"]):

        input_samples, labels = batch

        with torch.no_grad():
            var_input = input_samples.float().to(device).unsqueeze(1)
            preds = model3d(var_input)[0]
            out = torch.sigmoid(preds)
            out_arr = out.detach().cpu().numpy()

            print(var_input.shape)
            print(out_arr.shape)
            print(summary_stats(out_arr))

            out_arr_proc = out_arr.copy()
            show_images(
                [input_samples[0, i * 10, :, :].numpy() for i in range(1, 4)],
                figsize=(3, 3),
            )

        show_images(
            [1.0 - out_arr_proc[0, 1, i * 8, :, :] for i in range(1, 4)], figsize=(3, 3)
        )


