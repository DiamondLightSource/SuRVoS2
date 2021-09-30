import copy
import os
import time
from collections import defaultdict
from typing import Callable
from datetime import datetime
import numpy as np
from loguru import logger

from entityseg.models.fpn_util import accuracy
from survos2.frontend.nb_utils import show_images
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from tqdm.notebook import tqdm, trange

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_fpn3d(
    model, dataloaders, optimizer_ft, dice_weight=0.5, num_epochs=10, device=0
):
    losses = []
    for ee in range(num_epochs):
        epoch_samples = 0

        for img_batch, label_batch in tqdm(dataloaders["train"]):
            optimizer_ft.zero_grad()
            # print(f"img_batch, label_batch {img_batch} {label_batch}")
            img_batch = img_batch.to(device).unsqueeze(1).float()
            label_batch = label_batch.to(device).unsqueeze(1).float()

            outputs, smax, fmaps = model.train_fwd(img_batch)
            pred = outputs
            target = label_batch
            pred = pred[:, 0:1, :]
            # print(pred.shape)

            bce = F.binary_cross_entropy_with_logits(pred, target)
            pred = F.sigmoid(pred)
            dice = dice_loss(pred, target)
            loss = bce * (1 - dice_weight) + dice * dice_weight

            # print(metrics)
            # print(f"Loss bce {bce} dice {dice} \n")
            losses.append(loss)

            # loss.requires_grad = True
            loss.backward()
            optimizer_ft.step()
        # print(losses)
    return model, outputs, losses


def save_model(model, optimizer, filename, torch_models_fullpath):
    if not os.path.exists(torch_models_fullpath):
        os.makedirs(torch_models_fullpath)

    def save_model(model_dictionary):
        checkpoint_directory = torch_models_fullpath
        file_path = os.path.join(checkpoint_directory, filename)
        torch.save(model_dictionary, file_path)

    model_dictionary = {
        "model_state": model.state_dict(),
        "model_optimizer": optimizer.state_dict(),
    }
    save_model(model_dictionary)


def display_pred(inputs, label_batch, pred, patch_size=(1, 224, 224)):
    # input_images_rgb = [
    #    x.transpose((1, 2, 0)).reshape(patch_size) for x in inputs.cpu().numpy()
    # ]
    input_images_rgb = inputs.cpu().numpy()
    target_masks_rgb = [((x > 0) * 1.0)[0, :].T for x in label_batch.cpu().numpy()]
    pred_rgb = [pred[x, 0:1, :].cpu().detach().numpy() for x in range(pred.shape[0])]
    pred_rgb = [x / np.max(x) for x in pred_rgb]
    pred_rgb = [x.reshape(patch_size[1:3]) for x in pred_rgb]

    [
        show_images([input_images_rgb[i], target_masks_rgb[i], pred_rgb[i]])
        for i in range(pred.shape[0])
    ]
    return input_images_rgb, pred_rgb, target_masks_rgb


def dataset_from_numpy(*ndarrays, device=None, dtype=torch.float32):
    tensors = map(torch.from_numpy, ndarrays)
    return TensorDataset(*[t.to(device, dtype) for t in tensors])


def score_dice(pred, targ, dim=1, smoothing=1.0, eps=1e-7) -> Tensor:
    pred = torch.FloatTensor(pred)
    targ = torch.FloatTensor(targ)
    pred, targ = pred.contiguous(), targ.contiguous()
    intersection = (pred * targ).sum(dim=dim).sum(dim=dim)
    cardinality = (
        pred.sum(dim=dim).sum(dim=dim) + targ.sum(dim=dim).sum(dim=dim) + smoothing
    )
    loss = (2.0 * intersection + smoothing) / cardinality

    return loss.mean()


def loss_dice(pred: Tensor, targ: Tensor, dim=2, smooth=1.0, eps=1e-7) -> Tensor:
    # pred, targ = pred.contiguous(), targ.contiguous()
    intersection = (pred * targ).sum(dim=dim).sum(dim=dim)
    cardinality = (
        pred.sum(dim=dim).sum(dim=dim) + targ.sum(dim=dim).sum(dim=dim) + smooth
    )

    loss = (2.0 * intersection + smooth) / cardinality
    loss = 1.0 - loss

    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def load_pretrained_model():
    torch_models_fullpath = "c:/work/experiments"
    checkpoint_directory = torch_models_fullpath
    file_path = os.path.join(checkpoint_directory, "resnetUnet_model_0511_b.pt")
    load_mod = True

    if load_mod:

        def load_model_parameters(full_path):
            checkpoint = torch.load(full_path)
            # print(model_load['model_state'])
            # print(model_load['model_optimizer'])
            return checkpoint

        checkpoint_file = "model.pt"
        full_path = os.path.join(checkpoint_directory, checkpoint_file)

        checkpoint = load_model_parameters(file_path)

        model.load_state_dict(checkpoint["model_state"])
        optimizer_ft.load_state_dict(checkpoint["model_optimizer"])

        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

        model.eval()


def predict(test_loader, model, device):
    inputs, label_batch = next(iter(test_loader))
    inputs = inputs.to(device)
    label_batch = label_batch.to(device)
    print(inputs.shape, label_batch.shape)
    pred = model(inputs)
    pred = F.sigmoid(pred)
    # pred = F.softmax(pred)
    pred_np = pred.data.cpu().numpy()
    print(pred_np.shape)

    return pred


def fpn3d_loss(preds, label_batch, bce_weight):
    bce = F.binary_cross_entropy_with_logits(preds, label_batch)
    pred = F.sigmoid(preds)
    dice = dice_loss(preds, label_batch)
    # loss = bce * (1 - dice_weight) + dice * dice_weight
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def unet3d_loss2(preds, label_batch, bce_weight):
    # smax = F.softmax(preds, dim=1)
    pred = F.sigmoid(preds)  # , dim=1)
    bce = F.binary_cross_entropy_with_logits(pred, label_batch)
    dice = dice_loss(pred, label_batch)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def unet3d_loss(preds, label_batch, bce_weight):
    # smax = F.softmax(preds, dim=1)
    pred = F.sigmoid(preds)  # , dim=1)
    bce = F.binary_cross_entropy_with_logits(pred, label_batch)
    dice = mt_losses.dice_loss(pred, label_batch)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def prepare_labels_fpn3d(label_batch, input_batch):
    label_batch = label_batch.to(device).unsqueeze(1).float()
    return label_batch


def prepare_labels_unet3d(labels, input_batch):
    # label_values = list(np.unique(labels).astype(np.uint32))
    # label_values.remove(0)
    # num_out = len(label_values)

    label_batch = torch.zeros(
        (
            input_batch.shape[0],
            2,
            input_batch.shape[-3],
            input_batch.shape[-2],
            input_batch.shape[-1],
        )
    )

    label_batch[:, 0, :] = labels.float()
    label_batch[:, 1, :] = labels.float()

    # for idx, label_no in enumerate(label_values):
    #     # print(label_no, label_batch.shape)
    #     # l = (label_batch[:,:,:,:] == 1) * 1.0
    #     label_batch[:, idx, :] = l.float()

    return label_batch


class TrainerCallback:
    def __init__(self):
        self.debug_verbose = False

    def on_train_begin(self):
        if self.debug_verbose:
            logger.debug("On train begin")

    def on_train_end(self):
        if self.debug_verbose:
            logger.debug("On train end")

    def on_val_begin(self):
        if self.debug_verbose:
            logger.debug("On validation begin")

    def on_val_loss(self):
        if self.debug_verbose:
            logger.debug("On validation loss")

    def on_val_end(self):
        if self.debug_verbose:
            logger.debug("On validation end")

    def on_epoch_begin(self):
        if self.debug_verbose:
            logger.debug("On epoch begin")

    def on_epoch_end(self):
        if self.debug_verbose:
            logger.debug("On epoch end")

    def on_batch_begin(self):
        if self.debug_verbose:
            logger.debug("On batch begin")

    def on_batch_end(self):
        if self.debug_verbose:
            logger.debug("On batch end")

    def on_loss_begin(self):
        if self.debug_verbose:
            logger.debug("On loss begin")

    def on_loss_end(self):
        if self.debug_verbose:
            logger.debug("On loss end")

    def on_step_begin(self):
        if self.debug_verbose:
            logger.debug("On step begin")

    def on_step_end(self):
        if self.debug_verbose:
            logger.debug("On step end")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        lr_scheduler: torch.optim.lr_scheduler,
        dataloaders: dict,
        callback: TrainerCallback,
        prepare_labels: Callable,
        num_out_channels: int,
        num_epochs: int,
        initial_lr: float,
        device: int,
        accumulate_iters: int = 8,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.num_epochs = num_epochs
        self.initial_lr = initial_lr
        self.device = device
        self.callback = callback
        self._epoch = 0
        self.prepare_labels = prepare_labels
        self.num_out_channels = num_out_channels
        self.accumulate_iters = accumulate_iters

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run(self):
        progressbar = trange(self.num_epochs, desc="Progress")
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        for i in progressbar:
            _ = self.callback.on_epoch_begin()
            self._epoch += 1
            self._train()

            if "val" in self.dataloaders:
                self._validate()

                # print(self.callback.validation_loss)
                if self.callback.validation_loss[-1] < best_loss:
                    best_loss = self.callback.validation_loss[-1]
                    best_model_weights = copy.deepcopy(self.model.state_dict())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            _ = self.callback.on_epoch_end()

        self.model.load_state_dict(best_model_weights)
        return (
            self.callback.training_loss,
            self.callback.validation_loss,
            self.learning_rate,
        )

    def _train(self):
        ###

        _ = self.callback.on_train_begin()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_lr()[0]
        self.model.train()
        num_steps = 0

        batch_iter = tqdm(
            enumerate(self.dataloaders["train"]),
            "Training",
            total=len(self.dataloaders["train"]),
            leave=False,
        )

        for i, (input_batch, label_batch) in batch_iter:
            input_batch = input_batch.float().to(self.device).unsqueeze(1)
            label_batch = self.prepare_labels(label_batch, input_batch)
            # print(input_batch.shape, label_batch.shape)
            label_batch = label_batch.to(self.device)
            preds = self.model(input_batch)  # [0]
            # print(preds[0].shape)
            preds_post = preds[:, 0 : self.num_out_channels, :, :, :]
            loss = self.criterion(preds_post, label_batch)
            loss_value = loss.item()
            ###
            _ = self.callback.on_loss_end(loss_value)
            loss.backward()
            if ((i + 1) % self.accumulate_iters == 0) or (
                i + 1 == len(self.dataloaders["train"])
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            num_steps += 1
            batch_iter.set_description(f"Training: (loss {loss_value:.4f})")

        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])
        batch_iter.close()
        self.training_loss = self.callback.on_train_end()

    def _validate(self):
        self.model.eval()
        batch_iter = tqdm(
            enumerate(self.dataloaders["val"]),
            "Validation",
            total=len(self.dataloaders["val"]),
            leave=False,
        )

        for i, (input_batch, label_batch) in batch_iter:
            with torch.no_grad():
                input_batch = input_batch.float().unsqueeze(1)
                label_batch = self.prepare_labels(label_batch, input_batch)
                input_batch, label_batch = (
                    input_batch.to(self.device),
                    label_batch.to(self.device),
                )
                preds = self.model(input_batch)[:, 0 : self.num_out_channels, :, :, :]
                loss = self.criterion(preds, label_batch)
                loss_value = loss.item()
                # print(loss_value)
                ###
                _ = self.callback.on_val_loss(loss_value)
                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")
        _ = self.callback.on_val_end()

        batch_iter.close()


class MetricCallback(TrainerCallback):
    def __init__(self):
        super(MetricCallback, self).__init__()
        self.training_loss = []
        self.validation_loss = []

    def on_train_begin(self):
        # self.training_loss = []
        pass

    def on_loss_end(self, loss):
        self.training_loss.append(loss)

    def on_train_end(self):
        print(f"Avg training loss: {np.mean(self.training_loss)}")
        return self.training_loss

    def on_val_loss(self, loss):
        self.validation_loss.append(loss)

    def on_val_end(self):
        print(f"Avg validation loss: {np.mean(self.validation_loss)}")
        return self.validation_loss


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AverageValueMetric(Metric):
    def __init__(self):
        self.values = []

    def __call__(self, value):
        self.values.append(value)
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return "Average"


default_criterion = {
    "cross_entropy": lambda model, X, y: F.cross_entropy(model(X), y, reduction="mean"),
    "mse": lambda model, X, y: 0.5 * F.mse_loss(model(X), y, reduction="mean"),
}


#########################################
def log_metrics(metrics: dict, epoch_samples: int, phase: str):
    outputs = []

    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("Metrics -  {} {}".format(phase, ", ".join(outputs)))

    # for metric in metrics:
    #    message += "\t{}: {}".format(metric.name(), metric.value())


def train_unet3d(
    model3d,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=2,
    initial_lr=0.001,
    bce_weight=0.5,
    patch_size=(48, 96, 96),
    device=0,
    validate=True,
):
    device = torch.device(device)
    print(f"Training U-net")
    for epoch in tqdm(range(1, num_epochs + 1)):

        start_time = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        model3d.train()
        train_loss_total = 0.0
        num_steps = 0

        for batch in dataloaders["train"]:
            input_batch, labels = batch
            label_values = list(np.unique(labels).astype(np.uint32))
            # label_values.remove(0)
            num_out = 2  # len(label_values)

            input_batch = input_batch.float().to(device).unsqueeze(1)

            label_batch = torch.zeros(
                (
                    input_batch.shape[0],
                    num_out,
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                )
            )

            label_batch[:, 0, :] = labels.float()
            label_batch[:, 1, :] = labels.float()

            #             for idx, label_no in enumerate(label_values):
            #                 #print(label_no, label_batch.shape)
            #                 l = (label_batch[:,:,:,:] == 1) * 1.0
            #                 label_batch[:, idx, :] = l.float()

            label_batch = label_batch.to(device)
            # print(input_batch.shape, label_batch.shape)
            preds = model3d(input_batch)[:, 0:2, :, :, :]

            # smax = F.softmax(preds, dim=1) # non overlap...
            pred = F.sigmoid(preds)  # , dim=1)
            # print(pred.shape)

            bce = F.binary_cross_entropy_with_logits(pred, label_batch)
            dice = mt_losses.dice_loss(pred, label_batch)
            loss = bce * bce_weight + dice * (1 - bce_weight)

            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            print(loss)

        train_loss_total_avg = train_loss_total / num_steps
        print(f"Total loss: {train_loss_total_avg}")

        model3d.eval()
        val_loss_total = 0.0
        num_steps = 0

        train_acc = accuracy(
            preds.cpu().detach().numpy(), label_batch.cpu().detach().numpy()
        )

        metric_fns = [
            mt_metrics.dice_score,
            mt_metrics.hausdorff_score,
            mt_metrics.precision_score,
            mt_metrics.recall_score,
            mt_metrics.specificity_score,
            mt_metrics.intersection_over_union,
            mt_metrics.accuracy_score,
        ]

        metric_mgr = mt_metrics.MetricManager(metric_fns)
        tqdm._instances.clear()

        #
        # Validation
        #

        if validate:
            for batch in tqdm(dataloaders["val"]):
                input_batch, label_batch = batch

                with torch.no_grad():

                    input_batch, labels = batch
                    num_out = np.unique(labels)

                    print(input_batch.shape, labels.shape, num_out, patch_size)
                    input_batch = input_batch.float().to(device).unsqueeze(1)

                    label_batch = torch.zeros(
                        (
                            input_batch.shape[0],
                            2,
                            patch_size[0],
                            patch_size[1],
                            patch_size[2],
                        )
                    )
                    label_batch[:, 0, :] = labels.float()
                    label_batch[:, 1, :] = labels.float()
                    label_batch = label_batch.to(device)

                    preds = model3d(input_batch)
                    pred = F.sigmoid(preds)  # , dim=1)
                    bce = F.binary_cross_entropy_with_logits(pred, label_batch)

                    dice = mt_losses.dice_loss(pred, label_batch)
                    loss = bce * bce_weight + dice * (1 - bce_weight)
                    val_loss_total += loss.item()

                num_steps += 1

            val_loss_total_avg = val_loss_total / num_steps

            print(
                "\nTrain loss: {:.4f}, Training Accuracy: {:.4f} ".format(
                    train_loss_total_avg, train_acc
                )
            )
            # print('Val Loss: {:.4f}, Validation Accuracy: {:.4f} '.format(val_loss_total_avg))
    return model3d


def train_unet3d2(
    model3d,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=2,
    initial_lr=0.001,
    bce_weight=0.5,
    patch_size=(64, 64, 64),
    device=0,
    validate=True,
):
    device = torch.device(device)

    for epoch in tqdm(range(1, num_epochs + 1)):

        start_time = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        model3d.train()
        train_loss_total = 0.0
        num_steps = 0

        for batch in dataloaders["train"]:
            input_batch, labels = batch
            label_values = list(np.unique(labels).astype(np.uint32))
            # label_values.remove(0)
            num_out = 2  # len(label_values)

            input_batch = input_batch.float().to(device).unsqueeze(1)

            label_batch = torch.zeros(
                (
                    input_batch.shape[0],
                    num_out,
                    patch_size[0],
                    patch_size[1],
                    patch_size[2],
                )
            )

            label_batch[:, 0, :] = labels.float()
            label_batch[:, 1, :] = labels.float()

            #             for idx, label_no in enumerate(label_values):
            #                 #print(label_no, label_batch.shape)
            #                 l = (label_batch[:,:,:,:] == 1) * 1.0
            #                 label_batch[:, idx, :] = l.float()

            label_batch = label_batch.to(device)
            # print(input_batch.shape, label_batch.shape)
            preds = model3d(input_batch)[:, 0:2, :, :, :]

            # smax = F.softmax(preds, dim=1) # non overlap...
            pred = F.sigmoid(preds)  # , dim=1)
            # print(pred.shape)

            bce = F.binary_cross_entropy_with_logits(pred, label_batch)
            dice = loss_dice(pred, label_batch)
            loss = bce * bce_weight + dice * (1 - bce_weight)

            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            print(loss)

        train_loss_total_avg = train_loss_total / num_steps
        print(f"Total loss: {train_loss_total_avg}")

        model3d.eval()
        val_loss_total = 0.0
        num_steps = 0

        train_acc = accuracy(
            preds.cpu().detach().numpy(), label_batch.cpu().detach().numpy()
        )

        tqdm._instances.clear()

        #
        # Validation
        #

        if validate:
            for batch in tqdm(dataloaders["val"]):
                input_batch, label_batch = batch

                with torch.no_grad():

                    input_batch, labels = batch
                    num_out = np.unique(labels)

                    print(input_batch.shape, labels.shape, num_out, patch_size)
                    input_batch = input_batch.float().to(device).unsqueeze(1)

                    label_batch = torch.zeros(
                        (
                            input_batch.shape[0],
                            2,
                            patch_size[0],
                            patch_size[1],
                            patch_size[2],
                        )
                    )
                    label_batch[:, 0, :] = labels.float()
                    label_batch[:, 1, :] = labels.float()
                    label_batch = label_batch.to(device)

                    preds = model3d(input_batch)
                    pred = F.sigmoid(preds)  # , dim=1)
                    bce = F.binary_cross_entropy_with_logits(pred, label_batch)

                    dice = dice_loss(pred, label_batch)
                    loss = bce * bce_weight + dice * (1 - bce_weight)
                    val_loss_total += loss.item()

                num_steps += 1

            val_loss_total_avg = val_loss_total / num_steps

            print(
                "\nTrain loss: {:.4f}, Training Accuracy: {:.4f} ".format(
                    train_loss_total_avg, train_acc
                )
            )
            # print('Val Loss: {:.4f}, Validation Accuracy: {:.4f} '.format(val_loss_total_avg))
    return model3d


def train_fpn(
    model, dataloaders, optimizer_ft, dice_weight=0.5, num_epochs=10, device=0
):
    loss_values = []
    for ee in range(num_epochs):

        metrics = defaultdict(float)
        epoch_samples = 0

        for img_batch, label_batch in iter(dataloaders["train"]):
            # print(f"img_batch, label_batch {img_batch} {label_batch}")

            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            # print(img_batch.shape, label_batch.shape)
            outputs, smax, fmap = model.train_fwd(img_batch)

            # print(outputs.shape, smax.shape, fmap.shape)
            # print(summary_stats(outputs.cpu().detach().numpy()))
            # argmaxed = torch.argmax(smax, dim=1).unsqueeze(1).float()
            # loss = calc_loss(outputs, label_batch, metrics)

            pred = outputs
            target = label_batch

            pred = pred[:, 0:1, :]
            # print(pred.shape)

            bce = F.binary_cross_entropy_with_logits(pred, target)
            pred = F.sigmoid(pred)
            dice = dice_loss(pred, target)

            loss = bce * (1 - dice_weight) + dice * dice_weight

            metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
            metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
            metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

            # print(metrics)
            # print(f"Loss {loss}\n")
            loss_values.append(loss.detach().cpu().numpy())
            optimizer_ft.zero_grad()
            # loss.requires_grad = True
            loss.backward()
            optimizer_ft.step()

        print(np.mean(loss_values))

    return model, outputs


def loss_calc(pred: Tensor, target: Tensor, metrics: dict, dice_weight=0.5) -> float:
    # criterion = default_criterion.get(criterion, criterion)
    # assert callable(criterion)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = loss_dice(pred, target)
    loss = bce * (1 - dice_weight) + dice * (dice_weight)

    metrics["bce"] += bce.data.cpu().numpy() * target.size(0)
    metrics["dice"] += dice.data.cpu().numpy() * target.size(0)
    metrics["loss"] += loss.data.cpu().numpy() * target.size(0)

    return loss


def train_model(
    model,
    optimizer,
    scheduler,
    dataloaders,
    num_epochs=25,
    dice_weight=0.5,
    freeze_backbone=False,
    device=0,
):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        start_time = time.time()

        for phase in ["train", "val"]:

            if phase == "train":
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("Learning rate: ", param_group["lr"])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, label_batch in dataloaders[phase]:
                inputs = inputs.to(device)
                label_batch = label_batch.to(device)
                # print(inputs.shape, label_batch.shape)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_calc(outputs, label_batch, metrics, dice_weight=0.8)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            log_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics["loss"] / epoch_samples

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - start_time
        print("{:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    print("Best validation loss: {:4f}\n".format(best_loss))
    model.load_state_dict(best_model_weights)

    return model


def train_detector_head(
    model3d,
    optimizer,
    dataloaders,
    checkpoint_directory,
    num_epochs=5,
    device=0,
    batch_size=1,
    defrost_after=None,
    num_samples_per_log_entry=200,
):

    from survos2.entity.models.head_cnn import make_classifications

    epochs, losses = [], []
    iters_sub, train_acc, val_acc = [], [], []
    criterion = nn.CrossEntropyLoss(
        # weight=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.5]).to(0)
    )
    weight_decay = 1e-4
    n = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        print(f"Epoch: {epoch}")
        best_validation_accuracy = 0

        if defrost_after is not None:
            if epochs > defrost_after:
                for param in model3d.parameters():
                    param.requires_grad = True

        for img, label in dataloaders["train"]:

            optimizer.zero_grad()
            var_input = img[0].float().to(device).unsqueeze(1)
            out = model3d.forward_pyr(var_input)
            # plt.figure()
            # plt.imshow(out[0][0,0,16,:].cpu().detach().numpy())
            _, class_logits = model3d.Classifier(out[0])

            target_label = torch.Tensor([label["labels"]]).long().to(device)
            loss = criterion(
                class_logits,
                target_label,
            )

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu().numpy() / batch_size)

            if n % num_samples_per_log_entry == 0:
                iters_sub.append(n)
                train_acc.append(
                    make_classifications(model3d, dataloaders["train"], device)[1]
                )
                val_acc.append(
                    make_classifications(model3d, dataloaders["val"], device)[1]
                )

            n += 1
            if val_acc[-1] > best_validation_accuracy:
                print(f"Better validation accuracy: {val_acc[-1]}")
                best_validation_accuracy = val_acc[-1]

                now = datetime.now()
                dt_string = now.strftime("%d%m_%H%M")

                checkpoint_fname = "detector_" + dt_string + ".pt"
                save_model(model3d, optimizer, checkpoint_fname, checkpoint_directory)
                print(f"Wrote checkpoint: {checkpoint_fname}")
        print(train_acc)
        epochs.append(epoch)
        losses.append(np.mean(epoch_losses))

    accuracies = {}
    accuracies["train"] = train_acc
    accuracies["val"] = val_acc

    return epochs, losses, accuracies
