import os
import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet152, resnet18, resnet34, resnet50
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
from torch import Tensor

import torchvision.utils
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict

import time
import copy

from loguru import logger
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class AverageLossMetric(Metric):
    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average...'
      


class TrainerCallback():
    def __init__(self): 
        self.debug_verbose = False
    def on_train_begin(self): 
        if self.debug_verbose:
            logger.debug("On train begin")
    def on_train_end(self):
        if self.debug_verbose:
            logger.debug("On train end")
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



def reverse_transform(inp):    
    #inp = inp.transpose((1, 2, 0))
    #inp = np.clip(inp, 0, 1)
    #inp = (inp * 255).astype(np.uint8)
    
    return inp

def dataset_from_numpy(*ndarrays, device=None, dtype=torch.float32):
    tensors = map(torch.from_numpy, ndarrays)
    return TensorDataset(*[t.to(device, dtype) for t in tensors])

def tensorboard(fpath):
    writer = SummaryWriter(fpath)


def train_detmod_cbs(model, optimizer, scheduler,  dataloaders, callback, num_epochs=10):
    fpath = 'runs/encdec'    
    writer = SummaryWriter(fpath)

    
    bce_weight=0.2
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    
    for epoch in range(num_epochs):
        print('\nEpoch: {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 5)

        since = time.time()

        metrics = defaultdict(float)
        epoch_samples = 0
        
        for img_batch, label_batch in iter(dataloaders['train']):
            #print(f"img_batch, label_batch {img_batch} {label_batch}")
            
            img_batch = img_batch.to(device)
            label_batch = label_batch.to(device)
            print(img_batch.shape, label_batch.shape)
            outputs, smax, fmap = model.train_fwd(img_batch)
            
            #print(outputs.shape, smax.shape, fmap.shape)
            #print(summary_stats(outputs.cpu().detach().numpy()))
            #argmaxed = torch.argmax(smax, dim=1).unsqueeze(1).float()   
            #loss = calc_loss(outputs, label_batch, metrics)
            
            pred = outputs
            target = label_batch


            writer.add_image('encdec', pred[0,0,:])

            pred = pred[:,0:1,:]
            print(pred.shape)
            
            bce = F.binary_cross_entropy_with_logits(pred, target)
            pred = torch.sigmoid(pred)
            dice = loss_dice(pred, target)
            loss = bce * bce_weight + dice * (1 - bce_weight)
        
            metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
            metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
            metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
            
            print(metrics)
            print(loss)

            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()

            cl = 1

            smax_arr = smax.detach().cpu().data.numpy()
        
        skip = callback.on_epoch_end()        

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    print('Best validation loss: {:4f}'.format(best_loss))
    #model.load_state_dict(best_model_weights)
    
    out = [outputs, smax, label_batch]
    return model, out


def detmod_fit(epochs, model, loss_fn, opt, data, callbacks=None, metrics=None):
    cb_handler = Callback(callbacks)
    cb_handler.on_train_begin()
    
    for epoch in range(epochs):

        model.train()
        cb_handler.on_epoch_begin()
        
        for xb,yb in data.train_dl:
        
            xb, yb = cb_handler.on_batch_begin(xb, yb)
            loss,_ = loss_batch(model, xb, yb, loss_fn, opt, cb_handler)
            if cb_handler.on_batch_end(loss): break
        
        # get validation dataloader
        if hasattr(data,'valid_dl') and data.valid_dl is not None:
        
            model.eval()
        
            with torch.no_grad():
                *val_metrics,nums = zip(*[loss_batch(model, xb, yb, loss_fn, metrics=metrics)
                                for xb,yb in data.valid_dl])
            val_metrics = [np.sum(np.multiply(val,nums)) / np.sum(nums) for val in val_metrics]
            
        else: val_metrics=None
        if cb_handler.on_epoch_end(val_metrics): break
        
    cb_handler.on_train_end()


def detmod_loss_batch(model, xb, yb, loss_fn, opt=None, cb_handler=None, metrics=None):
    out = model(xb)

    if cb_handler is not None: out = cb_handler.on_loss_begin(out)
    
    loss = loss_fn(out, yb)
    mets = [f(out,yb).item() for f in metrics] if metrics is not None else []
    
    if opt is not None:
    
        if cb_handler is not None: loss = cb_handler.on_backward_begin(loss)
        loss.backward()
        if cb_handler is not None: cb_handler.on_backward_end()
        opt.step()
        if cb_handler is not None: cb_handler.on_step_end()
        opt.zero_grad()
        
    return (loss.item(),) + tuple(mets) + (len(xb),)

# + fix validation
def train_model_cbs(model, optimizer, scheduler,  dataloaders, callback, num_epochs=25):
    callback.on_train_begin()                 

    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        
        print('\nEpoch: {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 5)

        since = time.time()

        for phase in ['train', 'val']:
            #for metric in metrics:
                    #        metric.reset()
            metrics = defaultdict(float)

            if phase == 'train':
                scheduler.step()

                for param_group in optimizer.param_groups:
                    print("Learning rate: ", param_group['lr'])

                model.train()  
                skip = callback.on_epoch_begin()      

            else:
                model.eval()   
    
            epoch_num_samples = 0

            for inputs, labels in dataloaders[phase]:
                skip = callback.on_batch_begin() 
        
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_calc(outputs, labels, metrics, bce_weight=0.1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                skip = callback.on_loss_end()     
                epoch_num_samples += inputs.size(0)

            skip = callback.on_batch_end()           
            log_metrics(metrics, epoch_num_samples, phase)

            epoch_loss = metrics['loss'] / epoch_num_samples

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        skip = callback.on_epoch_end()        

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    print('Best validation loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_weights)
    
    return model




def train_model(model, optimizer, scheduler,  dataloaders, num_epochs=25):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        for phase in ['train', 'val']:
            
            if phase == 'train':
                scheduler.step()

                for param_group in optimizer.param_groups:
                    print("Learning rate: ", param_group['lr'])

                model.train()  
            else:
                model.eval()   

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_calc(outputs, labels, metrics, bce_weight=0.8)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            log_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best validation loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_weights)

    return model


def loss_dice(pred : Tensor, targ : Tensor, dim=2, smooth = 1., eps=1e-7) -> Tensor:
    #pred, targ = pred.contiguous(), targ.contiguous()    
    intersection = (pred * targ).sum(dim=dim).sum(dim=dim)
    cardinality = pred.sum(dim=dim).sum(dim=dim) + targ.sum(dim=dim).sum(dim=dim) + smooth
    
    loss = (2. * intersection + smooth) / cardinality
    loss = 1.0 - loss

    return loss.mean()




default_criterion = {
    "cross_entropy":
        lambda model, X, y: F.cross_entropy(model(X), y, reduction="mean"),
    "mse":
        lambda model, X, y: 0.5 * F.mse_loss(model(X), y, reduction="mean"),
}


def loss_calc(pred : Tensor, target : Tensor, metrics : dict, bce_weight=0.5) -> float:
    #criterion = default_criterion.get(criterion, criterion)
    #assert callable(criterion)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = loss_dice(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    #for metric in metrics:
    #    metric(outputs, target, loss_outputs)
    return loss



def log_metrics(metrics : dict, epoch_samples : int, phase : str):
    outputs = []
    
    for k in metrics.keys():

        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("Metrics -  {} {}".format(phase, ", ".join(outputs)))

    #for metric in metrics:
    #            message += '\t{}: {}'.format(metric.name(), metric.value())




def load_pretrained_model():
    torch_models_fullpath = 'c:/work/experiments'
    checkpoint_directory = torch_models_fullpath
    file_path = os.path.join(checkpoint_directory, 'resnetUnet_model_0511_b.pt')
    load_mod = True

    if load_mod:

        def load_model_parameters(full_path):
            checkpoint = torch.load(full_path)

            #print(model_load['model_state'])
            #print(model_load['model_optimizer'])

            return checkpoint

        checkpoint_file = 'model.pt'
        full_path = os.path.join(checkpoint_directory, checkpoint_file) 

        checkpoint = load_model_parameters(file_path)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer_ft.load_state_dict(checkpoint['model_optimizer'])

        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']

        model.eval()


def predict(test_loader, model, device):
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    print(inputs.shape, labels.shape)
    pred = model(inputs)
    pred = F.sigmoid(pred)
    #pred = F.softmax(pred)
    pred_np = pred.data.cpu().numpy()
    print(pred_np.shape)

    return pred

# def transform_back(tens):
#     tens = tens.numpy().transpose((1, 2, 0))
    
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
    
#     tens = std * tens + mean
#     tens = np.clip(tens, 0, 1)
#     tens = (tens * 255).astype(np.uint8)
    
#     return tens


def loss_dice2(pred, targ, smooth = 1.):

    pred = pred.contiguous()
    targ = targ.contiguous()    
    
    intersection = (pred * targ).sum(dim=2).sum(dim=2)

    loss = (1 - ( (2. * intersection + smooth) / 
                (pred.sum(dim=2).sum(dim=2) + 
                targ.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

