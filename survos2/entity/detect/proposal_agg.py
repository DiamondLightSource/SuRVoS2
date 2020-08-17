"""
Aggregation


Uses torchio

"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchio import IMAGE, LOCATION
from torchio.data.inference import GridSampler, GridAggregator

from torchvision import datasets, models, transforms

import torchio as tio
import torch.nn.functional as F


def predict_and_agg(model, input_array, patch_size=(1,224,224), patch_overlap=(0,16,16), batch_size=1, stacked_3chan=False, extra_unsqueeze=True):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tens = torch.FloatTensor(input_array)
    print(img_tens.shape)

    one_subject = tio.Subject(
            img=tio.Image(tensor=img_tens, label=tio.INTENSITY),
            label=tio.Image(tensor=img_tens, label=tio.LABEL)
    )

    img_dataset = tio.ImagesDataset([one_subject,])
    img_sample = img_dataset[-1]

    grid_sampler = GridSampler(img_sample, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(grid_sampler)

    input_tensors = []
    output_tensors = []

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            
            #input_tensor = patches_batch[IMAGE] 
            #locations = patches_batch[LOCATION]
            
            input_tensor = patches_batch['img']['data'] 
            location = patches_batch[LOCATION]
        
            print(f"Input tensor {input_tensor.shape}")
            
            inputs_t = input_tensor.squeeze(1)
            inputs_t = inputs_t.to(device)

            if stacked_3chan:
                inputs_t = torch.stack([inputs_t[:,0,:,:],inputs_t[:,0,:,:],inputs_t[:,0,:,:]], axis=1)
            else:
                inputs_t = inputs_t[:,0:1,:,:]
    
            print(f"inputs_t: {inputs_t.shape}")
            
            pred = model(inputs_t)
            #pred = torch.sigmoid(pred[0])
            pred = F.sigmoid(pred)
        
            #pred = pred.data.cpu()
            print(f"pred: {pred.shape}")

            if extra_unsqueeze:
                output = pred.unsqueeze(1)
            #output = pred.squeeze(0)
            
            input_tensors.append(input_tensor)
            output_tensors.append(output)

            print(output.shape, location)
            aggregator.add_batch(output, location)
            
    output_tensor = aggregator.get_output_tensor()
    print(input_tensor.shape,output_tensor.shape)

    return output_tensor