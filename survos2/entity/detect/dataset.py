import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
from typing import List, Optional

from matplotlib.patches import Rectangle
from matplotlib import patches

import skimage
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.measure import find_contours,regionprops, label
from skimage.morphology import disk
from skimage.morphology import erosion, dilation
from skimage import img_as_ubyte, img_as_float
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from UtilityCluster import summary_stats
import kornia
import cc3d

from survos2.entity.sampler import sample_bvol

 
class BoundingVolumeDataset(Dataset):
    def __init__(self, image, bvols : List[np.ndarray], labels=List[List[int]],transform=None, plot_verbose=False):
        self.image, self.bvols, self.labels = image, bvols, labels
        self.transform = transform
        self.plot_verbose = plot_verbose

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        bvol = self.bvols[idx]
        label = self.labels[idx]
        image = sample_bvol(self.image, bvol)

        if self.transform:
            image = self.transform(image)
        iscrowd = torch.zeros((len(bvol),), dtype=torch.int64)

        if self.labels is None:
            self.labels = torch.ones(len(bvol), dtype=torch.int64)

        box_volume = image.shape[0] * image.shape[1] * image.shape[2]        
        target = {}    

        #target["points"] = pts
        target["boxes"] = bvol
        target["labels"] = label
        target["image_id"] = idx
        target["box_volume"] = box_volume
        target["iscrowd"] = iscrowd
        
        return image, target


class MaskedDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        
        self.input_images, self.target_masks = images, masks
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        
        if self.transform:
            image = self.transform(image)

        return [image, mask]
    

def setup_dataloaders_masked():
    train_set = MaskedDataset(images_train, masks_train, transform = image_trans)
    val_set = MaskedDataset(images_val, masks_train, transform = image_trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    batch_size = 32

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }


    return dataloaders
    


def prepare_bb(mask_orig, bbox_padding=(15,15), plot_verbose=False, 
               dilation_amt = 1, eccentricity_min=0.9, bbox_area_min=500):
    """Generate bounding boxes from a mask image
    Fits an ellipse to a connected component and gets bounding box (from skimage regionprops)

    Arguments:
        mask_orig {np.array} -- Mask 

    Keyword Arguments:
        bbox_padding {tuple} -- Size of 2d bounding box around centroid of connected component (default: {(15,15)})
        plot_verbose {bool} -- Lots of debug info? (default: {False})
        dilation_amt {int} -- Morphology on masks  (default: {1})
        eccentricity_thresh {float} -- Only accept ellipses with a certain eccentricity (default: {0.9})
        bbox_area_thresh {int} -- Only accept generated bounding boxes larger than a minimum size (default: {500})

    Returns:
        expanded_bboxes, label_img
    """
    
    mask_orig = erosion(mask_orig, disk(dilation_amt))
    #print(f"Reshaped mask shape {mask_orig.shape}")
    
    edge_crop = 4
    mask_orig[0:edge_crop,:] = 0
    mask_orig[mask_orig.shape[0]-edge_crop:mask_orig.shape[1],:] = 0
    mask_orig[:,0:edge_crop] = 0
    mask_orig[:,mask_orig.shape[1]-edge_crop:mask_orig.shape[1]] = 0


    if plot_verbose:
        plt.figure()
        plt.imshow(mask_orig)


    print(summary_stats(mask_orig))
    if plot_verbose:

        blobs_log = blob_log(mask_orig, max_sigma=30, num_sigma=10, threshold=.1)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(mask_orig)
        ax[1].imshow(mask_orig)

        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='green', linewidth=2, fill=False)
            ax[1].add_patch(c)
            
        ax[1].set_axis_off()

        plt.tight_layout()
        plt.show()

    contours = find_contours(mask_orig, 0.4)

    label_img = label((mask_orig > 0.5 ) )
        
    if plot_verbose:
        plt.figure()
        plt.imshow(label_img)
        plt.title("Mask orig")

    props = regionprops(label_img)
    print(len(props))
    #print(f"Eccentricity: {props[0].eccentricity}, thresh {eccentricity_min}")
    #print(f"BBox Area Min {props[0].bbox_area}, thresh {bbox_area_min}")

    if plot_verbose:
        
        image_label_overlay = label2rgb(label_img) #, image=bin_mask)
        fig, ax = plt.subplots(figsize=(10, 6))
        #ax.imshow(image_label_overlay)
        
        
        ax.imshow(mask_orig)

        bboxes = []
        for region in regionprops(label_img):
            # take regions with large enough areas
            if region.area >= 10:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
            
                bboxes.append([minr,minc,maxr, maxc])
            
                rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
            
                ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    bounding_boxes= [(prop.bbox, prop.centroid) for prop in props
                     if prop.eccentricity > eccentricity_min 
                     and prop.bbox_area > bbox_area_min]
    

    bb_only =  [prop.bbox for prop in props
                     if prop.eccentricity > eccentricity_min 
                     and prop.bbox_area > bbox_area_min]
    # TODO: eliminate detections on the border
    
    padx, pady = bbox_padding
    
    expanded_bboxes = [((0, bb[0]-padx, bb[1]-pady,bb[2]+padx,bb[3]+pady),centroid) for bb,centroid in bounding_boxes]

    if plot_verbose:
        fig, ax = plt.subplots()
        ax.imshow(mask_orig, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax.plot(contours[n][:, 1], contours[n][:, 0], linewidth=2)
        plt.title(len(expanded_bboxes))
    
    
    #return expanded_bboxes, label_img
    return np.array(bb_only)



class BBDataset(Dataset):
    """Bounding Boxes from Masks Detection Dataset
    
    Modeled on COCO
    Uses prepare_bb to convert masks into bounding boxes.

    Arguments:
        Dataset {[type]} -- [description]
    """
    
    def __init__(self, images, masks, transform=None, plot_verbose=False):
        
        self.input_images, self.target_masks = images, masks
        self.transform = transform
        self.plot_verbose = plot_verbose
        
    def __len__(self):
        
        return len(self.input_images)

    def __getitem__(self, idx):
        
        image = self.input_images[idx]
        
        composite_mask = self.target_masks[idx]
        print(composite_mask.shape)
        mask_orig = composite_mask.reshape((image.shape[0],image.shape[1]))
        print(f"Reshaped mask shape {mask_orig.shape}")

        #eliminate detections on border
        edge_crop = 20
        mask_orig[0:edge_crop,:] = 0
        mask_orig[image.shape[0]-edge_crop:image.shape[1],:] = 0
        mask_orig[:,0:edge_crop] = 0
        mask_orig[:,image.shape[1]-edge_crop:image.shape[1]] = 0
        
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask_orig)
            
        #expanded_bboxes, lbl_img = prepare_bb(image,mask_orig, dilation_amt=0)
        bbox_padding=(20,20)
        padx, pady = bbox_padding

       
        morph_amt = 1
        eccentricity_thresh=0.95
        bbox_area_thresh=50
        mask_orig = erosion(mask_orig, disk(morph_amt))

        if self.plot_verbose:
            
            blobs_log = blob_log(mask_orig, max_sigma=30, num_sigma=10, threshold=.1)

            fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(mask_orig)
            ax[1].imshow(mask_orig)
            ax[2].imshow(image[0,:])

            for blob in blobs_log:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='green', linewidth=2, fill=False)
                ax[1].add_patch(c)

            ax[1].set_axis_off()

            plt.tight_layout()
            plt.show()

        contours = find_contours(mask_orig, 0.9)

        lbl_img = skimage.measure.label(mask_orig)
        print(f"Label image shape: {lbl_img.shape}")
        props = regionprops(lbl_img)

        bounding_boxes= [(prop.bbox, prop.centroid) for prop in props
                         if prop.eccentricity < eccentricity_thresh 
                         and prop.bbox_area > bbox_area_thresh]

        
        expanded_bboxes = np.array([(0, bb[0]-padx, bb[1]-pady,bb[2]+padx,bb[3]+pady) 
                                    for bb,_ in bounding_boxes])
        expanded_bboxes = np.clip(expanded_bboxes,0,image.shape[1])
        bboxes = [b for b in expanded_bboxes]
        print(f"Expanded boxes {expanded_bboxes}")
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        num_objs = len(bboxes)
        print(f"Number of bboxes: {num_objs}")
        
        obj_ids = np.unique(lbl_img)
        obj_ids = obj_ids[1:]
        print(obj_ids)

        #plt.figure()
        #plt.imshow(lbl_img)

        individual_masks = [lbl_img==obj_id for obj_id in obj_ids]
        individual_masks = torch.as_tensor(individual_masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        if num_objs > 0:
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        else: 
            area = 0
            
        # TODO: multiple classes
        labels = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["expanded_bboxes"] = expanded_bboxes
        target["labels"] = labels
        target["masks"] = individual_masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["label_image"] = lbl_img
        target["mask"] = mask_orig
        
        return image, target, mask_orig

    #def __len__(self):
    #    return len(self.input_images)
    

#LabeledDataset
class SmallVolDataset(Dataset):
    
    def __init__(self, images, labels, transform=None):
        
        self.input_images, self.target_labels = images, labels
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        
        image = self.input_images[idx]
        label = self.target_labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label



def setup_dataloaders_smallvol():

    smallvol_image_trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.450, 0.450, 0.450], [0.225, 0.225, 0.225]) # imagenet
    ])


    smallvol_mask_trans = transforms.Compose([    
        transforms.ToTensor(),
    ])

    train_set = SmallVolDataset(slice_shortlist, labs, transform = image_trans)

    val_set = MaskedDataset(slice_val, masks_train, transform = image_trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }


    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    }

    return dataloaders



class SmallThreeChanDataset(torch.utils.data.Dataset):   
    def __init__(self, patch_data, labels, im_dim=(80,80), num_folds = 5, fold = 0,
                   mode = 'train', augment_data = False, random_state = 42,  
                 onechan_to_3chan=True, threechan = False):
        
        self.X, self.X_test, self.y, self.y_test =  train_test_split(patch_data, labels)
        
        self.fold = fold            
        self.num_folds = num_folds
        self.mode = mode
        self.random_state = random_state        
        self.im_dim = im_dim   

        self.train_data= self.X
        self.test_data = self.X_test
        self.train_labels = self.y
        self.test_labels = self.y_test
        
        print(len(self.X), len(self.y), len(self.X_test), len(self.y_test))
         
        self.onechan_to_3chan = False
        self.threechan = threechan        
        self.num_samples = patch_data.shape[0]
        
        if self.onechan_to_3chan:            
            self.mean = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1]).mean()
            
            self.std = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1]).std()
    
        else:
            if self.threechan:
                self.mean = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1],3)[0].mean()
                self.std = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1],3)[0].std()
          
            else:
                self.mean = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1],1)[0].mean()
                self.std = self.X.reshape(self.X.shape[0], im_dim[0], im_dim[1],1)[0].std()
          
        self.class_names = ["BG","Salient", "Thing1", "Thing2"]


        stratifier = StratifiedKFold(n_splits = self.num_folds, 
                                     random_state = self.random_state, shuffle = True)
        f1, f2, f3, f4, f5 = stratifier.split(self.X,self.y)
        folds = [f1, f2, f3, f4, f5]
              
        self.augment_data = augment_data
        self.train_idx = folds[self.fold][0]
        self.test_idx = folds[self.fold][1] 
        
        if self.mode == 'train':
            self.train = True
            
        else:
            self.train=False
       
        self.transforms=transforms.Compose([
            #transforms.CenterCrop(28),
            #transforms.RandomRotation(degrees=30),
            #torchvision.transforms.Resize((28,28)),
            #transforms.RandomHorizontalFlip(p=0.25),                           
            transforms.ToTensor(),
            #transforms.Normalize(mean=[self.mean],
            #                     std=[self.std]),
            transforms.Normalize((0.1307,), (0.3081,))
            ]) 
         
        self.transform = self.transforms
        self.transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize(self.mean, self.std),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.transform_test = transforms.Compose([
            #transforms.CenterCrop(28),                
            transforms.ToTensor(),
            #transforms.Normalize(self.mean, self.std),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)

        elif self.mode == 'test':
            return len(self.test_idx)
    
    
    def __getitem__(self, idx):
                
        if self.mode == 'train':
            
            label_tensor = torch.tensor(self.y[self.train_idx[idx]],  dtype=torch.long) 
            
            if self.onechan_to_3chan:
                main_img = self.X[self.train_idx[idx]].reshape(self.im_dim[0], self.im_dim[1])
            
                img_out = np.zeros((self.im_dim[0],self.im_dim[1], 3))
            
                img_out[:,:,0] = main_img
                img_out[:,:,1] = main_img
                img_out[:,:,2] = main_img
                
                img_out = img_as_ubyte(img_out)
                #print(idx, img_out.shape)
                #print(img_out)
                return_tuple = (self.preprocess_img(PIL.Image.fromarray(img_out.astype(np.uint8))), 
                                label_tensor)
            else:
                
                if self.threechan:
                    main_img = self.X[self.train_idx[idx]].reshape(self.im_dim[0], self.im_dim[1], 3) * 255      
                    
                else:
                            
                    main_img = self.X[self.train_idx[idx]].reshape(self.im_dim[0], self.im_dim[1]) * 255
                
                return_tuple = (self.preprocess_img(
                            PIL.Image.fromarray(main_img.astype(np.uint8))),
                            label_tensor)
            
        elif self.mode == 'test':
            
            label_tensor = torch.tensor(self.y[self.test_idx[idx]], dtype=torch.long)
            
            if self.onechan_to_3chan:
                
                main_img = self.X[self.test_idx[idx]].reshape(self.im_dim[0], self.im_dim[1])
                img_out = np.zeros((self.im_dim[0],self.im_dim[1], 3))
            
                img_out[:,:,0] = main_img
                img_out[:,:,1] = main_img
                img_out[:,:,2] = main_img
                img_out = img_as_ubyte(img_out)
                
                return_tuple = (self.preprocess_img(
                    PIL.Image.fromarray(img_out.astype(np.uint8))), label_tensor)
                
            else:
                if self.threechan:
                    main_img = self.X[self.train_idx[idx]].reshape(self.im_dim[0], self.im_dim[1], 3) * 255
                
                else:
                    main_img = self.X[self.test_idx[idx]].reshape(self.im_dim[0], self.im_dim[1])
                return_tuple = (self.preprocess_img(
                                PIL.Image.fromarray(main_img.astype(np.uint8))),
                                label_tensor)           
        return return_tuple 
    
    
    def preprocess_img(self, img):
        
        if self.augment_data == False:
            preprocessing =  self.transforms          
        else:    
            preprocessing = self.transforms
        
        return preprocessing(img).numpy()  



