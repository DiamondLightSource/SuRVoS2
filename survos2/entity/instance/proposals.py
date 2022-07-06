import os
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

from unet import UNet3D

from survos2.entity.pipeline_ops import predict_agg_3d
from survos2.entity.instance.det import make_augmented_entities

from survos2.entity.models.head_cnn import prepare_fpn3d
from survos2.frontend.nb_utils import slice_plot, show_images
from survos2.entity.anno.pseudo import make_pseudomasks, make_anno, generate_annotation_volume
from survos2.entity.components import filter_proposal_mask, filter_small_components
from survos2.entity.train import train_oneclass_detseg, train_twoclass_detseg

from survos2.entity.models.unet3d import prepare_unet3d
from survos2.entity.utils import load_model
from survos2.entity.patches import make_patches

class ProposalSegmentor:
    def __init__(self, wf, project_file, roi_name, padding, bg_mask_all, patch_size=(64,64,64)):
        self.roi_name = roi_name
        self.wf = wf
        self.bg_mask_all = bg_mask_all
        self.project_file = project_file
        proposal_dir = self.wf.params['outdir']
        self.padding = padding
        self.patch_size = patch_size
    
    def debug_plot(self):
        print(self.wf.vols[0].shape, self.wf.locs.shape, self.wf.bg_mask.shape)
        slice_plot(self.wf.vols[0],self.wf.locs, self.wf.bg_mask, (50,30,40), figsize=(15,15))
        
    def generate_annotation_volume(self, entity_meta, gt_proportion=1.0,
                                    padding=(64,64,64), 
                                    generate_random_bg_entities=False, 
                                    num_before_masking=0, 
                                    stratified_selection=True,
                                    acwe=False,
                                    class_proportion={0:1, 1: 1, 2: 1.0, 5:1}):
        self.anno_masks, self.anno_all, self.gt_entities, self.random_bg_entities =  generate_annotation_volume(self.wf, 
                                                                    entity_meta,
                                                                    gt_proportion=1.0,
                                                                    padding=padding,
                                                                    generate_random_bg_entities=generate_random_bg_entities,
                                                                    num_before_masking=num_before_masking,
                                                                    stratified_selection=stratified_selection,
                                                                    acwe=acwe,
                                                                    class_proportion = class_proportion)
        

#class_proportion = {0:0.5, 1: 1, 2: 1, 5:0.3}) 
#class_proportion = {0:0.5, 1: 0.8, 2: 0.8, 5:0.25 }) #{0:0.3, 1: 1.0, 2: 1.0, 5: 0.1})
    def setup_combined_anno(self):
        #combined_anno  = anno_masks[0]['mask'] + anno_masks[1]['mask'] + anno_masks[2]['mask']  
        self.combined_anno_class1  = self.anno_masks['0']['mask'] + self.anno_masks['1']['mask'] + self.anno_masks['2']['mask'] #+ anno_masks['5']['mask']
        self.combined_anno_class2  =  self.anno_masks['5']['mask']
        self.combined_anno_all = self.combined_anno_class1 + self.combined_anno_class2
        #combined_shell_anno  = anno_masks['0']['shell_mask'] + anno_masks['1']['shell_mask'] + anno_masks['2']['shell_mask'] + anno_masks['5']['shell_mask']
        slice_plot(self.combined_anno_all, None, None, (50,50,50))

    def make_training_volumes(self, num_training_patches=None, num_augs=1):
        if num_training_patches:
            rng = np.random.default_rng()
            selected_entities = rng.choice(self.augmented_entities, num_training_patches)
        else:
            selected_entities = self.augmented_entities
        self.train_v_all = make_patches(self.wf, selected_entities, self.wf.params['outdir'], 
                               proposal_vol=(self.combined_anno_all > 0) * 1.0, 
                               padding=self.padding, num_augs=num_augs, max_vols=-1, plot_all=True, patch_size=self.patch_size)
        
    def make_multiclass_training_volumes(self, num_training_patches=None, num_augs=1):
        if num_training_patches:
            rng = np.random.default_rng()
            selected_entities = rng.choice(self.augmented_entities, num_training_patches)
        else:
            selected_entities = self.augmented_entities
        self.train_v_class1 = make_patches(self.wf, selected_entities, self.wf.params['outdir'], 
                               proposal_vol=(self.combined_anno_class1 > 0) * 1.0, 
                               padding=self.padding, num_augs=num_augs, max_vols=-1, patch_size=self.patch_size)
        self.train_v_all = make_patches(self.wf, selected_entities, self.wf.params['outdir'], 
                               proposal_vol=(self.combined_anno_all > 0) * 1.0, 
                               padding=self.padding, num_augs=num_augs, max_vols=-1, patch_size=self.patch_size)
        self.train_v_class2 = make_patches(self.wf, selected_entities, self.wf.params['outdir'], 
                               proposal_vol=(self.combined_anno_class2 > 0) * 1.0, #combined_anno_class2, 
                               padding=self.padding, num_augs=num_augs, max_vols=-1, patch_size=self.patch_size)
#         train_v_class2 = make_patches(wf, augmented_entities, wf.params['outdir'], 
#                                proposal_vol=((bg_mask_all +combined_anno_all) > 0) * 1.0, #combined_anno_class2, 
#                                padding=padding, num_augs=0, max_vols=-1, plot_all=True)
    def setup_augmented_points(self):
        self.aug_pts = np.concatenate((self.gt_entities,self.random_bg_entities))
        self.augmented_entities = make_augmented_entities(self.aug_pts)
        slice_plot(self.wf.vols[0], self.augmented_entities, None, (50,300,300), unique_color_plot=True)

    def train_model(self, num_epochs=2, model_type="fpn3d"):
        self.class1_model_file = train_oneclass_detseg(self.train_v_all, 
                                    self.project_file, 
                                    self.wf.params,
                                    num_epochs=num_epochs, 
                                    model_type=model_type,
                                    bce_weight=0.5)
        
    def train_multiple_class_model(self, num_epochs=2):
        self.class2_model_file = train_oneclass_detseg(self.train_v_class2, self.project_file, self.wf.params,num_epochs=num_epochs)
        training_vols = [self.train_v_class1, self.train_v_all]
        self.class1_model_file, self.class2_model_file = train_twoclass_detseg(self.wf, training_vols, self.project_file, num_epochs=num_epochs)
    def setup_model_files(self, class1_model_file, class_all_model_file, class2_model_file):
        self.class1_model_file = class1_model_file  
        self.class_all_model_file = class_all_model_file
        self.class2_model_file = class2_model_file
    def make_proposal(self, threshold_devs=2, patch_size=(64,64,64), patch_overlap=(32,32,32), overlap_mode="average", invert=True, model_type="fpn3d"):
        self.class1_proposal_fname, self.class1_proposal_thresh, self.class1_proposal = make_and_save_single_proposal(self.wf, self.class1_model_file, 
                                                         self.roi_name, 
                                                         patch_size=patch_size,
                                                         patch_overlap=patch_overlap,
                                                         overlap_mode=overlap_mode,
                                                         threshold_devs=threshold_devs,
                                                               invert=True, 
                                                               model_type=model_type)

    def make_multiclass_proposal(self, threshold_devs=2, patch_overlap=(32,32,32), overlap_mode="average"):
        (self.class1_proposal_fname, 
        self.class1_proposal_thresh, 
        self.class_all_proposal_fname, 
        self.class_all_proposal_thresh, 
        self.class1_proposal, 
        self.class_all_proposal) = make_prediction_proposal(self.wf,     
                                [self.class1_model_file, self.class2_model_file], 
                                self.roi_name, 
                                threshold_devs=threshold_devs,
                                patch_overlap=patch_overlap,
                                overlap_mode=overlap_mode)
    
        self.class2_proposal_fname, self.class2_proposal_thresh, self.class2_proposal = make_and_save_single_proposal(self.wf, self.class2_model_file, 
                                                         self.roi_name, 
                                                         patch_overlap=patch_overlap,
                                                         overlap_mode=overlap_mode,
                                                         threshold_devs=threshold_devs,
                                                               invert=True)
    def postprocess_multiclass_proposals(self, threshold_devs=2):
        self.class1_f, self.class1_f_s, self.class2_f_s_d = postprocess_class1_proposal(self.wf, 
                                                                         self.class1_proposal, 
                                                                         self.class_all_proposal, 
                                                                         self.class2_proposal,
                                                                         threshold_devs=threshold_devs)
        self.combined_2, self.class2_f_masked = postprocess_class2_proposal(self.wf, 
                                                                  self.class1_proposal, 
                                                                  self.class_all_proposal, 
                                                                  self.class2_proposal, 
                                                                  self.class2_f_s_d, 
                                                                  threshold_devs=threshold_devs)
        
def make_prediction_proposal(
    wf,
    proposal_model,
    roi_name,
    threshold_devs=1.5,
    patch_size=(128, 128, 128),
    patch_overlap=(32, 32, 32),
    overlap_mode="crop",
):
    class1_model_file, class2_model_file = proposal_model
    model_fullname = os.path.join(wf.params["torch_models_fullpath"], class1_model_file)
    print(f"Using trained model {model_fullname}")
    detector_proposal_class1 = make_proposal(
        wf.vols[0],
        model_fullname,
        model_type="fpn3d",
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
    )
    detector_proposal_class1 -= np.min(detector_proposal_class1)
    detector_proposal_class1 = detector_proposal_class1 / np.max(
        detector_proposal_class1
    )
    slice_plot(detector_proposal_class1, None, None, (40, 200, 200))

    model_fullname = os.path.join(wf.params["torch_models_fullpath"], class2_model_file)
    print(f"Using trained model {model_fullname}")
    detector_proposal_class2 = make_proposal(
        wf.vols[0],
        model_fullname,
        model_type="fpn3d",
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
    )

    detector_proposal_class2 -= np.min(detector_proposal_class2)
    detector_proposal_class2 = detector_proposal_class2 / np.max(
        detector_proposal_class2
    )
    slice_plot(detector_proposal_class2, None, None, (40, 200, 200))

    if np.mean(detector_proposal_class1) > 0.5:
        detector_proposal_class1 = 1.0 - detector_proposal_class1
    if np.mean(detector_proposal_class2) > 0.5:
        detector_proposal_class2 = 1.0 - detector_proposal_class2

    detector_proposal_class1_thresh = process_proposal(
        detector_proposal_class1, threshold_devs=threshold_devs, invert=True
    )
    detector_proposal_class2_thresh = process_proposal(
        detector_proposal_class2, threshold_devs=threshold_devs, invert=True
    )
    slice_plot(
        detector_proposal_class1_thresh,
        None,
        detector_proposal_class2_thresh,
        (40, 200, 200),
        plot_color=True,
    )
    class1_proposal_fname, class1_proposal_thresh = save_combined_proposal(
        detector_proposal_class1_thresh,
        np.zeros_like(detector_proposal_class2_thresh),
        roi_name + "_1",
        wf.params["outdir"],
    )

    slice_plot(class1_proposal_thresh, None, wf.vols[0], (50, 100, 100))
    class2_proposal_fname, class2_proposal_thresh = save_combined_proposal(
        detector_proposal_class2_thresh,
        np.zeros_like(detector_proposal_class2_thresh),
        roi_name + "_2",
        wf.params["outdir"],
    )
    slice_plot(class2_proposal_thresh, None, wf.vols[0], (50, 100, 100))

    return (
        class1_proposal_fname,
        class1_proposal_thresh,
        class2_proposal_fname,
        class2_proposal_thresh,
        detector_proposal_class1,
        detector_proposal_class2,
    )


def make_and_save_proposal(
    wf,
    proposal_model,
    roi_name,
    threshold_devs=1.5,
    patch_size=(128, 128, 128),
    patch_overlap=(32, 32, 32),
    overlap_mode="crop",
    model_type="fpn3d",
):
    class1_model_file, class2_model_file = proposal_model
    model_fullname = os.path.join(wf.params["torch_models_fullpath"], class1_model_file)
    print(f"Using trained model {model_fullname}")
    proposal_class1 = make_proposal(
        wf.vols[0],
        model_fullname,
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
    )

    slice_plot(proposal_class1, None, None, (50, 200, 200))

    model_fullname = os.path.join(wf.params["torch_models_fullpath"], class2_model_file)
    print(f"Using trained model {model_fullname}")
    proposal_class2 = make_proposal(
        wf.vols[0],
        model_fullname,
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
    )

    slice_plot(proposal_class2, None, None, (40, 200, 200))

    proposal_class1_thresh = (
        process_proposal(proposal_class1, threshold_devs=threshold_devs, invert=True)
        * wf.bg_mask
    )
    proposal_class2_thresh = (
        process_proposal(proposal_class2, threshold_devs=threshold_devs, invert=True)
        * wf.bg_mask
    )
    slice_plot(
        proposal_class1_thresh,
        None,
        proposal_class2_thresh,
        (40, 200, 200),
        plot_color=True,
    )
    proposal_fname, proposal_thresh = save_combined_proposal(
        proposal_class1_thresh, proposal_class2_thresh, roi_name, wf.params["outdir"]
    )
    slice_plot(proposal_thresh, None, wf.vols[0], (50, 100, 100))

    return proposal_fname, proposal_thresh


def make_and_save_single_proposal(
    wf,
    proposal_model,
    roi_name,
    threshold_devs=1.5,
    patch_size=(128, 128, 128),
    patch_overlap=(32, 32, 32),
    overlap_mode="crop",
    model_type="fpn3d",
    invert=True,
):
    class1_model_file = proposal_model
    model_fullname = os.path.join(wf.params["torch_models_fullpath"], class1_model_file)
    print(f"Using trained model {model_fullname}")
    proposal_class1 = make_proposal(
        wf.vols[0],
        model_fullname,
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        overlap_mode=overlap_mode,
    )

    slice_plot(proposal_class1, None, None, (50, 200, 200))

    proposal_class1 -= np.min(proposal_class1)
    proposal_class1 = proposal_class1 / np.max(proposal_class1)

    proposal_class1_thresh = process_proposal(
        proposal_class1, threshold_devs=threshold_devs, invert=invert
    )

    slice_plot(
        proposal_class1_thresh,
        None,
        wf.vols[0],
        (40, 200, 200),
        plot_color=True,
    )

    proposal_fname = save_proposal(
        proposal_class1_thresh, roi_name, wf.params["outdir"]
    )

    return proposal_fname, proposal_class1_thresh, proposal_class1


def detector_proposal(
    vol,
    model_fullname,
    roi_name,
    outdir,
    model_type="unet3d",
    patch_size=(128, 128, 128),
    patch_overlap=(16, 16, 16),
    threshold_devs=1.0,
    invert=True,
):
    # using model file predict proposal segmentation

    proposal = make_proposal(
        vol,
        model_fullname,
        model_type=model_type,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    slice_plot(proposal, None, vol, (54, 100, 100))

    # proposal = proposal / np.max(proposal)
    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)

    print(
        f"Proposal min, max, mean {np.min(proposal)}, {np.max(proposal)}, {np.mean(proposal)}"
    )
    if invert:
        proposal_thresh = (proposal) > (
            np.mean(proposal) + threshold_devs * np.std(proposal)
        )
    else:
        proposal_thresh = (proposal) < (
            np.mean(proposal) - threshold_devs * np.std(proposal)
        )

    slice_plot(proposal_thresh, None, vol, (54, 100, 100))

    proposal_fname = save_proposal(proposal_thresh, roi_name, outdir)

    return proposal_fname, proposal_thresh


def process_proposal(proposal, threshold_devs=1.0, invert=False):
    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)

    print(
        f"Proposal min, max, mean {np.min(proposal)}, {np.max(proposal)}, {np.mean(proposal)}"
    )
    if invert:
        proposal_thresh = (
            proposal > (np.mean(proposal) + threshold_devs * np.std(proposal)) * 1.0
        )
    else:
        proposal_thresh = (
            proposal < (np.mean(proposal) - threshold_devs * np.std(proposal)) * 1.0
        )

    return proposal_thresh * 1.0


def save_combined_proposal(proposal_thresh1, proposal_thresh2, roi_name, outdir):
    proposal_thresh2 = proposal_thresh2 * (1.0 - proposal_thresh1)
    proposal_combined = proposal_thresh1 + (proposal_thresh2 * 2)
    proposal_fname = save_proposal(proposal_combined, roi_name, outdir)

    return proposal_fname, proposal_combined

def save_proposal(proposal_thresh, roi_name, out_dir):
    now = datetime.now()
    dt_string = now.strftime("%d%m_%H%M")
    proposal_fname = "proposal_" + roi_name + "_" + dt_string + ".h5"
    map_fullpath = os.path.join(out_dir, proposal_fname)

    with h5py.File(map_fullpath, "w") as hf:
        hf.create_dataset("map", data=proposal_thresh)
    print(f"Saved proposal to {map_fullpath}")
    return proposal_fname

def make_proposal(
    vol,
    model_fullname,
    model_type,
    nb=True,
    patch_size=(64, 64, 64),
    patch_overlap=(0, 0, 0),
    overlap_mode="crop",
    gpu_id=0,
):

    if model_type == "unet3d":
        model3d, optimizer, scheduler = prepare_unet3d(device=gpu_id)
    elif model_type == "fpn3d":
        model3d, optimizer, scheduler = prepare_fpn3d(gpu_id=gpu_id)

    if model_type == "unet3d":
        model3d = load_model(model3d, model_fullname)
        aggregator = predict_agg_3d(
            vol,
            model3d,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            device=gpu_id,
            fpn=False,
            overlap_mode=overlap_mode,
        )
        output_tensor1 = aggregator.get_output_tensor()
        print(f"Aggregated volume of {output_tensor1.shape}")
        seg_out = np.nan_to_num(output_tensor1.squeeze(0).numpy())

    elif model_type == "fpn3d":
        model3d = load_model(model3d, model_fullname)
        aggregator = predict_agg_3d(
            vol,
            model3d,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            device=gpu_id,
            fpn=False,
        )
        output_tensor1 = aggregator.get_output_tensor()
        print(f"Aggregated volume of {output_tensor1.shape}")
        seg_out = np.nan_to_num(output_tensor1.squeeze(0).numpy())

    return seg_out


def display_preds_output_fpn3d(outputs):
    [
        show_images(
            [
                outputs[idx, 0, 0, :].cpu().detach().numpy(),
            ],
            figsize=(4, 4),
        )
        for idx in range(outputs.shape[0])
    ]


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


def display_preds_dets_fpn3d(inputs, outputs, entities, loc=(32, 32, 32)):
    for idx in range(outputs.shape[0]):
        slice_plot(
            outputs[idx, 0, :].cpu().detach().numpy(),
            entities,
            outputs[idx, 0, :].cpu().detach().numpy(),
            loc,
            figsize=(2, 2),
        )


def display_preds_fpn(outputs):
    [
        show_images(
            [
                outputs[idx, 0, :].cpu().detach().numpy(),
                # smax[idx,0,:].cpu().detach().numpy(),
                # label_batch[idx,0,:].cpu().float(),
                #  input_imgs[idx]]
            ],
            figsize=(4, 4),
        )
        for idx in range(8)
    ]




def postprocess_class1_proposal(
    wf, class1_proposal, class_all_proposal, class2_proposal, threshold_devs=1.5
):
    class1_proposal_thresh = process_proposal(
        class1_proposal, threshold_devs=threshold_devs, invert=True
    )
    slice_plot(class1_proposal_thresh, None, wf.vols[0], (50, 200, 200))

    combined = ((class1_proposal_thresh + (class_all_proposal > 0) * 1.0) == 2) * 1.0
    slice_plot(combined, None, wf.vols[0], (50, 200, 200))

    class1_f = filter_proposal_mask(
        combined, num_erosions=4, num_dilations=0, num_medians=0
    )
    slice_plot(class1_f, None, wf.vols[0], (50, 200, 200), suptitle="class1_f")

    class1_f_s, tables, li = filter_small_components(
        [class1_f], min_component_size=1000
    )
    slice_plot(class1_f_s, None, wf.vols[0], (50, 200, 200), suptitle="class1_f_s")

    class1_f_s_d = filter_proposal_mask(
        class1_f_s, num_erosions=0, num_dilations=8, num_medians=0
    )
    slice_plot(class1_f_s_d, None, wf.vols[0], (50, 200, 200), suptitle="class1_f_s_d")

    return class1_f, class1_f_s, class1_f_s_d


def postprocess_class2_proposal(
    wf, class1_proposal, class_all_proposal, class2_proposal, class1_f_s_d, threshold_devs=1.5
):
    class_all_proposal_thresh = process_proposal(
        class_all_proposal, threshold_devs=threshold_devs, invert=True
    )
    slice_plot(class_all_proposal_thresh, None, wf.vols[0], (50, 200, 200))
    class_2_proposal_thresh = process_proposal(
        class2_proposal, threshold_devs=threshold_devs, invert=True
    )
    slice_plot(
        class_2_proposal_thresh,
        None,
        wf.vols[0],
        (50, 200, 200),
        suptitle="class2_proposal_thresh",
    )

    combined_2 = ((class_2_proposal_thresh + class_all_proposal_thresh) == 2) * 1.0

    # combined_2 = ((combined_2 + density_thresh > 0)) * 1.0
    slice_plot(combined_2, None, wf.vols[0], (50, 200, 200), suptitle="combined_2")

    combined_2_s, tables, li = filter_small_components(
        [combined_2], min_component_size=500
    )

    # class2_f = filter_small_components([class2_proposal_thresh * wf.bg_mask], component_size=1000)
    class2_f = filter_proposal_mask(
        class_2_proposal_thresh - class1_f_s_d,
        num_erosions=1,
        num_dilations=0,
        num_medians=0,
    )
    class2_f_masked = class_2_proposal_thresh * (1.0 - class1_f_s_d)
    slice_plot(
        class2_f_masked, None, wf.vols[0], (50, 200, 200), suptitle="class2_f_masked"
    )

    return combined_2_s, class2_f_masked


