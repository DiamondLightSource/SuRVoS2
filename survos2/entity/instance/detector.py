import os
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import tqdm
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
from loguru import logger
from pathlib import Path

from survos2.entity.models.head_cnn import prepare_fpn3d
from survos2.entity.entities import make_entity_bvol, make_entity_df, offset_points
from survos2.entity.components import measure_components
from survos2.entity.sampler import centroid_to_bvol, viz_bvols

from survos2.server.filtering import (
    gaussian_blur_kornia,
)
from survos2.server.filtering.morph import dilate, erode, median
from survos2.entity.pipeline_ops import make_features
from survos2.server.state import cfg
from survos2.frontend.nb_utils import slice_plot

from survos2.entity.entities import make_entity_df
from survos2.entity.sampler import offset_points
from survos2.entity.components import filter_proposal_mask, filter_small_components
from survos2.entity.patches import pad_vol
from survos2.frontend.nb_utils import summary_stats
from survos2.entity.sampler import offset_points

from survos2.entity.instance.detector_eval import analyze_detector_result
from survos2.entity.instance.dataset import FilteredVolumeDataset
from survos2.entity.instance.detector_eval import analyze_detector_result

from survos2.entity.patches import pad_vol
from survos2.api import workspace as ws

class ObjectVsBgClassifier:    
    def __init__(self, wf, augmented_entities, class1_proposal_thresh, 
                 class1_proposal,bg_mask_all, bvol_dim=(32,32,32), padding=(32,32,32), area_min=0, area_max=1e14, n_components=4, plot_debug=False):
        self.wf = wf
        self.class1_proposal_thresh = class1_proposal_thresh 
        self.class1_proposal = class1_proposal 
        self.bg_mask_all = bg_mask_all 
        self.area_min = area_min
        self.area_max = area_max
        self.padding = padding
        self.bvol_dim = bvol_dim
        #print(f"Offset points before sampling: {self.bvol_dim}")
        entities = offset_points(np.array(augmented_entities), np.array(self.bvol_dim) // 2)
        self.augmented_entities = np.array(make_entity_df(entities, flipxy=False))
        #self.augmented_entities = augmented_entities
        self.n_components = n_components
        self.plot_debug = plot_debug
    def reset_bg_mask(self):
        self.bg_mask_all = np.zeros_like(self.wf.vols[0])
    def make_component_tables(self):
        print("\nmake_component_tables\n")
        class1_component_table, class1_padded_proposal = make_component_table(self.wf,
                                                                              self.class1_proposal_thresh, 
                                                                              #self.class1_proposal_thresh * (1.0 - self.bg_mask_all),
                                                                              area_min=self.area_min, 
                                                                              area_max=self.area_max)
        print(summary_stats(class1_component_table["area"]))
        print("\n")
        filtered_component_table = filter_component_table(class1_component_table, area_min=self.area_min, area_max=self.area_max)
        self.class1_proposal_entities = np.array(make_entity_df(np.array(filtered_component_table[["z", "x", "y", "class_code"]]), flipxy=True))
        
        entities = make_entity_df(np.array(self.augmented_entities), flipxy=False)
        self.class1_gold_entities = np.array(entities)
        print(f"Number of class1 gold entities {len(self.class1_gold_entities)}")
        self.class1_dets, class1_mask_gt, class1_mask_dets = analyze_detector_result(self.wf, 
                                           self.class1_gold_entities, 
                                           self.class1_proposal_entities,
                                           padding=self.padding, 
                                           mask_bg=True, 
                                           plot_location=(50,100,200),
                                           bvol_dim=self.bvol_dim,
                                           bg_vol=self.class1_proposal_thresh)
        
    def prepare_detector_data(self):
        print("\nprepare_classical_detector_data\n")
        from survos2.entity.models.head_classical import prepare_classical_detector_data2
        self.class1_train_entities, self.class1_val_entities, self.fvol = prepare_classical_detector_data2(self.wf.vols[1],
                                                                                   self.augmented_entities, 
                                                                                   self.class1_proposal_thresh,
                                                                                   self.padding, 
                                                                                   additional_feature_vols=[self.class1_proposal],                                  
                                                                                   area_min=0,
                                                                                   area_max=1e14, 
                                                                                   flip_xy=False)



    def train_validate_model(self, model_file=None):
        print("\nTrain_validate_model\n")
        from survos2.entity.models.head_classical import trainvalidate_classical_head
        self.trained_classifiers, filtered_patch_dataset_train, filtered_patch_dataset_val, self.feats = trainvalidate_classical_head(self.wf, 
                                                                       self.fvol, 
                                                                       self.class1_train_entities, 
                                                                       self.class1_val_entities, 
                                                                       model_file=model_file,
                                                                       n_components=self.n_components, 
                                                                       bvol_dim=self.bvol_dim, 
                                                                       flip_xy=False, 
                                                                       plot_all=True)
    def predict(self, locs, score_thresh=0.95, model_file=None, offset=False):
        print("\npredict\n")
        from survos2.entity.models.head_classical import classical_prediction
        detected, preds, proba, fvd = classical_prediction(self.wf, 
                                                         self.fvol, 
                                                         self.trained_classifiers["etc"]["classifier"], 
                                                         locs, 
                                                         model_file=model_file,
                                                         score_thresh=score_thresh,
                                                         n_components=self.n_components,
                                                         bvol_dim=self.bvol_dim,
                                                         plot_all=True, 
                                                         flip_xy=False,
                                                         offset=offset)
        self.detections = locs[detected]
    
    def analyze_result(self, gold, raw_dets, classified_dets):
        print("\nanalyze_result\n")
        class1_detected_entities, class1_mask_gt, class1_mask_dets  = analyze_detector_result(self.wf, 
                                               gold, 
                                               raw_dets, 
                                               padding=(0,0,0), 
                                               mask_bg=False, 
                                               plot_location=(self.wf.vols[0].shape[0]//2,200,200),
                                               bvol_dim=self.bvol_dim,
                                               bg_vol=self.class1_proposal_thresh)
        class1_detected_entities, class1_mask_gt, class1_mask_dets  = analyze_detector_result(self.wf, 
                                               gold, 
                                               classified_dets, 
                                               padding=(0,0,0), 
                                               mask_bg=False, 
                                               plot_location=(self.wf.vols[0].shape[0]//2,200,200),
                                               bvol_dim=self.bvol_dim,
                                               bg_vol=self.class1_proposal_thresh)


class ObjectVsBgCNNClassifier(ObjectVsBgClassifier):
    def __init__(self, wf, 
                 augmented_entities, 
                 class1_proposal_thresh, 
                 class1_proposal,
                 bg_mask_all, 
                 bvol_dim=(32,32,32), 
                 padding=(32,32,32), 
                 area_min=0, 
                 area_max=1e14, 
                 n_components=4, 
                 plot_debug=False):
        super().__init__(wf, 
                        augmented_entities, 
                        class1_proposal_thresh, 
                        class1_proposal,bg_mask_all, 
                        bvol_dim=(32,32,32), 
                        padding=(32,32,32), 
                        area_min=0, 
                        area_max=1e14, 
                        n_components=4, 
                        plot_debug=False)
    def prepare_detector_data(self):
        print("\nprepare_cnn_detector_data\n")
        self.dataloaders, gt_train_entities, gt_val_entities = prepare_patch_dataloaders_and_entities(self.wf.vols[0], 
                                                                                         self.augmented_entities, 
                                                                                         flip_xy=True,
                                                                                         train_proportion=0.8,
                                                                                         padding=2 * np.array(self.padding))
        if self.plot_debug:
            patches = []
            for patch in self.dataloaders["train"]:
                plt.figure()
                plt.title(patch[1]["labels"])
                plt.imshow(patch[0][0][0][self.padding[0]//2,:])
                patches.append(patch[0][0][0].numpy())

    def train_validate_model(self, model_file, workspace):
        print("\nTrain_validate_model\n")
        from survos2.entity.models.head_cnn import make_classifications, setup_training
        from survos2.entity.trainer import train_detector_head

        self.detmod, optimizer = setup_training(model_file)

        ws_object = ws.get(workspace)
        data_out_path = Path(ws_object.path, "fcn")
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        model_fn = f"{dt_string}_trained_fcn_model"
        model_out = str(Path(data_out_path, model_fn).resolve())
        
        logger.info(f"Saving fcn model to: {model_out}")

        epochs, losses, accuracies = train_detector_head(self.detmod, 
                                                 optimizer, 
                                                 self.dataloaders,
                                                 model_out, 
                                                 num_epochs=6)

        plt.figure()
        plt.plot(epochs,losses)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Loss')


        plt.figure()
        plt.plot(range(len(accuracies["train"])), accuracies["train"])
        plt.ylabel('Accuracy')
        plt.xlabel('Iters')
        plt.title('Train accuracy')

        plt.figure()
        plt.plot(range(len(accuracies["val"])), accuracies["val"])
        plt.ylabel('Accuracy')
        plt.xlabel('Iters')
        plt.title('Validation accuracy')

    def predict(self, locs, score_thresh=0.95, model_file=None, offset=False):
        from survos2.entity.instance.detector import process_predictions
        from survos2.entity.models.head_cnn import make_classifications

        print("\npredict cnn\n")
        padded_vol = pad_vol(self.wf.vols[0], np.array(self.padding))
        all_patches, _ = prepare_filtered_patch_dataset(locs,[padded_vol], flip_xy=False, bvol_dim=(32,32,32))
        dataloader = {
                "all": DataLoader(
                    all_patches, batch_size=1, shuffle=False, num_workers=0
                ),
            }

        plot_all = False
        if plot_all:
            for patch in dataloader["all"]:
                print(patch[0][0][0].shape)
                plt.figure()
                plt.imshow(patch[0][0][0][32,:])
        device=0
        predictions, acc, logits = make_classifications(self.detmod, dataloader["all"], device)
        print(f"Made predictions of shape: {predictions.shape}")

        predicted_target_entities, predicted_bg_entities = process_predictions(locs, 
                                                                            predictions, 
                                                                            logits, 
                                                                            dataloader, 
                                                                            score_thresh=0.015)

        #self.detections = locs[detected]
        self.detections = predicted_target_entities
 


class InstanceClassifierThreeClass:    
    def __init__(self, wf, augmented_entities, class1_proposal_thresh, 
                 class2_proposal_thresh, 
                 class_all_proposal_thresh, 
                 class1_proposal, class2_proposal, bg_mask_all, padding=(64,64,64)):

        self.wf = wf
        self.class1_proposal_thresh = class1_proposal_thresh
        self.class2_proposal_thresh = class2_proposal_thresh
        self.class_all_proposal_thresh = class_all_proposal_thresh
        self.class1_proposal = class1_proposal
        self.class2_proposal = class2_proposal
        self.bg_mask_all = bg_mask_all
        self.area_min = 5000
        self.area_max = 90000000
        self.augmented_entities = augmented_entities
        self.padding = padding
    def reset_bg_mask(self):
        self.bg_mask_all = np.zeros_like(self.wf.vols[0])
        
    def make_component_tables(self):
        class1_component_table, class1_padded_proposal = make_component_table(self.wf, self.class1_proposal_thresh * (1.0 - self.bg_mask_all),
                                                                             area_min=self.area_min, area_max=self.area_max)
        print(summary_stats(class1_component_table["area"]))
        print("\n")
        filtered_component_table = filter_component_table(class1_component_table, area_min=self.area_min, area_max=self.area_max)
        self.class1_proposal_entities = np.array(filtered_component_table[["z", "x", "y", "class_code"]])
        
        gt_entities2 = make_entity_df(self.wf.gold, flipxy=True)
        self.class1_gold_entities = np.array(gt_entities2[gt_entities2["class_code"] !=5])
        print(f"Number of class1 gold entities {len(self.class1_gold_entities)}")
        self.class1_dets, class1_mask_gt, class1_mask_dets = analyze_detector_result(self.wf, 
                                           self.class1_gold_entities, 
                                           self.class1_proposal_entities,
                                           self.padding, 
                                           mask_bg=False, 
                                           plot_location=(50,100,200))
        
        class2_component_table, class2_padded_proposal = make_component_table(self.wf, 
                                                                          (self.class2_proposal_thresh * (1.0 - self.bg_mask_all)), 
                                                                          area_min=self.area_min, 
                                                                          area_max=self.area_max, 
                                                                          padding=(0,0,0))

        
        self.class2_gold_entities = np.array(gt_entities2[gt_entities2["class_code"] == 5])
        print(f"Number of class2 gold entities: {len(self.class2_gold_entities)}")
        
        filtered_class2_component_table = filter_component_table(class2_component_table, 
                                                                 area_min=self.area_min, 
                                                                 area_max=self.area_max)
        self.class2_proposal_entities = np.array(filtered_class2_component_table[["z", "x", "y", "class_code"]])

        self.class2_dets, class2_mask_gt, class2_mask_dets = analyze_detector_result(self.wf, 
                                           self.class2_gold_entities, 
                                           self.class2_proposal_entities, 
                                           bvol_dim=self.padding,
                                           mask_bg=False, 
                                           plot_location=(50,100,200))
        
#         class_both_component_table, class_both_proposal = make_component_table(wf, 
#                                                                           (((class1_f + combined_2) > 0) * 1.0) * (1.0 - bg_mask_all), 
#                                                                           area_min=5000, 
#                                                                           area_max=4000000, 
#                                                                           padding=(0,0,0))
#         gt_entities2 = make_entity_df(wf.gold, flipxy=False)
#         class_all_gold_entities = np.array(gt_entities2[gt_entities2["class_code"] != 2])

#         #class_all_gold_entities = np.array(gt_entities2[gt_entities2["class_code"] ==5])
#         print(len(class_all_gold_entities))

#         filtered_class_both_component_table = filter_component_table(class_both_component_table, area_min=0)

#         class_both_proposal_entities = np.array(filtered_class_both_component_table[["z", "x", "y", "class_code"]])

#         class_both_dets, class_both_mask_gt, class_all_mask_dets = analyze_detector_result(wf, 
#                                            class_all_gold_entities, 
#                                            class_both_proposal_entities, 
#                                            bvol_dim=(42,32,32),
#                                            mask_bg=False, 
#                                            plot_location=(50,100,200))
        
    def prepare_detector_data(self):
        from survos2.entity.models.head_classical import prepare_classical_detector_data2
        self.class1_train_entities, self.class1_val_entities, self.fvol = prepare_classical_detector_data2(self.wf.vols[1],
                                                                                   self.augmented_entities, 
                                                                                   self.class1_proposal_thresh,
                                                                                   (64,64,64), 
                                                                                   additional_feature_vols=[self.class1_proposal, 
                                                                                                            self.class2_proposal],                                  
                                                                                   area_min=0,
                                                                                   area_max=30000000, 
                                                                                   flip_xy=True)
        
    def view_patch_dataset(self):
        stage_train=True
        if stage_train:
            fvd, targs_all = prepare_filtered_patch_dataset(
                self.class1_train_entities, self.fvol.filtered_layers, bvol_dim=(32,32,32), flip_xy=True
            )
        else:
            fvd = prepare_filtered_patch_dataset(
                np.array(self.dets), self.fvol, bvol_dim=self.padding, flip_xy=False
            )
        for i in range(10):
            plt.figure()
            print(fvd[i][0][0].shape)
            plt.title(fvd[i][1])
            plt.imshow(fvd[i][0][2][32,:])

    def train_validate_model(self):
        self.n_components = 24
        from survos2.entity.models.head_classical import trainvalidate_classical_head
        self.trained_classifiers, filtered_patch_dataset_train, filtered_patch_dataset_val, self.feats = trainvalidate_classical_head(self.wf, 
                                                                       self.fvol, 
                                                                       self.class1_train_entities, 
                                                                       self.class1_val_entities, 
                                                                       model_file=None,
                                                                       n_components=self.n_components, 
                                                                       bvol_dim=(32,32,32), 
                                                                       flip_xy=True, 
                                                                       plot_all=True)
    def predict(self, locs, score_thresh=0.55):
        from survos2.entity.models.head_classical import classical_prediction
        detected, preds, proba, fvd = classical_prediction(self.wf, 
                                                         self.fvol, 
                                                         self.trained_classifiers["mlp"]["classifier"], 
                                                         locs, 
                                                         model_file=None,
                                                         score_thresh=score_thresh,
                                                         n_components=self.n_components,
                                                         bvol_dim=(32,32,32),
                                                         plot_all=True, 
                                                         flip_xy=True)
        self.detections = locs[detected]
    def analyze_result(self, gold, raw_dets, classified_dets):
        #class1_dets = offset_class1_entities[offset_class1_entities[:,3] == 1]
        #gt_entities2 = make_entity_df(wf.gold, flipxy=True)
        #class1_gold_entities = np.array(gt_entities2[gt_entities2["class_code"] != 5])
        #print(len(classical_class1_dets),len(class1_gold_entities))
        #print(class1_dets.shape, classical_class1_dets.shape)
        #class1_gold_entities = gt_entities[(gt_entities[:,3] == 0) | (gt_entities[:,3] == 1) | (gt_entities[:,3] == 2) | (gt_entities[:,3] == 5) ]
        #class1_gold_entities = gt_entities[(gt_entities[:,3] == 0) ]

        class1_detected_entities, class1_mask_gt, class1_mask_dets  = analyze_detector_result(self.wf, 
                                               gold, 
                                               raw_dets, 
                                               padding=(0,0,0), 
                                               mask_bg=False, 
                                               plot_location=(50,200,200))

        #class1_gold_entities = gt_entities[(gt_entities[:,3] == 0) | (gt_entities[:,3] == 1) | (gt_entities[:,3] == 2) | (gt_entities[:,3] == 5) ]
        #class1_gold_entities = gt_entities[(gt_entities[:,3] == 0) ]
        class1_detected_entities, class1_mask_gt, class1_mask_dets  = analyze_detector_result(self.wf, 
                                               gold, 
                                               classified_dets, 
                                               padding=(0,0,0), 
                                               mask_bg=False, 
                                               plot_location=(50,200,200))
    

def prepare_patch_dataloaders_and_entities(
    main_vol,
    entities,
    padding=(64, 64, 64),
    gt_proportion=1.0,
    train_proportion=0.70,
    flip_xy=False,
    offset_type="full"
):

    gt_train_entities, gt_val_entities = entity_traintest_split(
        entities, gt_proportion, train_proportion=train_proportion
    )
    patch_train, patch_val = prepare_patch_dataset(
        main_vol, gt_train_entities, gt_val_entities, padding, flip_xy=flip_xy, offset_type=offset_type
    )
    dataloaders = prepare_patch_dataloaders(patch_train, patch_val, plot_all=False)

    return dataloaders, gt_train_entities, gt_val_entities


def prepare_detector_training(
    wf,
    proposal_fname,
    proposal_dir,
    padding=(32, 32, 32),
    gt_proportion=1.0,
    flip_xy=True,
):

    gt_train_entities, gt_val_entities = prepare_gt(
        wf, gt_proportion, train_proportion=0.7
    )

    map_fullpath = os.path.join(proposal_dir, proposal_fname)
    bbs_tables, selected_entities, padded_proposal = region_proposal(
        map_fullpath, padding
    )

    patch_train, patch_val = prepare_patch_dataset(
        wf, gt_train_entities, gt_val_entities, padding, flip_xy=flip_xy
    )
    dataloaders = prepare_patch_dataloaders(patch_train, patch_val, plot_all=False)
    return dataloaders, gt_train_entities, gt_val_entities, padded_proposal


def prepare_patch_dataset(
    main_vol, gt_train_entities, gt_val_entities, padding, flip_xy=False, offset_type="full"
):
    padded_vol = pad_vol(main_vol, np.array(padding) // 2)
    print(f"Padding image volume with padding {np.array(padding) //2}")
    
    fvd_train, _ = prepare_filtered_patch_dataset(
        gt_train_entities,
        [padded_vol],
        flip_xy=flip_xy,
        bvol_dim=np.array(padding) // 2,
        offset_type=offset_type
    )
    fvd_val, _ = prepare_filtered_patch_dataset(
        gt_val_entities, [padded_vol], flip_xy=flip_xy, bvol_dim=np.array(padding) // 2
    )
    return fvd_train, fvd_val


def prepare_patch_dataloaders(fvd_train, fvd_val, batch_size=1, plot_all=False):
    dataloaders = {
        "train": DataLoader(
            fvd_train, batch_size=batch_size, shuffle=True, num_workers=0
        ),
        "val": DataLoader(fvd_val, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    print(f"Prepared dataloaders using volumes of shape {fvd_train[0][0][0].shape}")
    if plot_all:
        for img, label in dataloaders["val"]:
            # print(img[0].shape)
            plt.figure()
            plt.imshow(img[0][0, 32, :])
            plt.title(label["labels"][0].numpy())
            # print(label['labels'][0].numpy())

    return dataloaders


def prepare_filtered_patch_dataset(
    entities, filtered_layers, bvol_dim=(32, 32, 32), flip_xy=False, offset_type="full"
):
    entities = np.array(make_entity_df(entities, flipxy=True))
    
    if offset_type=="full":
        print(f"Offset points before sampling: {bvol_dim}")
        entities = offset_points(entities, np.array(bvol_dim) // 2)
    elif offset_type=="half":
        print(f"Offset points before sampling: {bvol_dim}")
        entities = offset_points(entities, np.array(bvol_dim))
    
    target_bvol = centroid_to_bvol(entities, bvol_dim=bvol_dim, flipxy=False)
    print(f"Produced {len(target_bvol)} target bounding volumes. ")
    labels = entities[:, 3]
    print(f"Making FilteredVolumeDataset with bounding volume dimensions of {bvol_dim}")
    fvd = FilteredVolumeDataset(
        filtered_layers, target_bvol, labels, patch_size=np.array(bvol_dim)
    )

    return fvd, target_bvol


def entity_traintest_split(ents, gt_proportion=0.5, train_proportion=0.8):
    ents_sel = np.random.choice(range(len(ents)), int(gt_proportion * len(ents)))
    idx = int(train_proportion * len(ents_sel))
    gt_entities = ents[ents_sel]
    gt_train_entities = gt_entities[:idx]
    gt_val_entities = gt_entities[idx:]
    print(
        f"Produced {len(gt_entities)} entities, split into train set of len {len(gt_train_entities)} and val set of len {len(gt_val_entities)}"
    )
    return gt_train_entities, gt_val_entities


def prepare_gt(wf, gt_proportion=0.5, train_proportion=0.8):
    wf_sel = np.random.choice(range(len(wf.locs)), int(gt_proportion * len(wf.locs)))
    idx = int(train_proportion * len(wf_sel))
    gt_entities = wf.locs[wf_sel]
    gt_train_entities = gt_entities[:idx]
    gt_val_entities = gt_entities[idx:]
    print(
        f"Produced {len(gt_entities)} entities, split into train set of len {len(gt_train_entities)} and val set of len {len(gt_val_entities)}"
    )
    return gt_train_entities, gt_val_entities


def prepare_proposal(proposal):
    proposal -= np.min(proposal)
    proposal = proposal / np.max(proposal)
    proposal = ((proposal) > (np.mean(proposal)) + 1.0 * np.std(proposal)) * 1.0
    cfg["feature_params"] = {}
    from survos2.entity.pipeline import Patch

    p = Patch({"Main": proposal}, {}, {}, {})
    cfg["feature_params"]["out2"] = [[erode, {"thresh": 0.5, "num_iter": 3}]]
    p = make_features(p, cfg)
    return p.image_layers["out2"]

def prepare_proposal2(masked_proposal):
    cfg["feature_params"] = {"out2": [[gaussian_blur_kornia, {"sigma": (15, 15, 15)}]]}

    # cfg["feature_params"]["out2"] = [[erode, {'thresh' : 0.05, 'num_iter': 1} ]]
    # cfg["feature_params"]["out2"] = [[dilate, {'thresh' : 0.2, 'num_iter': 2} ]]
    # cfg["feature_params"]["out2"] = [[erode, {'thresh' : 0.2, 'num_iter': 1} ]]
    p = Patch({"Main": masked_proposal}, {}, {}, {})
    p = make_features(p, cfg)
    return p.image_layers["out2"]

def load_filtered_tables(fullname):
    filtered_tab = pd.read_csv(fullname)
    filtered_tab.drop(
        filtered_tab.columns[filtered_tab.columns.str.contains("unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    return filtered_tab

def pad_and_detect_blobs(masked_proposal, padding):
    print(masked_proposal.shape)
    bbs_tables, selected_entities = detect_blobs(padded_proposal)
    return bbs_tables, selected_entities


def make_anno_masks(wf, gt_entities, mask_brush_size=(15, 15, 15), flipxy=True):
    # mask_brush_size = wf.params["entity_meta"]["0"]["size"]
    target_cents = np.array(gt_entities)[:, 0:4]
    target_cents = target_cents[:, [0, 2, 1, 3]]
    targs_all = centroid_to_bvol(target_cents, bvol_dim=mask_brush_size, flipxy=flipxy)
    mask_all = viz_bvols(wf.vols[0], targs_all)

    # target_cents = np.array(wf.locs[::4])[:, 0:4]
    # target_cents = target_cents[:, [0, 2, 1, 3]]
    # targs_all = centroid_to_bvol(target_cents, bvol_dim=mask_brush_size, flipxy=flipxy)
    # gt_locs_mask = viz_bvols(wf.vols[0], targs_all)

    return mask_all  # , gt_locs_mask


def component_bounding_boxes(images):
    bbs_tables = []
    bbs_arrs = []

    for image in images:
        bbs_arr = measure_components(image)
        bbs_arrs.append(bbs_arr)
        bbs_table = make_entity_bvol(bbs_arr)
        bbs_tables.append(bbs_table)

    return bbs_tables, bbs_arrs


def detect_blobs(
    padded_proposal,
    area_min=50,
    area_max=50000,
    plot_all=False,
):
    images = [padded_proposal]
    bbs_tables, bbs_arrs = component_bounding_boxes(images)
    print(f"Detecting blobs on image of shape {padded_proposal.shape}")
    zidx = padded_proposal.shape[0] // 2

    from survos2.frontend.nb_utils import summary_stats

    print("Component stats: ")
    print(f"{summary_stats(bbs_tables[0]['area'])}")

    if plot_all:
        for idx in range(len(bbs_tables)):
            print(idx)
            plt.figure(figsize=(5, 5))
            plt.imshow(images[idx][zidx, :], cmap="gray")
            plt.scatter(bbs_arrs[idx][:, 4], bbs_arrs[idx][:, 3])

    selected_entities = bbs_tables[0][
        (bbs_tables[0]["area"] > area_min) & (bbs_tables[0]["area"] < area_max)
    ]
    print(f"Number of selected entities {len(selected_entities)}")

    return bbs_tables, selected_entities


def filter_component_table(component_table, area_min=0, area_max=1e9):
    print(component_table)
    print(len(component_table))
    from survos2.frontend.nb_utils import summary_stats

    print(summary_stats(component_table["area"]))
    plt.hist(np.array(component_table["area"]))
    component_table = component_table[component_table["area"] > area_min]
    component_table = component_table[component_table["area"] < area_max]
    return component_table


def region_proposal(
    map_fullpath, padding, padded_mask=None, area_max=10, area_min=100000
):
    proposal_filt = load_and_prepare_proposal(map_fullpath)

    padded_proposal = pad_vol(proposal_filt, padding)
    if padded_mask is not None:
        proposal_filt = padded_proposal * ((padded_mask > 0) * 1.0)
    bbs_tables, selected_entities = detect_blobs(
        proposal_filt, area_min=area_min, area_max=area_max
    )
    return bbs_tables, selected_entities, padded_proposal


def make_component_table(
    wf,
    proposal,
    padding=(32, 32, 32),
    area_min=50,
    area_max=100000,
):
    padded_proposal = proposal  # pad_vol(proposal, padding)

    bbs_tables, component_table = detect_blobs(
        padded_proposal, area_min=area_min, area_max=area_max
    )
    proposal_entities = np.array(component_table[["z", "x", "y", "class_code"]])

    slice_plot(
        (padded_proposal),
        proposal_entities,
        None,
        (
            padded_proposal.shape[0] // 2,
            padded_proposal.shape[1] // 2,
            padded_proposal.shape[2] // 2,
        ),
    )
    return component_table, padded_proposal


def prepare_component_table(
    wf,
    proposal_fullpath,
    bg_mask=None,
    padding=(32, 32, 32),
    area_min=50,
    area_max=100000,
):
    padded_vol = pad_vol(wf.vols[1], np.array(padding) // 2)

    if bg_mask is None:
        bg_mask = np.ones_like(wf.vols[1])

    padded_mask = pad_vol(bg_mask, np.array(padding) // 2)

    bbs_tables, component_table, padded_proposal = region_proposal(
        proposal_fullpath,
        np.array(padding) // 2,
        padded_mask,
        area_max=area_max,
        area_min=area_min,
    )

    proposal_entities = np.array(component_table[["z", "x", "y", "class_code"]])
    slice_plot(
        (padded_proposal) * padded_mask,
        proposal_entities,
        padded_vol,
        (
            padded_vol.shape[0] // 2,
            padded_vol.shape[1] // 2,
            padded_vol.shape[2] // 2,
        ),
    )
    return component_table, padded_proposal


def process_predictions(selected_entities, predictions, logits, dataloader, score_thresh=0):
    prediction_entities = selected_entities.copy()
    scores = np.abs([l[0][1] / l[0][0] for l in logits])
    plt.hist(np.abs(scores) / np.max(np.abs(scores)))
    print(f"Number predicted as background class: {np.sum(predictions == 1)}")
    scores_scaled = np.abs(scores) / np.max(np.abs(scores))

    print(np.mean(scores_scaled))
    confident = scores_scaled > score_thresh  # < np.mean(scores_scaled)
    prediction_entities[:, 3] = predictions
    predictions_s = predictions[confident]
    prediction_entities = prediction_entities[confident]
    predicted_target_entities = prediction_entities[predictions_s == 0]
    predicted_bg_entities = prediction_entities[predictions_s == 1]
    print(f"Produced {prediction_entities.shape} predictions")
    print(
        f"Targets: {predicted_target_entities.shape}, Bg: {predicted_bg_entities.shape}"
    )

    return predicted_target_entities, predicted_bg_entities


def process_predictions2(wf, selected_entities, predictions, logits, dataloader):
    prediction_entities = selected_entities.copy()
    scores = np.abs([l[0][1] / l[0][0] for l in logits])
    plt.hist(np.abs(scores) / np.max(np.abs(scores)))
    print(f"Number predicted as background class: {np.sum(predictions == 1)}")
    scores_scaled = np.abs(scores) / np.max(np.abs(scores))

    plot_all = False
    if plot_all:
        for i, b in enumerate(dataloader["all"]):

            print(b[0][0][0].shape)
            plt.figure()
            plt.imshow(b[0][0][0][32, :])
            plt.title(str(predictions[i]) + " " + str(scores[i]))

    confident = scores_scaled > 0  # < np.mean(scores_scaled)
    prediction_entities[:, 3] = predictions
    predictions_s = predictions[confident]
    prediction_entities = prediction_entities[confident]
    predicted_target_entities = prediction_entities[predictions_s == 0]
    predicted_bg_entities = prediction_entities[predictions_s == 1]
    print(f"Produced {prediction_entities.shape} predictions")
    print(
        f"Targets: {predicted_target_entities.shape}, Bg: {predicted_bg_entities.shape}"
    )
    slice_plot(
        wf.vols[0], prediction_entities, None, (40, 300, 300), unique_color_plot=True
    )

    return predicted_target_entities, predicted_bg_entities


def analyze_predictions(wf, gold_entities, proposal_entities, predicted_entities):
    print(f"Analyzing {len(gold_entities)} predictions")

    fg_dets, fg_mask_gt, fg_mask_dets = analyze_detector_result(
        wf,
        gold_entities,
        proposal_entities,
        (32, 32, 32),
        mask_bg=False,
        plot_location=(50, 100, 200),
    )
    print("-" * 80)
    fg_dets, fg_mask_gt, fg_mask_dets = analyze_detector_result(
        wf,
        gold_entities,
        predicted_entities,
        (32, 32, 32),
        mask_bg=False,
        plot_location=(50, 100, 200),
    )

