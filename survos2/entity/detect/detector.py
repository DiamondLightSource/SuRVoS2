import numpy as np
import pandas as pd 
from skimage import data, measure
import skimage
from matplotlib import pyplot as plt
import torch
import numpy as np
from UtilityCluster import show_images

from typing import Collection
from matplotlib import patches, patheffects
from matplotlib.patches import Rectangle, Patch

#from survos2.entity.Various import draw_rect

from roi_pooling.functions.roi_pooling import roi_pooling_2d

from torchvision.ops import roi_align


from scipy import ndimage



def roi_pool_vol(cropped_vol, filtered_tables):
    roi_aligned = []
    print(cropped_vol.shape)
    cols = ['bb_s_x', 'bb_s_y', 'bb_f_x', 'bb_f_y', 'bb_s_z', 'bb_f_z']
    bb2d_sf = np.array(filtered_tables[0][cols])

    cropped_t = np.moveaxis(cropped_vol, 0, 1)
    cropped_t = torch.FloatTensor(cropped_t)
    
    for jj, z_l in enumerate(range(10,cropped_vol.shape[1],10)):
        
        z_u = z_l + 10
        print(z_l, z_u)
        good_bb = []    
        
        for bb in bb2d_sf:
            x_s,y_s,x_f,y_f, z_s,z_f = bb
            
            #print(x_s,y_s,x_f,y_f, z_s,z_f)
            if (z_l >= z_s) and (z_f >= z_u):
                #print(f"bb ok {bb} {z_l} {z_u}")
                good_bb.append(bb)
            else:
                pass
                #print(f"rejected {bb}, {z_l}, {z_u}")

        bb2d_centdim = [(x_s,y_s,x_f-x_s,y_f-y_s) for (x_s,y_s,x_f,y_f, z_s,z_f) 
                        in bb2d_sf if (z_l >= z_s) and (z_f >=  z_u) ]

        preds = ['a' for i in range(bb2d_sf.shape[0])]
        scores = [1.0 for i in range(bb2d_sf.shape[0])]

        pred_info = {'bbs' : bb2d_centdim,
                     'preds' : preds,
                     'scores' : scores}

        plot_bb_2d(cropped_vol[0,(z_l+z_u)//2,:], pred_info)
        padx, pady = 10,10
        
        expanded_bbox = [(0,bb[1]-padx,bb[0]-pady,
                            bb[3]+padx,bb[2]+pady) 
                         for i,bb in enumerate(good_bb)]
        
        #rpooled = roi_pool_2d(cropped_vol[(z_l+z_u)//2,:], expanded_bbox, output_size = (128,128))
        #rpooled = rpooled.detach().cpu().numpy()
        
        print(expanded_bbox)
        
        rois = torch.FloatTensor(expanded_bbox)
        aligned = roi_align(cropped_t, rois, output_size=(64,64))
        
        #from UtilityCluster import summary_stats, show_images
        #show_images([im.reshape((output_size[0],output_size[1])) for im in rpooled[0:len(rpooled)]])
        
        roi_aligned.append(aligned.detach().cpu().numpy())

    return roi_aligned

def calc_precision(iou_thresh):
    matches = match_bboxes(targs, preds, IOU_THRESH=iou_thresh)
    matches[0].sort()
    targs_s = set(range(len(targs)))
    targs_matches_s = set(matches[0])

    preds_s = set(range(len(preds)))
    matches[1].sort()
    preds_matches_s = set(matches[1])

    fnegs = targs_s - targs_matches_s
    fnegatives = len(fnegs)

    predicted_not_matched = preds_s - preds_matches_s
    print(predicted_not_matched)
    fpostives = len(predicted_not_matched)

    matched = torch.FloatTensor(matches[-1])
    print(matched.shape)
    tpositives = np.array(matched.sum())

    print(f"TP: {tpositives}, FP: {fpositives}, FN: {fnegatives}")

    precision = tpositives / (tpositives + fpositives)
    print(f"Precision {precision}")

    recall = tpositives / (tpositives + fnegatives)
    print(f"Recall{recall}")
    
    return precision


def _draw_outline(o:Patch, lw:int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax:plt.Axes, b:Collection[int], color:str='white', text=None, text_size=14):
    "Draw bounding box on `ax`."
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    _draw_outline(patch, 4)
    if text is not None:
        patch = ax.text(*b[:2], text, verticalalignment='top', color=color, fontsize=text_size, weight='bold')
        _draw_outline(patch,1)

def bvol_iou(boxA, boxB):
  
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])
    
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    
    interW = xB - xA + 1
    interH = yB - yA + 1
    interD = zB - zA + 1
    
    if interW <=0 or interH <=0 or interD <=0 :
        return -1.0  # value for non-overlapping box

    interVol = interW * interH * interD
    
    boxAVol = (boxA[3] - boxA[0] + 1) * (boxA[4] - boxA[1] + 1) * (boxA[5] - boxA[2] + 1)
    boxBVol = (boxB[3] - boxB[0] + 1) * (boxB[4] - boxB[1] + 1) * (boxB[5] - boxB[2] + 1)
    
    iou = interVol / float(boxAVol + boxBVol - interVol)

    return iou


def roi_pool_2d(img_orig, bbs, input_size = (128,128),output_size = (128,128),
               batch_size = 1, n_channels = 1, spatial_scale = 1.0):
    rois = torch.FloatTensor(bbs)

    x_np = np.arange(batch_size * n_channels *
                     input_size[0] * input_size[1],
                     dtype=np.float32)

    x_np = x_np.reshape((batch_size, n_channels, *input_size))
    np.random.shuffle(x_np)
    
    x_np = (img_orig.reshape((1,1,img_orig.shape[0],img_orig.shape[1])))
    
    x = torch.from_numpy(x_np)
    x = x.cuda()
    rois = rois.cuda()
    
    y = roi_pooling_2d(x, rois, output_size,spatial_scale=spatial_scale)

    return y


def get3dcc(pred_vol):
    output_tensor = torch.FloatTensor(pred_vol)
    input_tens = torch.abs(output_tensor.unsqueeze(0))
    #input_tens = input_tens / torch.max(input_tens)
    pred_ssm = kornia.dsnt.spatial_softmax_2d(input_tens, temperature=0.1)
    
    print(f"SSM stats: {summary_stats(pred_ssm.numpy())}")
    
    pred_ssm_arr = pred_ssm.squeeze(0).detach().numpy()
    pred_vol_mask = (((pred_vol > 0.45)) * 1.0).astype(np.uint16)
    #pred_vol_mask = pred_ssm_arr.astype(np.uint16)
    #print(pred_vol_mask.shape)
    connectivity = 6
    labels_out = cc3d.connected_components(pred_vol_mask,  connectivity=connectivity) # 26-connected

    return labels_out


# prepare bounding volumes
# I: mask 
# O: list of ROI
def prepare_bv(mask):
    label_vol = get3dcc(pred_vol)
    list_of_roi = regionprops(label_vol)
    return list_of_roi

def plot_bb_2d(img, pred_info):
    ax = None
    if ax is None: _, ax = plt.subplots(1,1)   
    ax.imshow(img)
    
    bbs = pred_info['bbs']
    preds = pred_info['preds']
    scores = pred_info['scores']
    
    for bbox, c, scr in zip(bbs, preds, scores):
        txt = c
        draw_rect(ax, [bbox[1],bbox[0],bbox[3],bbox[2]], text=f'{txt} {scr:.2f}')
       

