from typing import Collection

import numpy as np
import scipy
import torch
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from survos2.entity.utils import pad_vol
from survos2.entity.sampler import centroid_to_bvol, viz_bvols, offset_points
from survos2.frontend.nb_utils import slice_plot
from torch import Tensor


def analyze_detector_predictions(
    gt_entities,
    detected_entities,
    bvol_dim=(24, 24, 24),
    debug_verbose=True,
):
    """Calculates and prints detector evaluation given ground truth and detected entity arrays.

    Parameters
    ----------
    gt_entities : np.ndarray
        Entity array
    detected_entities : np.ndarray
        Entity array 
    bvol_dim : tuple, optional
        Dimension of bounding volume, by default (24, 24, 24)
    debug_verbose : bool, optional
        Print debug output, by default True

    """
    print(f"Evaluating detections of shape {detected_entities.shape}")

    preds = centroid_to_bvol(detected_entities, bvol_dim=bvol_dim)
    targs = centroid_to_bvol(gt_entities, bvol_dim=bvol_dim)
    eval_result = eval_matches(preds, targs, debug_verbose=debug_verbose)



def analyze_detector_result(
    wf,
    gt_entities,
    detected_entities,
    padding=(0, 0, 0),
    mask_bg=False,
    plot_location=(30, 200, 200),
    bvol_dim=(24, 24, 24),
    debug_verbose=True,
    bg_vol=None
):

    print(f"Evaluating detections of shape {detected_entities.shape}")


    padded_vol = wf.vols[0]
    padded_mask = wf.bg_mask
    #padded_vol = pad_vol(wf.vols[1], padding)
    #padded_mask = pad_vol(wf.bg_mask, padding)

    if mask_bg:
        # remove bg detections
        #padded_mask = pad_vol((wf.bg_mask == 1) * 1.0, padding)
        #detected_entities = offset_points(detected_entities, np.array(padding))
        centroid_locs = detected_entities[:, 0:3]
        centroid_mask = np.zeros_like(padded_vol)

        for p in centroid_locs:
            centroid_mask[p[0], p[1], p[2]] = 1

        centroid_mask = centroid_mask * padded_mask
        zs, xs, ys = np.where(centroid_mask == 1)

        dets = []
        for i in range(len(zs)):
            dets.append((zs[i], xs[i], ys[i], 0))

        detected_entities = np.array(dets)
        print(f"After masking have detections of shape {detected_entities.shape}")

    mask_dets = viz_bvols(
        bg_vol, centroid_to_bvol(detected_entities, bvol_dim=bvol_dim, flipxy=True)
    )
    mask_gt = viz_bvols(
        bg_vol, centroid_to_bvol(gt_entities, bvol_dim=bvol_dim, flipxy=True)
    )

    slice_plot(bg_vol, 
                None, 
                mask_dets, 
                np.array(plot_location) + np.array(padding), 
                suptitle="Analyze Detector Results", 
                plot_color=True)

    preds = centroid_to_bvol(detected_entities, bvol_dim=bvol_dim)
    targs = centroid_to_bvol(gt_entities, bvol_dim=bvol_dim)
    eval_result = eval_matches(preds, targs, debug_verbose=debug_verbose)

    detected_entities = offset_points(detected_entities, bvol_dim)

    return detected_entities, mask_gt, mask_dets


def score_dice(pred, targ, dim=1, smooth=1) -> Tensor:
    """Calculate the Dice score between predicted and target arrays

    Parameters
    ----------
    pred : np.ndarray
        Predicted image volume
    targ : np.ndarray
        Target image volume
    dim : int, optional
        Dimension to sum along, by default 1
    smooth : int, optional
        Smoothing eps value, by default 1

    Returns
    -------
    Tensor
        Dice score
    """
    pred = torch.FloatTensor(pred)
    targ = torch.FloatTensor(targ)
    pred, targ = pred.contiguous(), targ.contiguous()
    intersection = (pred * targ).sum(dim=dim).sum(dim=dim)
    cardinality = (
        pred.sum(dim=dim).sum(dim=dim) + targ.sum(dim=dim).sum(dim=dim) + smooth
    )
    loss = (2.0 * intersection + smooth) / cardinality

    return loss.mean()


def bvol_iou(boxA, boxB):
    """Calculate intersection-over-union between bounding volumes

    Parameters
    ----------
    boxA : np.ndarray
        bounding box coordinates in [x_start, y_start, z_start, x_end, y_end, z_end]
    boxB : np.ndarray
        bounding box coordinates

    Returns
    -------
    float
        iou
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])

    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])

    interW = xB - xA + 1
    interH = yB - yA + 1
    interD = zB - zA + 1

    if interW <= 0 or interH <= 0 or interD <= 0:
        return -1.0  # non-overlapping

    interVol = interW * interH * interD

    boxAVol = (
        (boxA[3] - boxA[0] + 1) * (boxA[4] - boxA[1] + 1) * (boxA[5] - boxA[2] + 1)
    )

    boxBVol = (
        (boxB[3] - boxB[0] + 1) * (boxB[4] - boxB[1] + 1) * (boxB[5] - boxB[2] + 1)
    )

    iou = interVol / float(boxAVol + boxBVol - interVol)

    return iou


# adapted from medicaldetectiontoolkit 2d
def match_bvol(iou_matrix, bvol_gt, bvol_pred, iou_thresh=0.5):
    """Match bounding volumes

    Parameters
    ----------
    iou_matrix : Precalculated array of bounding volume ious
        [description]
    bvol_gt : np.ndarray
        Gold bvol
    bvol_pred : [type]
        [description]
    iou_thresh : float, optional
        [description], by default 0.5

    Returns
    -------
    idx_gt_actual,
    idx_pred_actual,
    ious_actual,
    label,
    """
    n_true = bvol_gt.shape[0]
    n_pred = bvol_pred.shape[0]

    min_iou = 0.0

    if n_pred > n_true:
        diff = n_pred - n_true
        iou_matrix = np.concatenate(
            (iou_matrix, np.full((diff, n_pred), min_iou)), axis=0
        )

    if n_true > n_pred:
        diff = n_true - n_pred
        iou_matrix = np.concatenate(
            (iou_matrix, np.full((n_true, diff), min_iou)), axis=1
        )

    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]

    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]

    sel_valid = ious_actual > iou_thresh
    label = sel_valid.astype(int)

    return (
        idx_gt_actual[sel_valid],
        idx_pred_actual[sel_valid],
        ious_actual[sel_valid],
        label,
    )


def detector_eval(preds, targs, matches, debug_verbose=True):
    """Evaluation of detector results (TP, FP, FN, Precision, Recall, F1)

    Parameters
    ----------
    preds : np.ndarray
        Predictions
    targs : np.ndarray
        Targets
    matches : np.ndarray
        Matches
    debug_verbose : bool, optional
        Print debug output, by default True

    Returns
    -------
    Tuple
        Evaluation results
    """
    targs_s = set(range(len(targs)))
    targs_matches_s = set(matches[0])

    preds_s = set(range(len(preds)))
    matches[1].sort()
    preds_matches_s = set(matches[1])

    fnegs = targs_s - preds_matches_s
    fnegatives = len(fnegs)

    predicted_not_matched = preds_s - preds_matches_s
    fpositives = len(predicted_not_matched)

    matched = torch.FloatTensor(matches[-1])
    tpositives = matched.sum().numpy()
    precision = tpositives / (tpositives + fpositives)
    recall = tpositives / (tpositives + fnegatives)
    f1 = 2 * (precision * recall) / (precision + recall)

    if debug_verbose:
        print(f"TP: {tpositives}, FP: {fpositives}, FN: {fnegatives}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")

    return (tpositives, fpositives, fnegatives, precision, recall, f1)


def eval_matches(
    preds,
    targs,
    iou_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9],
    debug_verbose=False,
):
    iou_matrix = get_matches(preds, targs)
    result = []
    for iout in iou_thresholds:
        matches = match_bvol(iou_matrix, preds, targs, iou_thresh=iout)
        num_matches = np.sum(matches[-1])
        num_targets = len(matches[-1])
        result.append(detector_eval(preds, targs, matches, debug_verbose=debug_verbose))
        if debug_verbose:
            print(
                f"At IOU thresh of {iout} No. Matches {num_matches}, No.targets {num_targets}\n"
            )
    return result

def get_matches(bbox_gt, bbox_pred):
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    print(n_true, n_pred)
    iou_matrix = np.zeros((n_true, n_pred))
    ctr = 0

    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bvol_iou(bbox_gt[i, :], bbox_pred[j, :])
        ctr += 1
        if (ctr % 100) == 0:
            print(ctr)

    return iou_matrix


def _draw_outline(o: Patch, lw: int):
    "Outline bounding box onto image `Patch`."
    o.set_path_effects(
        [patheffects.Stroke(linewidth=lw, foreground="black"), patheffects.Normal()]
    )


def draw_bb(
    ax: plt.Axes, b: Collection[int], color: str = "white", text=None, text_size=14
):
    patch = ax.add_patch(
        patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2)
    )

    patch.set_path_effects(
        [patheffects.Stroke(linewidth=4, foreground="black"), patheffects.Normal()]
    )

    if text is not None:
        patch = ax.text(
            *b[:2],
            text,
            verticalalignment="top",
            color=color,
            fontsize=text_size,
            weight="bold",
        )
        patch.set_path_effects(
            [patheffects.Stroke(linewidth=1, foreground="black"), patheffects.Normal()]
        )


def detection_to_mask(
    bbs_table,
    img_vol,
    bvol_dim=(32, 24, 24),
    outlier_size_thresh=5,
    size_thresh=None,
    flipxy=False,
):
    filtered_tables = [bbs_table]
    tbl_idx = 0

    outlier_size_thresh = np.mean(
        filtered_tables[tbl_idx]["area"]
    ) + outlier_size_thresh * np.std(filtered_tables[tbl_idx]["area"])

    sel_entities = filtered_tables[tbl_idx][
        filtered_tables[tbl_idx]["area"] <= outlier_size_thresh
    ]

    sel_entities = filtered_tables[0]
    cols = ["z", "x", "y", "class_code"]

    if size_thresh is not None:
        sel_entities = sel_entities[sel_entities["area"] > size_thresh]

    pred_cents = np.array(sel_entities[cols])
    preds_bvols = centroid_to_bvol(pred_cents, bvol_dim=bvol_dim, flipxy=flipxy)
    preds_mask = viz_bvols(img_vol, preds_bvols)

    return preds_mask, pred_cents, preds_bvols


def analyze_detections_(
    wf,
    gt_entities,
    detected_entities,
    padding,
    mask_bg=False,
    plot_location=(30, 200, 200),
    plot_all=False,
):
    print(f"Evaluating detections of shape {detected_entities.shape}")

    padded_vol = pad_vol(vol, padding)

    if plot_all:
        mask_all = viz_bvols(
            wf.vols[0], centroid_to_bvol(detected_entities, bvol_dim=(20, 20, 20))
        )
        mask_gt = viz_bvols(
            wf.vols[0], centroid_to_bvol(gt_entities, bvol_dim=(20, 20, 20))
        )
        slice_plot(
            mask_gt,
            detected_entities,
            mask_all * wf.bg_mask,
            plot_location,
            plot_color=True,
        )

        mask_dets = viz_bvols(
            padded_vol, centroid_to_bvol(detected_entities, bvol_dim=(20, 20, 20))
        )

        slice_plot(mask_dets, None, padded_vol, plot_location, plot_color=True)

    preds = centroid_to_bvol(detected_entities, bvol_dim=(40, 30, 30))
    targs = centroid_to_bvol(wf.locs, bvol_dim=(40, 30, 30))
    eval_matches(preds, targs)

    return detected_entities

