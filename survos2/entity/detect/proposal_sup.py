"""Prepare supervision for Proposal Net

"""
import hdbscan
from collections import Counter
from statistics import mode, StatisticsError
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import time
import glob

import collections
import numpy as np
import pandas as pd
from typing import NamedTuple
import itertools

from scipy import ndimage
import torch.utils.data as data

import skimage
from skimage.morphology import thin
from skimage.io import imread, imread_collection
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from numpy.lib.stride_tricks import as_strided as ast
from numpy.random import permutation
from numpy import linalg

from UtilityCluster import summary_stats
from numpy.linalg import LinAlgError

#warnings.filterwarnings("ignore")
#warnings.filterwarnings(action='once')

from survos2.entity.anno.geom import centroid_3d, rescale_3d



#subsample usefu for really large point clouds
#make sure it should be on if it is though...
def chip_cluster(orig_pts, chip, offset_x, offset_y, 
                 plot_all= False, debug_verbose=False):
    
    X  = orig_pts.copy()
    img_sample = chip[0,:]
    
    data_proj = np.array(X)

    subsample=False
    
    print(f"Wide patch {chip.shape}")
    if subsample:
        samp_prop = 0.3
        num_samp = int(samp_prop * len(data_proj))

        data_proj = np.floor(permutation(data_proj))[0:num_samp, :]


    scale_minmax = False
    
    if scale_minmax:
        
        scale_x = 1.0/np.max(X[:,0])
        scale_y = 1./np.max(X[:,1])
        scale_z = 1./np.max(X[:,2])
        if debug_verbose:
            print("Scaling by {} {} {}".format(scale_x,scale_y,scale_z))

    #X_rescaled = rescale_3d(orig_pts, scale_x, scale_y, scale_z)
        x_scale = xend-xstart
        y_scale = yend-ystart
        slicestart=0
        sliceend = vol_shape_z
        z_scale = (sliceend-slicestart)*5 # HACK (not sure how best to scale z)
        X_rescaled = rescale_3d(X, scale_x, scale_y, scale_z)
    else:
        X_rescaled = X.copy()
   
    if scale_minmax:
        xlim = (np.min(X_rescaled[:,1]).astype(np.uint16)-0.1, np.max(X_rescaled[:,1]).astype(np.uint16)+0.1)
        ylim = (np.min(X_rescaled[:,2]).astype(np.uint16)-0.1, np.max(X_rescaled[:,2]).astype(np.uint16)+0.1)
        zlim = (np.min(X_rescaled[:,0]).astype(np.uint16)-0.1, np.max(X_rescaled[:,0]).astype(np.uint16)+0.1)
    else:
        xlim = (0,chip.shape[1])
        ylim = (0,chip.shape[2])
        zlim = (0,chip.shape[0])
        
    #print(np.max(X_rescaled[:,0]), np.max(X_rescaled[:,1]), np.max(X_rescaled[:,2]))
    #print(np.min(X_rescaled[:,0]), np.min(X_rescaled[:,1]), np.min(X_rescaled[:,2]))
    
    #print(np.max(X_rescaled[:,0]), np.max(X_rescaled[:,1]), np.max(X_rescaled[:,2]))
    #print(np.min(X_rescaled[:,0]), np.min(X_rescaled[:,1]), np.min(X_rescaled[:,2]))

    #print(np.max(X_rescaled[:,0]), np.max(X_rescaled[:,1]), np.max(X_rescaled[:,2]))
    #print(np.min(X_rescaled[:,0]), np.min(X_rescaled[:,1]), np.min(X_rescaled[:,2]))

    #
    # Point cloud cluster
    #
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=6).fit(X_rescaled)

    label_code = clusterer.labels_

    num_clusters_found = len(np.unique(label_code))
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.75)

    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

    X_rescaled_cl = np.delete(X_rescaled,  outliers,axis=0)
    label_code_cl = np.delete(label_code,  outliers,axis=0)
    cluster_probs_cl = np.delete(clusterer.probabilities_, outliers, axis=0)
    num_outliers_removed = X_rescaled.shape[0]-X_rescaled_cl.shape[0]
    
    if debug_verbose:
        #print(X_rescaled_cl.shape, label_code_cl.shape)
        print("Limits: {} {} {} ".format(xlim,ylim,zlim))
        print(np.min(X[:,0]), np.min(X[:,1]), np.min(X[:,2]))
        print("Orig: {} Clean: {} Num points rem: {}".format(X_rescaled.shape[0], X_rescaled_cl.shape[0], num_outliers_removed))
        print("Proportion removed: {}".format(num_outliers_removed/X_rescaled.shape[0]))
    
    if plot_all:
        
        plt.figure(figsize=(14,14))
        #pal = sns.color_palette('husl', len(label_code_cl))
        pal = sns.color_palette('cubehelix', num_clusters_found + 1)
        #pal = sns.color_palette('Paired', num_clusters_found + 1)

        cleaned_colors = [sns.desaturate(pal[col], sat) for col, sat in zip(label_code_cl,
                                                                    cluster_probs_cl)]
        plt.imshow(img_sample, cmap='gray')
        plt.scatter(X_rescaled_cl[:,1] -offset_x , X_rescaled_cl[:,2] - offset_y, s=60, linewidth=0, c=cleaned_colors, alpha=0.25)

    cluster_coords = []
    cluster_sizes = []

    for l in np.unique(label_code_cl)[0:]:
        cluster_coords.append(X_rescaled_cl[label_code_cl==l])

    cluster_coords = np.array(cluster_coords)
    
    cluster_sizes = np.array([len(cluster_coord) 
                              for cluster_coord in cluster_coords])
    print(f"Cluster sizes {cluster_sizes.shape}")

    cluster_centroids = np.array([centroid_3d(cluster_coord) 
                                  for cluster_coord in cluster_coords 
                                  if len(cluster_coord) > 5])

    cluster_centroids = np.array(cluster_centroids)
    
    #sns.distplot(cluster_centroids[np.isfinite(cluster_centroids)], rug=True)
    
    cluster_sizes = np.array(cluster_sizes)
    
    #sns.distplot(cluster_sizes[np.isfinite(cluster_sizes)], rug=True)

    title_str = "Number of original clicks: {0} Number of final centroids: {1} Av clicks cluster {2}".format(X_rescaled.shape[0], cluster_sizes.shape[0], X_rescaled.shape[0]/ cluster_sizes.shape[0])
    
    slice_center = int(chip.shape[0]/2.0)
    cc2 = np.roll(cluster_centroids, shift=2, axis=1)
    slice_top, slice_bottom = slice_center+5, slice_center-5
    
    centroid_coords_woffset = cc2.copy()  
    centroid_coords_woffset[:,1] = centroid_coords_woffset[:,1] - offset_y
    centroid_coords_woffset[:,0] = centroid_coords_woffset[:,0] - offset_x
    
    cc = []
    
    for c in cluster_coords:
        
        cluster_classes = list(c[:,3].astype(np.uint32))
        
        try:
            classes_mode = mode(cluster_classes)
            
        except StatisticsError as e:
            classes_mode = np.random.choice(cluster_classes)
                
        cc.append(classes_mode)
        #print(f"Assigned class for cluster: {classes_mode}")
    
    if debug_verbose:
        print(f"Number of clusters: {len(cluster_coords)}")
        print(f"Cluster classes {cc}")
        print(f"Len cluster classes {len(cc)}")
    
    if plot_all:
        show_images_and_points([img_sample,], centroid_coords_woffset, cc, figsize=(12,12))
    
    return cluster_centroids, cc
    
def show_chips(chipsdata,  source_positions, patch_size=(32,32), plot_all=False, debug_verbose=False):
    
    chips_arr = []
    chips_maps = []
    centroids_data = []
    
    kk=0
    for (wide_patch, wide_img_titles, click_data_wide_arr),source_pos in zip(chipsdata, chips_pos):
    
        if kk % 100 == 0:
            print(f"\n {kk}\n")
        kk+=1
        
        print(f"Wide patch {wide_patch}")
        img_samples = wide_patch[range(1,wide_patch.shape[0],2),:]
        img_sample = img_samples[0]
        
        offset_x = source_pos[1]-(patch_size[1]/2.0)
        offset_y = source_pos[2]-(patch_size[2]/2.0)
        offset_z = 0
        
        if debug_verbose:
            print(f"\n {kk}\n")
            print(f"Offset {offset_x, offset_y, offset_z}")
            print(patch_size[1], patch_size[2])
        
        x = click_data_wide_arr[:,1].copy()
        y = click_data_wide_arr[:,2].copy() 
        z = click_data_wide_arr[:,0].copy() 
        
        clrs = click_data_wide_arr[:,3]
        clrs =  clrs.astype(np.uint32)
        sns.reset_orig()  # get default matplotlib styles back
        
        #flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        
        huslpal = sns.color_palette("husl", 10)
        pal = np.array(sns.color_palette(huslpal, n_colors=10))  # a list of RGB tuples
        clrs = pal[clrs]
   
        x_ip = click_data_wide_arr[:,1] - offset_x
        y_ip = click_data_wide_arr[:,2] - offset_y
        z_ip = click_data_wide_arr[:,0] - offset_z

        try:
            if img_sample.shape[0] > 0:
                d_map = density_map(x_ip,y_ip,z_ip,clrs, img_sample)
                
            else:
                d_map = np.zeros_like(img_sample)
                img_sample = np.zeros((10,10))

        except (ValueError,LinAlgError) as e:
            d_map = np.zeros_like(img_sample)

        if plot_all:
            plt.figure(figsize=(6,6))
            middle_of_patch = int(img_sample.shape[0]/2)
            plt.imshow(img_sample, cmap='gray')
            plt.scatter(x_ip, y_ip, alpha=0.2, c=clrs)
        
        try:
            if img_sample.shape[0] > 0:

                cluster_centroids, cc = chip_cluster(click_data_wide_arr, wide_patch, 
                                                offset_x, offset_y, debug_verbose=False, plot_all=plot_all)
                x_cent = cluster_centroids[:,1] - offset_x
                y_cent = cluster_centroids[:,2] - offset_y
                z_cent = cluster_centroids[:,0] - offset_z

                centroid_density = density_map(x_cent,y_cent,z_cent,clrs, img_sample)
                centroid_classes = np.array(cc).reshape((len(cc),1))

                centroids_alldata = np.append(cluster_centroids, centroid_classes, axis=1)
                centroids_data.append(centroids_alldata)
            
            else:
                centroid_density = np.zeros_like(img_sample)
                centroids_data.append([0,0,0,0])
                
        except (ValueError,LinAlgError) as e:
            
            centroid_density = np.zeros_like(img_sample)
            centroids_data.append([0,0,0,0])
        
        chips_maps.append((img_sample,d_map, centroid_density))
        chips_arr.append([x,y,z,clrs, x_ip,y_ip,z_ip, offset_x, offset_y,offset_z])
    
    chips_arr = np.array(chips_arr)
    
    return chips_arr, chips_maps, centroids_data


"""
change the underlying bb rep from center + w,h,d to lbl to utr
resample z? (x 5?) then resize the image?
"""

def prepare_chip_data(img_data, orig_click_data, 
                            wide_patch_pos=(63, 650, 650), 
                            z_depth=3, patch_size=(40,200,200),
                      debug_verbose=False):  
    """
    Takes an image vol and a set of locations and produces a cropped vol and set of locations
    
    """
    img_titles = []
    patch_size=np.array(patch_size)
    sliceno,x,y = wide_patch_pos

    vol_shape_z = img_data.shape[0]
    
    z_depth, p_x, p_y  = patch_size
    
    w = int(p_x / 2.0)
    h = int(p_y / 2.0)
    
    #sel_cropped_clicks.append((sliceno, x,y,w,h))
    
    z, x_bl, x_ur, y_bl, y_ur = int(sliceno), x-w, x+w, y-h, y+h
    
    slice_start = np.max([0,wide_patch_pos[0] - np.int(patch_size[0]/2.0)])
    slice_end = np.min([wide_patch_pos[0] + np.int(patch_size[0]/2.0) , vol_shape_z])
    
    out_of_bounds_w = np.hstack((np.where(orig_click_data[:,1] >= x_ur)[0],
                           np.where(orig_click_data[:,1] <= x_bl)[0], 
                           np.where(orig_click_data[:,2] >= y_ur)[0],  
                           np.where(orig_click_data[:,2] < y_bl)[0],
                           np.where(orig_click_data[:,0] <= slice_start)[0],
                           np.where(orig_click_data[:,0] >= slice_end)[0]))

    click_data_w= np.delete(orig_click_data, out_of_bounds_w, axis=0)  
    
    click_data_wide_arr = np.array(click_data_w)
    num_clicks_selected = click_data_wide_arr.shape[0]
    
    if debug_verbose:
        print("\n x y w h, sliceno: {}".format((x,y,w,h,sliceno)))
        print(z,x_bl, x_ur, y_bl, y_ur)
        print("Slice start, slice end {} {}".format(slice_start, slice_end))

        print("Click_data_wide_arr shape: {}".format(click_data_wide_arr.shape))

        #click_data_wide_arr[:,0] = click_data_wide_arr[:,0] - x_bl
        #click_data_wide_arr[:,1] = click_data_wide_arr[:,0] - y_bl
        print("Length of original click_data {}".format(orig_click_data.shape[0]))
        print("Length after deleting out of bounds clicks: {}".format(num_clicks_selected))
    
    if num_clicks_selected < 20:
        click_data_wide_arr = click_data_wide_arr[0:1]

   # try:
   #     sel_wide_clicks, x_coords_range, y_coords_range = generate_clicklist(click_data_wide_arr, crop_roi,slice_start,slice_end)
   # except ValueError as e:

    if z_depth > 1:
        img = get_vol_in_bbox(img_data, slice_start, slice_end, int(np.ceil(y)),int(np.ceil(x)),h,w)
    else:
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)),int(np.ceil(y)),w,h)

    y_str = "{:10.4f}".format(y)
    x_str = "{:10.4f}".format(x)

    img_titles.append(x_str + " " + y_str + "\n " + "Slice no: " + str(sliceno))
    
    return img, img_titles, click_data_wide_arr



def make_kde2d(data, bandwidth_scaling=2.5):
    values = data.T
    kde = stats.gaussian_kde(values)

    
    kde.set_bandwidth(bw_method=kde.factor / bandwidth_scaling) 
    density = kde(values)
    
    return density, kde



#classwise
#bandwidth_scaling=5.1
def density_map(x,y,z,c, image1, plot_all=False, debug_verbose=False, 
                bandwidth_scaling=5.8):
    z = [1]*len(x)
    density_data = np.vstack((x, y,z)).T
    density_data.shape
    data = density_data.copy()

    try:
        density, kde = make_kde2d(data[:,0:2], bandwidth_scaling=bandwidth_scaling)
        d_thr = np.mean(density) + 1 * np.std(density)
        
        density_over_thr = d_thr > d_thr
        density_thresholded = density.copy()
        density_thresholded[density_over_thr] = 0
        density_thresholded_norm = (density_thresholded/np.max(density_thresholded)).astype(np.float16)
        
        summary_stats(density_thresholded_norm)
        density_colors = np.vstack([density_thresholded_norm] * 3).T
        
        
        if plot_all:
            plt.figure()
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.scatter(data[:,0], data[:,1])#, c=density_colors)
        
        
        xmin, xmax = 0, image1.shape[0]
        ymin,ymax = 0, image1.shape[1]
        
        
        small_side_ratio = ymax/xmax

        if debug_verbose:
            print(density_colors.shape, data.shape)
            print(f"Small side ratio {small_side_ratio}")
            print(xmin, xmax, ymin, ymax)
            print(np.max(density), np.min(density), np.mean(density))
            print(np.sum(density > d_thr))

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:(100j * small_side_ratio)]
        positions = np.vstack([yy.ravel(), xx.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)
            
        if plot_all:
            plt.figure()
            fig, ax = plt.subplots(figsize=(6,6))
            cset = ax.contourf(yy, xx, f, cmap='coolwarm')
            ax.scatter(data[:,0], data[:,1], c=density_thresholded, alpha=0.4)
            plt.gca().invert_yaxis()

        f_norm = (f.T.copy() - np.min(f))
        f_norm = f_norm/np.max(f_norm)
        f_norm = skimage.transform.resize(f_norm, (image1.shape[0], image1.shape[1]))
        f_norm = f_norm.T
        if plot_all:
            plt.figure()
            plt.imshow(f_norm)
            print(summary_stats(f))
            print(summary_stats(f_norm))
    except ValueError:

        density = 0
        f_norm = np.zeros_like(image1)

    return f_norm

#density_map(x,y,z,clrs)


#
#  Group ROI
#


"""
change the underlying bb rep from center + w,h,d to lbl to utr
resample z? (x 5?) then resize the image?
"""


def generate_wide_plot_data(img_data, click_coords, wide_patch_pos=(63, 650, 650), z_depth=3,
                            patch_size=(40, 200, 200)):
    img_titles = []
    patch_size = np.array(patch_size)
    sliceno, x, y = wide_patch_pos

    z_depth, p_x, p_y = patch_size
    w = int(p_x / 2.0)
    h = int(p_y / 2.0)
    print("x y w h, sliceno: {}".format((x, y, w, h, sliceno)))
    # sel_cropped_clicks.append((sliceno, x,y,w,h))

    z, x_bl, x_ur, y_bl, y_ur = int(sliceno), x - w, x + w, y - h, y + h
    print(z, x_bl, x_ur, y_bl, y_ur)

    slice_start = np.max([0, wide_patch_pos[0] - np.int(patch_size[0] / 2.0)])
    slice_end = np.min([wide_patch_pos[0] + np.int(patch_size[0] / 2.0), 165])

    print("Slice start, slice end {} {}".format(slice_start, slice_end))
    out_of_bounds_w = np.hstack((np.where(orig_click_data[:, 1] >= x_ur)[0],
                                 np.where(orig_click_data[:, 1] <= x_bl)[0],
                                 np.where(orig_click_data[:, 2] >= y_ur)[0],
                                 np.where(orig_click_data[:, 2] < y_bl)[0],
                                 np.where(orig_click_data[:, 0] <= slice_start)[0],
                                 np.where(orig_click_data[:, 0] >= slice_end)[0]))

    click_data_w = np.delete(orig_click_data, out_of_bounds_w, axis=0)

    click_data_wide_arr = np.array(click_data_w)
    print("Click_data_wide_arr shape: {}".format(click_data_wide_arr.shape))

    # click_data_wide_arr[:,0] = click_data_wide_arr[:,0] - x_bl
    # click_data_wide_arr[:,1] = click_data_wide_arr[:,0] - y_bl
    print("Length of original click_data {}".format(orig_click_data.shape[0]))
    print("Length after deleting out of bounds clicks: {}".format(click_data_wide_arr.shape[0]))

    sel_wide_clicks, x_coords_range, y_coords_range = generate_clicklist(click_data_wide_arr, crop_roi, slice_start,
                                                                         slice_end)

    if z_depth > 1:
        img = get_vol_in_bbox(img_data, slice_start, slice_end, int(np.ceil(y)), int(np.ceil(x)), h, w)
    else:
        img = get_img_in_bbox(img_data, sliceno, int(np.ceil(x)), int(np.ceil(y)), w, h)

    y_str = "{:10.4f}".format(y)
    x_str = "{:10.4f}".format(x)

    img_titles.append(x_str + " " + y_str + "\n " + "Slice no: " + str(sliceno))

    return img, img_titles, click_data_wide_arr

def patchify(img, patch_shape):
    X, Y, a = img.shape
    x, y = patch_shape
    shape = (X - x + 1, Y - y + 1, x, y, a)
    X_str, Y_str, a_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, a_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def sliding_patches(a, BSZ):
    hBSZ = (BSZ - 1) // 2
    a_ext = np.dstack(np.pad(a[..., i], hBSZ, 'wrap') for i in range(a.shape[2]))
    return patchify(a_ext, (BSZ, BSZ))


def mask2vectors(mask):
    distances, indices = distance_transform(mask)
    grid_indices = np.indices((mask.shape[0], mask.shape[1]))
    distances[distances == 0] = 1
    return (indices * (mask > 255 // 2) - grid_indices * (mask > 255 // 2)) / np.asarray([distances, distances])


 

"""Use a density map to extract entities. 
Different scales result in different entities.

"""
def density_trimap(data, image_2d, bg_thr=0.6, obj_thr=1.4, orig_img_dim=(100, 100, 100)):

    data = np.vstack((x, y)).T
    data.shape

    density, kde = make_kde2d(data)

    print(np.max(density), np.min(density), np.mean(density))
    d_thr = np.mean(density) + 1 * np.std(density)
    print(np.sum(density > d_thr))

    density_over_thr = d_thr > d_thr
    density_thresholded = density.copy()
    density_thresholded[density_over_thr] = 0
    density_thresholded_norm = density_thresholded / np.max(density_thresholded)

    # In[]

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # x, y, z = values
    # ax.scatter(x, y, z, c=density_thresholded)
    # plt.show()

    # In[]

    # deltaX = (np.max(x) - np.min(x)) / 3
    # deltaY = (np.max(y) - np.min(y)) / 3

    # xmin = np.min(x) #- deltaX
    # xmax = np.max(x) #+ deltaX
    # ymin = np.min(y) #- deltaY
    # ymax = np.max(y) #+ deltaY
    xmin, xmax = 0, image_2d.shape[0]
    ymin, ymax = 0, image_2d.shape[1]
    print(xmin, xmax, ymin, ymax)

    # xmin, xmax = 0, wide_patch.shape[1]
    # ymin,ymax = 0, wide_patch.shape[2]
    # print(xmin, xmax, ymin, ymax)

    small_side_ratio = ymax / xmax
    print(small_side_ratio)
    fig, ax = plt.subplots(figsize=(6, 6))
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:(100j * small_side_ratio)]

    # fig, ax = plt.subplots(figsize=(6,6))
    # xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([yy.ravel(), xx.ravel()])
    f = np.reshape(kde(positions).T, xx.shape)
    cset = ax.contourf(yy, xx, f, cmap='coolwarm')
    ax.scatter(x, y, c=density_thresholded, alpha=0.4)
    plt.gca().invert_yaxis()

    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 10))
    # x, y, z = values
    ax.imshow(image_2d, cmap='gray')
    ax.scatter(x, y, c=density_thresholded, alpha=0.4)

    f_norm = (f.T.copy() - np.min(f))
    f_norm = f_norm / np.max(f_norm)
    f_norm = skimage.transform.resize(f_norm, (orig_img_dim[0], orig_img_dim[1]))
    plt.imshow(f_norm)
    print(summary_stats(f))
    print(summary_stats(f_norm))

    map_farbg = f_norm > (np.mean(f_norm) - np.std(f_norm))

    map_bg = f_norm < (np.mean(f_norm) + bg_thr * np.std(f_norm))
    map_obj = f_norm < (np.mean(f_norm) + obj_thr * np.std(f_norm))
    map_unk = (f_norm > (np.mean(f_norm) + bg_thr * 0.8 * np.std(f_norm))) ^ map_obj

    map_unk = 1.0 - (1.0 * map_unk)
    map_obj = 1.0 - (1.0 * map_obj)
    map_obj = dilation(map_obj, disk(2))
    show_images([map_obj * image_2d, map_bg * image_2d, map_unk * image_2d])

    trimap = np.vstack([map_bg, map_obj, map_unk])

    return trimap




