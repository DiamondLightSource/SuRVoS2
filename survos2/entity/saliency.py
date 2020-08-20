import numpy as np
import pandas as pd 
from skimage import data, measure
import skimage
import os

from scipy import ndimage
import torch.utils.data as data
from typing import List

from matplotlib import pyplot as plt
import torch
import numpy as np
from survos2.frontend.nb_utils import show_images

from typing import Collection
from matplotlib import patches, patheffects
from matplotlib.patches import Rectangle, Patch


def filter_proposal_mask(proposal_mask, thresh=0.5, num_erosions=3, num_dilations=3, num_medians=1):
    holdout = (proposal_mask >= thresh) * 1.0
    struct2 = ndimage.generate_binary_structure(3, 2)
    
    for i in range(num_erosions):
        holdout = ndimage.binary_erosion(holdout, structure=struct2).astype(holdout.dtype)
        
    for i in range(num_dilations):
        holdout = ndimage.binary_dilation(holdout, structure=struct2).astype(holdout.dtype)

    for i in range(num_medians):
        holdout= ndimage.median_filter(holdout, 4).astype(holdout.dtype)
    
    return holdout


def measure_regions(labeled_images, properties = ['label', 'area', 'centroid', 'bbox']):
    tables = [skimage.measure.regionprops_table(image, properties=properties)
              for image in labeled_images]

    tables = [pd.DataFrame(table) for table in tables]

    tables = [table.rename(columns={'label' : 'class_code', 
                  'centroid-0':'z',
                  'centroid-1':'x',
                  'centroid-2':'y',
                  'bbox-0':'bb_s_z',
                  'bbox-1':'bb_s_x',
                  'bbox-2':'bb_s_y',
                  'bbox-3':'bb_f_z',
                  'bbox-4':'bb_f_x',
                  'bbox-5':'bb_f_y',                  
                  }) for table in tables]
    return tables



def filter_small_components(images):

    labeled_images = [measure.label(image) for image in images]
    tables = measure_regions(labeled_images)
    #coss (components_of_sufficient_size)
    coss = [tables[i][tables[i]['area'] > 1000] for i in range(len(tables))]
    cois = [tables[i][tables[i]['area'] < 1000] for i in range(len(tables))]
    
    filtered_images = []

    for img_idx in range(len(images)):
        too_small = list(coss[img_idx]['class_code'])
        total_mask = np.zeros_like(images[img_idx])

        for idx in too_small:
            mask = (labeled_images[img_idx] == idx) * 1.0
            total_mask = total_mask + mask

        filtered_images.append((total_mask * images[img_idx]) * 1.0)
        
    return filtered_images


def measure_big_blobs(images : List[np.ndarray]):
    filtered_images = filter_small_components(images)
    labeled_images = [measure.label(image) for image in filtered_images]
    filtered_tables = measure_regions(labeled_images)
    return filtered_tables
    
def get_entity_at_loc(entities_df, selected_idx):
    return entities_df[np.logical_and( 
                            np.logical_and(
                            entities_df.z == selected_idx[0], 
                            entities_df.x == selected_idx[1]),
                            entities_df.y == selected_idx[2]) ]


def generate_click_plot_data1(img_data, click_coords):
    img_shortlist = []
    img_titles = []

    for j in range(len(click_coords)):

        if j % 5000 == 0:
            print("Generating click plot data: {}".format(j))

        sliceno, y, x = click_coords[j]
        w, h = (100, 100)
        print(x, y, w, h, sliceno)

        img = get_img_in_bbox(img_data, 75, int(np.ceil(x)), int(np.ceil(y)), w, h)
        img_shortlist.append(img)

        y_str = "{:10.4f}".format(y)
        x_str = "{:10.4f}".format(x)
        img_titles.append(x_str + " " + y_str + " " + "Slice no: " + str(sliceno))

    return img_shortlist, img_titles

