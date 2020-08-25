import numpy as np
from numpy import nonzero, zeros_like, zeros
from numpy.random import permutation
from napari import gui_qt
from napari import Viewer as NapariViewer
import napari
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import cv2
import time



def resource(*args):
    rdir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(rdir, 'resources', *args))

def sample_from_bw(bwimg, sample_prop):
    pp = nonzero(bwimg)
    points=zeros([len(pp[0]), 2])
    points[:,0] = pp[0]; points[:,1]=pp[1] 
    num_samp = sample_prop * points.shape[0]
    points = np.floor(permutation(points))[0:num_samp, :]
 
    return points

def quick_norm(imgvol1):
    imgvol1 -= np.min(imgvol1)
    imgvol1 = imgvol1 /np.max(imgvol1)
    return imgvol1

# center-size, single slice
def get_img_in_bbox(image_volume, sliceno, x,y,w,h):
        return image_volume[int(sliceno), x-w:x+w, y-h:y+h]


# TODO
# center-size for x/y but interval for slice
def get_vol_in_bbox(image_volume, slice_start, slice_end, x,y,w,h):
        return image_volume[slice_start:slice_end, x-w:x+w, y-h:y+h]



def prepare_point_data(pts, patch_pos):
    
    offset_z = patch_pos[0]
    offset_x = patch_pos[1]
    offset_y = patch_pos[2]
    
    print(f"Offset: {offset_x}, {offset_y}, {offset_z}")
   
    z = pts[:,0].copy() - offset_z
    x = pts[:,1].copy() - offset_x
    y = pts[:,2].copy() - offset_y
    
    c = pts[:,3].copy() 
    
    
    offset_pts = np.stack([z,x,y, c], axis=1)
    
    return offset_pts
    

def grid_of_images_and_clicks(image_list, clicks, point, n_rows, n_cols, image_titles="", figsize=(20,20)):  
  images = [image_list[i] for i in range(n_rows * n_cols)]
  f, axarr = plt.subplots(n_rows,n_cols, figsize=figsize)
  for i in range(n_rows):
    for j in range(n_cols):
      
      axarr[i,j].imshow(images[i*n_cols + j], cmap='gray', origin='lower')
      axarr[i,j].set_title(img_titles[i*n_cols + j])
      #xs,ys = clicks[i*n_cols + j][0], clicks[i*n_cols + j][1]
      axarr[i,j].scatter([point[0]],[point[1]], c="red")
      
      
  plt.rcParams["axes.grid"] = False  
  plt.tight_layout()



#plotting the selected grid around the click points
def grid_of_images2(image_list, n_rows, n_cols, image_titles="", bigtitle="", figsize=(20,20)):
    images = [image_list[i] for i in range(n_rows * n_cols)]

    if image_titles == "":
        image_titles = [str(t) for t in list(range(len(images)))]

    f, axarr = plt.subplots(n_rows,n_cols, figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):
          axarr[i,j].imshow(images[i*n_cols + j], cmap='gray')
          axarr[i,j].set_title(image_titles[i*n_cols + j],  fontsize=10)
          axarr[i,j].tick_params(labeltop= False, labelleft=False, labelbottom=False)    
    f.suptitle(bigtitle)
  
def grid_of_images(image_list, n_rows, n_cols, image_titles="", figsize=(20,20)):
  images = [image_list[i] for i in range(n_rows * n_cols)]
  
  f, axarr = plt.subplots(n_rows,n_cols, figsize=figsize)
  
  for i in range(n_rows):
    for j in range(n_cols):
      axarr[i,j].imshow(images[i*n_cols + j])
      if image_titles=="":
        axarr[i,j].set_title(str(i*n_cols + j))
      else:
        axarr[i,j].set_title("Label: "+ str(image_titles[(i*n_cols+j)]))  
          
  plt.tight_layout()    



def show_images_and_points(images,points, cluster_classes, titles=None, figsize=(12,4)):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure(figsize=figsize)
    n = 1
    plt.style.use('ggplot')
    for image,title in zip(images,titles):
        
        a = fig.add_subplot(1,n_ims,n) 
        plt.imshow(image, cmap="gray")
        scat = a.scatter(points[:,0], points[:,1], c=cluster_classes, cmap="jet_r")
        a.legend(handles=scat.legend_elements()[0], labels=cats_classes)
        
        a.set_title(title)
        n += 1


def grid_of_images_and_clicks1(image_list, clicks, point, n_rows, n_cols, image_titles="", figsize=(20, 20)):
    images = [image_list[i] for i in range(n_rows * n_cols)]
    f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n_rows):
        for j in range(n_cols):
            axarr[i, j].imshow(images[i * n_cols + j], cmap='gray', origin='lower')
            axarr[i, j].set_title(img_titles[i * n_cols + j])
            # xs,ys = clicks[i*n_cols + j][0], clicks[i*n_cols + j][1]
            axarr[i, j].scatter([point[0]], [point[1]], c="red")

    plt.rcParams["axes.grid"] = False
    plt.tight_layout()


def grid_of_images1(image_list, n_rows, n_cols, image_titles=[], figsize=(20, 20)):
    images = [image_list[i] for i in range(n_rows * n_cols)]

    f, axarr = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i in range(n_rows):
        for j in range(n_cols):
            axarr[i, j].imshow(images[i * n_cols + j])
            if len(np.array(image_titles)) == 0:
                axarr[i, j].set_title(str(i * n_cols + j))
            else:
                axarr[i, j].set_title(str(image_titles[(i * n_cols + j)]))

    plt.tight_layout()

