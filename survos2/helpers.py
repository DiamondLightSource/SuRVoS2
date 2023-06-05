"""
Various utility functions

"""

import os
import ast

import numpy as np
from numpy import cos, sin, ravel
from numpy import sum, nonzero, max, min
from numpy import zeros
from numpy.lib.stride_tricks import as_strided as ast
from numpy.random import permutation

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.ndimage import grey_dilation

from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import threshold_isodata, threshold_li, threshold_otsu
from skimage.filters import (
    threshold_yen,
    threshold_mean,
    threshold_triangle,
    threshold_minimum,
)
from skimage.morphology import closing
from skimage.morphology import dilation, opening
from skimage.morphology import disk
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_ubyte

import scipy.cluster.vq as vq
import scipy.signal as signal


import napari


def view_volume(imgvol, points=[], name=""):
    """Simple single-volume viewing with napari

    Arguments:
        imgvol {numpy array} -- The image volume to view

    Keyword Arguments:
        points {list} -- list of points to add to the scene (default: {[]})
        name {str} -- name of the layer to add (default: {""})

    Returns:
        viewer object -- napari viewer object
    """
    translate_limits = (28, 100)
    size = np.array([2] * len(points))

    with napari.gui_qt():
        viewer = napari.Viewer()

        viewer.add_image(imgvol, name=name)
        if len(points) > 0:
            points = viewer.add_points(points, size=size)

    return viewer


def view_volume2(imgvol, name=""):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(imgvol, name=name)


def view_volume(imgvol, points=[], name=""):
    translate_limits = (28, 100)
    size = np.array([2] * len(points))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.theme = "light"
        viewer.add_image(imgvol, name=name)

        if len(points) > 0:
            points = viewer.add_points(points, size=size)

    return viewer


def stdize(image):
    mean, std = np.mean(image), np.std(image)
    image = image - mean
    image = image / std
    return image


def simple_norm(arr, minim=0.0, maxim=1.0):
    normed_arr = arr.copy()
    normed_arr -= np.min(arr)

    normed_arr = normed_arr / (maxim - minim)
    return normed_arr


def threechan_norm(img_data):
    img_data = img_data - img_data.mean()
    img_data = img_data - img_data.min()
    img_data /= np.max(img_data)

    return img_data


def prepare_3channel(selected_images, patch_size=(28, 28)):
    selected_3channel = []

    for i in range(len(selected_images)):
        img_out = np.zeros((patch_size[0], patch_size[1], 3))

        if i % 1000 == 0:
            print(i, selected_images[i].shape)

        try:
            img_data = selected_images[i]
            img_data = threechan_norm(img_data)

            img_out[:, :, 0] = img_data
            img_out[:, :, 1] = img_data
            img_out[:, :, 2] = img_data

            selected_3channel.append(img_out)

        except ValueError as e:
            print(e)

    return selected_3channel


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


def plot_2d_data(data, labels, titles=["x", "y"], suptitle="2d data"):
    fig = plt.figure(figsize=(8, 6))
    t = fig.suptitle(suptitle, fontsize=14)
    ax = fig.add_subplot(111)

    xs = list(data[:, 0])
    ys = list(data[:, 1])

    # sns.reset_orig()  # get default matplotlib styles back
    rgb_palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))  # a list of RGB tuples

    clrs = [rgb_palette[idx] for idx in labels]

    ax.scatter(xs, ys, c=clrs, marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")


def plot_3d_data_in_2d(
    data,
    labels,
    titles=["x", "y", "z"],
    xlim=(-50, 50),
    ylim=(-50, 50),
    suptitle="3d data in 2d",
):
    """Plot 3d data as a 2d plot with variable point size

    Arguments:
        data {np.ndarray} -- Numpy array with NRows x (X,Y)
        labels {np.ndarray} -- Array of integer labels of len NRows

    Keyword Arguments:
        titles {list[str]} -- [description] (default: {['x', 'y', 'z']})
        xlim {Tuple[float, float]} -- X axis limits (default: {(-50, 50)})
        ylim {Tupel[float, float]} -- Y axis limits (default: {(-50, 50)})
        suptitle {str} -- Overall diagram title (default: {"3d data in 2d"})
    """

    fig = plt.figure(figsize=(8, 6))
    t = fig.suptitle(suptitle, fontsize=14)
    ax = fig.add_subplot(111)

    xs = list(data[:, 0])
    ys = list(data[:, 1])
    size = list(data[:, 2] * 5)

    # sns.reset_orig()  # get default matplotlib styles back
    rgb_palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))  # a list of RGB tuples

    clrs = [rgb_palette[idx] for idx in labels]

    ax.scatter(xs, ys, c=clrs, s=size, marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_3d_data(
    data,
    labels,
    titles=["x", "y", "z"],
    suptitle="3d data",
    xlim=(-50, 50),
    ylim=(-50, 50),
    zlim=(-50, 50),
    figsize=(10, 10),
):
    """Plot 3d data as points in a 3d plot

    Arguments:
        data {np.ndarray} -- 3d numpy array
        labels {np.ndarray} -- 1d array with same number of elements as rows in data array

    Keyword Arguments:
        titles {list[str]} -- Axis titles (default: {['x', 'y', 'z']})
        suptitle {str} -- Overall diagram title (default: {"3d data"})
        xlim {Tuple[float,float]} -- X axis limits (default: {(-50, 50)})
        ylim {Tuple[float,float]} -- Y axis limits (default: {(-50, 50)})
        zlim {Tuple[float,float]} -- Z axis limits (default: {(-50, 50)})
        figsize {Tuple[float,float]} -- Figure size  (default: {(10, 10)})
    """
    fig = plt.figure(figsize=figsize)
    t = fig.suptitle(suptitle, fontsize=14)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    xs = list(data[:, 0])
    ys = list(data[:, 1])
    zs = list(data[:, 2])

    # sns.reset_orig()  # default matplotlib
    rgb_palette = sns.color_palette(
        "Spectral", n_colors=len(np.unique(labels)) + 10
    )  # a list of RGB tuples

    clrs = [rgb_palette[idx] for idx in labels]

    ax.scatter(xs, ys, zs, c=clrs, marker=".")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.grid()
    return fig, ax


def plot_4d_data(data, labels, titles=["P", "Q", "R", "S"], suptitle="4d data"):
    """Plot 4d data as 3d data with variable point size

    Arguments:
        data {np.ndarray} -- 4d numpy array
        labels {nd.ndarray} -- color values

    Keyword Arguments:
        titles {list[str]} -- List of four strings for the titles (default: {['P', 'Q', 'R', 'S']})
        suptitle {str} -- Overall title for diagram (default: {"4d data"})
    """
    fig = plt.figure(figsize=(8, 6))
    t = fig.suptitle(suptitle, fontsize=14)
    ax = Axes3D(fig)  # Method 1

    xs = list(data[:, 0])
    ys = list(data[:, 1])
    zs = list(data[:, 2])
    size = list(data[:, 3] * 3)

    # sns.reset_orig()  # get default matplotlib styles back
    rgb_palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))  # a list of RGB tuples

    clrs = [rgb_palette[idx] for idx in labels]

    ax.scatter(xs, ys, zs, c=clrs, marker="o", edgecolors="none", s=size)

    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    ax.set_zlabel(titles[2])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

        self.__dict__ = self


def make_dirs(dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin


def get_window(image_volume, sliceno, xstart, ystart, xend, yend):
    return image_volume[sliceno, xstart:xend, ystart:yend]


def parse_tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return p
        return

    except:
        return
