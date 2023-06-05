import numpy as np
from scipy import ndimage
from loguru import logger
from skimage.morphology import skeletonize as skeletonize_skimage


def erode(I, num_iter, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0

    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_erosion(I, structure=struct2).astype(I.dtype)

    I = I.astype("int")
    print(I)
    I = np.nan_to_num(I)
    return I.astype("int")


def dilate(I, num_iter, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0

    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_dilation(I, structure=struct2).astype(I.dtype)

    return np.nan_to_num(I)


def opening(I, num_iter, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0

    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_opening(I, structure=struct2).astype(I.dtype)

    return I


def closing(I, num_iter, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0

    struct2 = ndimage.generate_binary_structure(3, 2)

    for i in range(num_iter):
        I = ndimage.binary_closing(I, structure=struct2).astype(I.dtype)

    return I


def distance_transform_edt(I, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0

    I = ndimage.distance_transform_edt(I)

    return I


def median(I, median_size, num_iter, thresh=0.5):
    """Median filter, using ndimage implementation.

    Parameters
    ----------
    data : np.ndarray (D,H,W)
        Input image
    median_size : int
        Median size
    num_iter : int

    Returns
    -------
    np.ndarray (D,H,W)
        Median filtered image
    """
    I = I * 1.0

    for i in range(num_iter):
        I = ndimage.median_filter(I, median_size).astype(I.dtype)

    return I


def skeletonize(I, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    I = (I >= thresh) * 1.0
    skeleton = skeletonize_skimage(I)  # returns 0-255
    skeleton = (skeleton > 0) * 1.0

    return skeleton


def watershed(I, markers, thresh=0.5):
    I -= np.min(I)
    I = I / np.max(I)
    from skimage import img_as_ubyte

    I = img_as_ubyte(I)
    # xm, ym, zm = np.ogrid[0:I.shape[0]:10, 0:I.shape[1]:10, 0:I.shape[2]:10]
    markers = ((markers > 0) * 1.0).astype(np.int16)

    markers = ndimage.label(markers)[0]
    # markers[xm, ym, zm]= np.arange(xm.size*ym.size*zm.size).reshape((xm.size,ym.size, zm.size))
    ws = ndimage.watershed_ift(I, markers)
    return ws
