from matplotlib.pyplot import box
import numpy as np
from loguru import logger

_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def get_order(viewer_order):
    """Calculate the new order of the axes. Follows napari viewer order.

    Args:
        viewer_order (_type_): Current order of the axes.

    Returns:
        _type_: New order of the axes.
    """
    viewer_order_str = "".join(map(str, viewer_order))
    if viewer_order_str == "201":
        new_order = np.roll(viewer_order, 1)
    elif viewer_order_str == "120":
        new_order = np.roll(viewer_order, -1)
    elif (
        viewer_order_str == "012"
        or viewer_order_str == "021"
        or viewer_order_str == "102"
        or viewer_order_str == "210"
    ):
        new_order = viewer_order

    return new_order


def annotate_voxels(
    dataset,
    slice_idx=0,
    yy=None,
    xx=None,
    label=0,
    parent_mask=None,
    viewer_order=(0, 1, 2),
):
    """Annotate individual voxels in a dataset.

    Args:
        dataset (Dataset): Dataset object.
        slice_idx (int, optional): Which slice to annotate. Defaults to 0.
        yy (list, optional): List of y-coordinates to annotate. Defaults to None.
        xx (list, optional): List of x-coordinates to annotate. Defaults to None.
        label (int, optional): Label value to set. Defaults to 0.
        parent_mask (np.ndarray, optional): Mask image. Defaults to None.
        viewer_order (tuple, optional): Axes order. Defaults to (0, 1, 2).
    Raises:
        ValueError: Label index must be less than 16 and greater than 0.
    """
    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1
    modified = dataset.get_attr("modified")
    modified = [0]

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")

    ds = dataset[:]
    # original_shape = ds.shape
    if len(viewer_order) == 3:
        ds_t = np.transpose(ds, viewer_order)
    else:
        ds_t = ds

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)
    #    box_half_dim = int(brush_size // 2)

    data_slice = ds_t[slice_idx, :]
    data_slice[yy, xx] = (data_slice[yy, xx] & _MaskPrev) | label

    if parent_mask is not None:
        parent_mask_t = np.transpose(parent_mask, viewer_order)
        mask = parent_mask_t
        mask = mask > 0
        mask = mask[slice_idx, :]
        data_slice = data_slice * mask

    ds_t[slice_idx, :] = data_slice

    if len(viewer_order) == 3:
        new_order = get_order(viewer_order)
        ds_o = np.transpose(ds_t, new_order)
    else:
        ds_o = ds_t
    dataset[:] = ds_o


def annotate_regions(
    dataset, region, r=None, label=0, parent_mask=None, bb=None, viewer_order=(0, 1, 2)
):
    """Annotate superregions in a dataset.

    Args:
        dataset (Dataset): Dataset object
        region (np.ndarray): Region image
        r (int, optional): Region index. Defaults to None.
        label (int, optional): Label value. Defaults to 0.
        parent_mask (np.ndarray, optional): Mask image. Defaults to None.
        bb (list, optional): Bounding box coordinates. Defaults to None.
        viewer_order (tuple, optional): Viewer axes order. Defaults to (0, 1, 2).

    Raises:
        ValueError: If the labels are greater than 15 or less than 0.

    Returns:
        _type_: Modified label image.
    """
    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    if r is None or len(r) == 0:
        return

    reg = region[:]
    ds = dataset[:]

    viewer_order_str = "".join(map(str, viewer_order))
    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        ds_t = np.transpose(ds, viewer_order)
        reg_t = np.transpose(reg, viewer_order)
    else:
        ds_t = ds
        reg_t = reg

    mask = np.zeros_like(reg_t)

    try:
        if not bb or bb[0] == -1:
            bb = [0, 0, 0, ds_t.shape[0], ds_t.shape[1], ds_t.shape[2]]
        for r_idx in r:
            mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] += (
                reg_t[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] == r_idx
            )
    except Exception as e:
        logger.debug(f"annotate_regions exception {e}")

    mask = (mask > 0) * 1.0

    if parent_mask is not None:
        parent_mask_t = np.transpose(parent_mask, viewer_order)
        mask = mask * parent_mask_t

    mask = mask > 0

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)
    ds_t[mask] = (ds_t[mask] & _MaskPrev) | label

    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        new_order = get_order(viewer_order)
        ds_o = np.transpose(ds_t, new_order)  # .reshape(original_shape)
    else:
        ds_o = ds_t
    return ds_o


def undo_annotation(dataset):
    """Undo the last annotation

    Args:
        dataset (Dataset): Dataset object
    """
    modified = dataset.get_attr("modified")

    #    logger.debug("Undoing annotation")

    if len(modified) == 1:
        data = dataset[:]
        # data = (data << _MaskSize) | (data >> _MaskSize)
        data = data >> _MaskSize
        dataset[:] = data  # & 15
    else:  # annotate voxels
        for i in range(dataset.total_chunks):
            if modified[i] & 1 == 0:
                continue

            idx = dataset.unravel_chunk_index(i)
            chunk_slices = dataset.global_chunk_bounds(idx)
            data = dataset[chunk_slices]

            # data = (data << _MaskSize) | (data >> _MaskSize)
            data = data >> _MaskSize
            dataset[chunk_slices] = data

    dataset.set_attr("modified", modified)


def erase_label(dataset, label=0):
    """Erase label with particular value from the dataset.

    Args:
        dataset (Dataset): Dataset object
        label (int, optional): Label value to erase. Defaults to 0.

    Raises:
        ValueError: Modified dataset.
    """

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    lmask = _MaskCopy - label
    # remove label from all history
    nbit = np.dtype(dataset.dtype).itemsize * 8
    btop = 2**nbit - 1

    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        data_chunk = dataset[chunk_slices]
        modified = False

        for s in range(nbit // _MaskSize):
            shift = s * _MaskSize
            cmask = _MaskCopy << shift
            rmask = data_chunk & cmask == label << shift  # Check presence of label
            if np.any(rmask):  # Delete label
                modified = True
                hmask = btop - (label << shift)
                data_chunk[rmask] &= hmask
        if modified:
            dataset[chunk_slices] = data_chunk
