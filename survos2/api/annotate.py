import numpy as np
from loguru import logger

_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def get_order(viewer_order):
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
    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1
    modified = dataset.get_attr("modified")
    modified = [0]

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")

    ds = dataset[:]
    original_shape = ds.shape
    if len(viewer_order) == 3:
        ds_t = np.transpose(ds, viewer_order)
        logger.info(
            f"viewer_order {viewer_order} Dataset after first transpose: {ds_t.shape}"
        )
    else:
        ds_t = ds

    # mask = np.zeros_like(ds)
    # mask = (mask > 0) * 1.0
    # if parent_mask is not None:
    #     # print(f"Using parent mask of shape: {parent_mask.shape}")
    #     mask = mask * parent_mask
    # mask = mask > 0

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)
    data_slice = ds_t[slice_idx, :]
    data_slice[yy, xx] = (data_slice[yy, xx] & _MaskPrev) | label

    if parent_mask is not None:
        parent_mask_t = np.transpose(parent_mask, viewer_order)
        print(f"Using parent mask of shape: {parent_mask.shape}")
        mask = parent_mask_t
        mask = mask > 0
        mask = mask[slice_idx, :]
        data_slice = data_slice * mask

    ds_t[slice_idx, :] = data_slice

    if len(viewer_order) == 3:
        new_order = get_order(viewer_order)
        logger.info(
            f"new order {new_order} Dataset before second transpose: {ds_t.shape}"
        )
        ds_o = np.transpose(ds_t, new_order)  # .reshape(original_shape)
    else:
        ds_o = ds_t
    # ds = np.transpose(ds, np.roll(viewer_order,1)).reshape(original_shape)
    logger.info(f"Dataset after second transpose: {ds_o.shape}")
    dataset[:] = ds_o


def annotate_regions(
    dataset, region, r=None, label=0, parent_mask=None, bb=None, viewer_order=(0, 1, 2)
):
    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    if r is None or len(r) == 0:
        return

    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1
    rmax = np.max(r)
    modified = dataset.get_attr("modified")
    modified = [0]

    reg = region[:]
    ds = dataset[:]

    viewer_order_str = "".join(map(str, viewer_order))
    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        ds_t = np.transpose(ds, viewer_order)
        reg_t = np.transpose(reg, viewer_order)
    else:
        ds_t = ds
        reg_t = reg

    # ds = np.transpose(ds, viewer_order)
    logger.debug(f"After viewer_order transform {ds.shape}")
    # print(f"Dataset shape {ds.shape}")
    mask = np.zeros_like(reg_t)

    # print(f"BB: {bb}")
    try:
        if not bb or bb[0] == -1:
            print("No bb")
            bb = [0, 0, 0, ds_t.shape[0], ds_t.shape[1], ds_t.shape[2]]
        # else:
        # print(f"Masking using bb: {bb}")
        for r_idx in r:
            mask[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] += (
                reg_t[bb[0] : bb[3], bb[1] : bb[4], bb[2] : bb[5]] == r_idx
            )
    except Exception as e:
        print(f"annotate_regions exception {e}")

    mask = (mask > 0) * 1.0

    if parent_mask is not None:
        parent_mask_t = np.transpose(parent_mask, viewer_order)
        # print(f"Using parent mask of shape: {parent_mask.shape}")
        mask = mask * parent_mask_t

    mask = mask > 0
    # print(mask.shape)
    # if not np.any(mask):
    #    modified[i] = (modified[i] << 1) & mbit

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)
    ds_t[mask] = (ds_t[mask] & _MaskPrev) | label
    logger.debug(f"Returning annotated region ds {ds.shape}")

    if viewer_order_str != "012" and len(viewer_order_str) == 3:
        new_order = get_order(viewer_order)
        logger.info(
            f"new order {new_order} Dataset before second transpose: {ds_t.shape}"
        )
        ds_o = np.transpose(ds_t, new_order)  # .reshape(original_shape)
        logger.info(f"Dataset after second transpose: {ds_o.shape}")
    else:
        ds_o = ds_t
    return ds_o

    # dataset[:] = ds_o


def undo_annotation(dataset):
    modified = dataset.get_attr("modified")
    print(modified)
    print("Undoing annotation")

    if len(modified) == 1:
        data = dataset[:]
        # data = (data << _MaskSize) | (data >> _MaskSize)
        data = data >> _MaskSize
        dataset[:] = data #& 15
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
    print(f"Finished undo")


def erase_label(dataset, label=0):

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    lmask = _MaskCopy - label
    # remove label from all history
    nbit = np.dtype(dataset.dtype).itemsize * 8
    btop = 2 ** nbit - 1

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
