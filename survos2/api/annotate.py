from matplotlib.pyplot import box
import numpy as np
from loguru import logger
from survos2.entity.anno.masks import ellipsoidal_mask

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
    three_dim = False,
    brush_size= 10,
    centre_point = (8,8,8)
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

    ds_t = (ds_t & _MaskCopy) | (ds_t << _MaskSize)

    box_half_dim = int(brush_size // 2)

    print(f"Slice idx {slice_idx}")
    if three_dim:
        if parent_mask is not None:
            parent_mask_t = np.transpose(parent_mask, viewer_order)
        mask = np.zeros_like(ds_t)
        logger.info(f"Drawing voxels in 3d at {centre_point}")
        ellipse_size = brush_size
        ellipse_mask = ellipsoidal_mask(ellipse_size,ellipse_size,ellipse_size, radius=box_half_dim, center=(ellipse_size//2, ellipse_size//2, ellipse_size//2 )).astype(np.bool_)   
        print(ellipse_mask.shape)
        for i in range(len(yy)):
            bbsz, bbfz, bbsy,bbfy, bbsx, bbfx = slice_idx-box_half_dim, slice_idx+box_half_dim, xx[i]-box_half_dim, xx[i]+box_half_dim, yy[i]-box_half_dim, yy[i]+box_half_dim     
            bbsz, bbfz, bbsy,bbfy, bbsx, bbfx = np.max((0,bbsz)), np.min((ds_t.shape[0], bbfz)),np.max((0,bbsy)), np.min((ds_t.shape[1], bbfy)), np.max((0,bbsx)), np.min((ds_t.shape[2], bbfx))
            d,w,h = bbfz-bbsz, bbfy-bbsy, bbfx-bbsx
            if ((d != brush_size) | (w != brush_size) | (h != brush_size)):
                ellipse_mask = ellipse_mask[ellipse_mask.shape[0]-d:(ellipse_mask.shape[0]-d)+d, ellipse_mask.shape[1]-w:(ellipse_mask.shape[1]-w)+w,ellipse_mask.shape[2]-h:(ellipse_mask.shape[2]-h)+h]
            
            mask[bbsz:bbfz,bbsy:bbfy,bbsx:bbfx] = ellipse_mask
            if parent_mask is not None:
                mask = mask * parent_mask_t
            mask = mask > 0
            #ds_t[bbsz:bbfz,bbsy:bbfy,bbsx:bbfx][ellipse_mask] = ((ds_t[bbsz:bbfz,bbsy:bbfy,bbsx:bbfx][ellipse_mask] & _MaskPrev) | label)
            ds_t[mask] = (ds_t[mask] & _MaskPrev) | label


        print(ds_t[bbsz:bbfz,bbsy:bbfy,bbsx:bbfx].shape, ellipse_mask.shape)
    
        
    else:
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
