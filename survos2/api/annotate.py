import numpy as np
from loguru import logger

_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def get_order(viewer_order):
    # ABC (210)
    # CBA
    # (210)

    # ABC (201)
    # CAB
    # (120)

    # ABC(120)
    # BCA
    # (201)

    # ABC (102)
    # BAC
    # (102)

    # ABC (021)
    # ACB
    # 021

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
        if not bb:
            # print("No mask")
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
        dataset[:] = data & 15
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


# def annotate_regions4(dataset, region, r=None, label=0, parent_mask=None):
#     if label >= 16 or label < 0 or type(label) != int:
#         raise ValueError("Label has to be in bounds [0, 15]")
#     if r is None or len(r) == 0:
#         return

#     mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

#     rmax = np.max(r)
#     modified = dataset.get_attr("modified")
#     logger.debug(f"Annotating {dataset.total_chunks}")

#     for i in range(dataset.total_chunks):
#         idx = dataset.unravel_chunk_index(i)
#         chunk_slices = dataset.global_chunk_bounds(
#             idx
#         )  # get slice() for the current chunk
#         reg_chunk = region[chunk_slices]  # take a chunk of the superregion volume

#         total = max(rmax + 1, np.max(reg_chunk) + 1)
#         mask = np.zeros(total, np.bool)
#         mask[r] = True  # set to true the chunks that have been painted
#         mask = mask[reg_chunk]  #

#         if parent_mask is not None:
#             parent_mask_chunk = parent_mask[chunk_slices]
#             mask = mask * parent_mask_chunk

#         if not np.any(mask):
#             modified[i] = (modified[i] << 1) & mbit
#             continue

#         data_chunk = dataset[chunk_slices]
#         data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)
#         data_chunk[mask] = (data_chunk[mask] & _MaskPrev) | label

#         dataset[chunk_slices] = data_chunk
#         modified[i] = (modified[i] << 1) & mbit | 1

#     dataset.set_attr("modified", modified)


# def annotate_regions3(dataset, region, r=None, label=0, parent_mask=None):
#     if label >= 16 or label < 0 or type(label) != int:
#         raise ValueError("Label has to be in bounds [0, 15]")
#     if r is None or len(r) == 0:
#         return

#     mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

#     rmax = np.max(r)
#     modified = dataset.get_attr("modified")
#     logger.debug(f"Annotating {dataset.total_chunks}")

#     for i in range(dataset.total_chunks):
#         idx = dataset.unravel_chunk_index(i)
#         chunk_slices = dataset.global_chunk_bounds(
#             idx
#         )  # get slice() for the current chunk
#         reg_chunk = region[chunk_slices]  # take a chunk of the superregion volume

#         #         total = max(rmax + 1, np.max(reg_chunk) + 1)
#         #         mask = np.zeros(total, bool)
#         #         print(mask.shape)
#         #         mask[r] = True # set to true the chunks that have been painted
#         #         print(mask.shape)
#         #         mask_ = mask
#         #         mask = mask[reg_chunk] # only select pixels in current chunk

#         mask = np.zeros_like(reg_chunk)

#         for r_idx in r:
#             mask += (reg_chunk == r_idx) * 1
#         mask = mask > 0

#         if parent_mask is not None:
#             parent_mask_chunk = parent_mask[chunk_slices]
#             mask = mask * parent_mask_chunk

#         if not np.any(mask):
#             modified[i] = (modified[i] << 1) & mbit
#             continue

#         data_chunk = dataset[chunk_slices]
#         data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)
#         data_chunk[mask] = (data_chunk[mask] & _MaskPrev) | label

#         dataset[chunk_slices] = data_chunk
#         modified[i] = (modified[i] << 1) & mbit | 1

#     dataset.set_attr("modified", modified)
#     return mask, reg_chunk


# def annotate_voxels2(dataset, slice_idx=0, yy=None, xx=None, label=0, parent_mask=None):
#     mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

#     def remove_masked_points(xs, ys, bg_mask):
#         pts_slice = np.zeros_like(bg_mask[slice_idx])
#         print(pts_slice.shape)
#         for i in range(len(xs)):
#             pts_slice[xs[i], ys[i]] = 1
#         pts_slice = pts_slice * bg_mask
#         xs, ys = np.where(pts_slice == 1)
#         return xs, ys

#     def tobounds(slices):
#         zs, ys, xs = slices
#         if slice_idx < zs.start or slice_idx >= zs.stop:
#             return False
#         yp, xp = [], []
#         for y, x in zip(yy, xx):
#             if ys.start <= y < ys.stop and xs.start <= x < xs.stop:
#                 yp.append(y - ys.start)
#                 xp.append(x - xs.start)
#         if len(yp) > 0:
#             return slice_idx - zs.start, yp, xp
#         return False

#     if label >= 16 or label < 0 or type(label) != int:
#         raise ValueError("Label has to be in bounds [0, 15]")

#     modified = dataset.get_attr("modified")
#     if (
#         len(modified) == 1
#     ):  # resetting chunk based modified parameter for voxel annotation
#         modified = [0] * dataset.total_chunks

#     for i in range(dataset.total_chunks):
#         idx = dataset.unravel_chunk_index(i)
#         chunk_slices = dataset.global_chunk_bounds(idx)
#         result = tobounds(chunk_slices)

#         if result is False:
#             modified[i] = (modified[i] << 1) & mbit
#             continue

#         idx, yp, xp = result
#         data_chunk = dataset[chunk_slices]
#         data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)

#         mask = np.zeros_like(data_chunk[0, :])
#         data_slice = data_chunk[idx]
#         data_slice[yp, xp] = (data_slice[yp, xp] & _MaskPrev) | label

#         if parent_mask is not None:
#             parent_mask_chunk = parent_mask[chunk_slices]
#             parent_mask_slice = parent_mask_chunk[idx]
#             data_slice = data_slice * parent_mask_slice

#         # data_chunk[idx, yp, xp] = (data_chunk[idx, yp, xp] & _MaskPrev) | label

#         data_chunk[idx, :] = data_slice

#         dataset[chunk_slices] = data_chunk
#         modified[i] = (modified[i] << 1) & mbit | 1

#     dataset.set_attr("modified", modified)
