import numpy as np
from loguru import logger

_MaskSize = 4  # 4 bits per history label
_MaskCopy = 15  # 0000 1111
_MaskPrev = 240  # 1111 0000


def annotate_voxels(dataset, slice_idx=0, yy=None, xx=None, label=0, parent_mask=None):
    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1
    
    def remove_masked_points(xs,ys, bg_mask):
        pts_slice = np.zeros_like(bg_mask[slice_idx])
        print(pts_slice.shape)
        for i in range(len(xs)):
            pts_slice[xs[i], ys[i]] = 1
        pts_slice = pts_slice * bg_mask
        xs, ys = np.where(pts_slice == 1)
        return xs, ys

    def tobounds(slices):
        zs, ys, xs = slices
        if slice_idx < zs.start or slice_idx >= zs.stop:
            return False
        yp, xp = [], []
        for y, x in zip(yy, xx):
            if ys.start <= y < ys.stop and xs.start <= x < xs.stop:
                yp.append(y - ys.start)
                xp.append(x - xs.start)
        if len(yp) > 0:
            return slice_idx - zs.start, yp, xp
        return False

    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")

    modified = dataset.get_attr("modified")

    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx)
        result = tobounds(chunk_slices)

        if result is False:
            modified[i] = (modified[i] << 1) & mbit
            continue

        idx, yp, xp = result
        print(f"Len points {len(xp)}") 
        data_chunk = dataset[chunk_slices]
        data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)

        mask = np.zeros_like(data_chunk[0,:])
        print(mask.shape)

        data_slice=data_chunk[idx]
        data_slice[yp, xp] = (data_slice[yp, xp] & _MaskPrev) | label

        if parent_mask is not None:
            parent_mask_chunk = parent_mask[chunk_slices]
            parent_mask_slice = parent_mask_chunk[idx]
            data_slice = data_slice * parent_mask_slice

        #data_chunk[idx, yp, xp] = (data_chunk[idx, yp, xp] & _MaskPrev) | label

        data_chunk[idx,:] = data_slice

        dataset[chunk_slices] = data_chunk
        modified[i] = (modified[i] << 1) & mbit | 1

    dataset.set_attr("modified", modified)


def annotate_regions(dataset, region, r=None, label=0, parent_mask=None):
    if label >= 16 or label < 0 or type(label) != int:
        raise ValueError("Label has to be in bounds [0, 15]")
    if r is None or len(r) == 0:
        return

    mbit = 2 ** (np.dtype(dataset.dtype).itemsize * 8 // _MaskSize) - 1

    rmax = np.max(r)
    modified = dataset.get_attr("modified")
    logger.debug(f"Annotating {dataset.total_chunks}")

    for i in range(dataset.total_chunks):
        idx = dataset.unravel_chunk_index(i)
        chunk_slices = dataset.global_chunk_bounds(idx) # get slice() for the current chunk
        reg_chunk = region[chunk_slices] # take a chunk of the superregion volume
    
        total = max(rmax + 1, np.max(reg_chunk) + 1)
        mask = np.zeros(total, np.bool)
        mask[r] = True # set to true the chunks that have been painted
        mask = mask[reg_chunk] #

        if parent_mask is not None:
            parent_mask_chunk = parent_mask[chunk_slices]
            mask = mask * parent_mask_chunk

        if not np.any(mask):
            modified[i] = (modified[i] << 1) & mbit
            continue

        data_chunk = dataset[chunk_slices]
        data_chunk = (data_chunk & _MaskCopy) | (data_chunk << _MaskSize)
        data_chunk[mask] = (data_chunk[mask] & _MaskPrev) | label
        

        dataset[chunk_slices] = data_chunk
        modified[i] = (modified[i] << 1) & mbit | 1

    dataset.set_attr("modified", modified)


def undo_annotation(dataset):
    modified = dataset.get_attr("modified")
    print(modified)

    for i in range(dataset.total_chunks):

        if modified[i] & 1 == 0:
            continue

        print("Undoing annotation")

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
