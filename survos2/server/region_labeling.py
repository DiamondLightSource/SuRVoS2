import numpy as np
from loguru import logger

from numba import jit

# original rlabels always was hanging, replaced with quick numba fix


# y: labels (annotation vol)
# R: superregions
# nr: number of superregions
# ny: number of labels
@jit
def simple_rlabels(y, R, ny, nr, min_ratio):
    N = R.shape[0]

    sizes = np.zeros(nr, dtype=np.uint32)
    counts = np.zeros((nr, ny), dtype=np.uint32)
    out = np.zeros(nr, dtype=np.uint32)

    for i in range(N):
        label = y[i]
        region = R[i]
        sizes[region] += 1

        if label > 0:
            counts[region, label] += 1

    for i in range(nr):
        cmax = 0
        smin = sizes[i] * min_ratio

        for j in range(1, ny):
            curr = counts[i, j]

            if curr > cmax and curr >= smin:
                cmax = curr
                out[i] = j

    return out


def rlabels(
    y: np.uint16,
    R: np.uint32,
    nr: int = None,
    ny: int = None,
    norm: bool = None,
    min_ratio: int = 0,
):
    # WARNING: silently fails if types are not exactly correct
    y = np.array(y)
    R = np.array(R)

    nr = nr or R.max() + 1
    ny = ny or y.max() + 1

    try:
        features = simple_rlabels(y.ravel(), R.ravel(), ny, nr, min_ratio)

    except Exception as err:
        logger.error(f"simple_rlabels exception: {err}")

    # features = mappings._rlabels(y.ravel(), R.ravel(), ny, nr, min_ratio)

    return features
    # return normalize(features).astype(np.uint16, norm=norm)
