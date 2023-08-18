import numpy as np
from loguru import logger

def remove_masked_entities(bg_mask, entities):
    pts_vol = np.zeros_like(bg_mask)
    logger.debug(f"Masking on mask of shape {pts_vol.shape}")
    entities = entities.astype(np.uint32)
    for pt in entities:
        if (
            (pt[0] > 0)
            & (pt[0] < pts_vol.shape[0])
            & (pt[1] > 0)
            & (pt[1] < pts_vol.shape[1])
            & (pt[2] > 0)
            & (pt[2] < pts_vol.shape[2])
        ):
            pts_vol[pt[0], pt[1], pt[2]] = 1
    pts_vol = pts_vol * (1.0 - bg_mask)
    zs, xs, ys = np.where(pts_vol == 1)
    masked_entities = []
    for i in range(len(zs)):
        pt = [zs[i], ys[i], xs[i], 6]
        masked_entities.append(pt)
    return np.array(masked_entities)