import torch
import numpy as np

class HardRandomBorderMasking(object):
    """

     """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)


        top_y = np.random.randint(h)

        y1 = np.random.randint(h // 2)
        y2 = np.random.randint(h // 2 ) + (h // 2)
        
        # horizontal bands
        
        x1=0
        x2=w
        
        mask[0: y1, x1: x2] = 0.        
        mask[y2:h, x1: x2] = 0.

        x1 = np.random.randint(w  // 2)
        x2 = np.random.randint(w  // 2) + (w // 2)
       

        # vertical bands
        y1 = 0
        y2 = h
        
        mask[y1: y2, 0: x1] = 0.
        mask[y1: y2, x2: w] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        
        img = img * mask

        return img