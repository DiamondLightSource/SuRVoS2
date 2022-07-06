from typing import List

import numpy as np
from skimage.measure import label
from torch.utils.data import Dataset


def sample_bounding_volume(img_volume, bvol, patch_size):
    z_st, x_st, y_st, z_end, x_end, y_end = bvol
    # print(img_volume.shape, z_st, x_st, y_st, z_end, x_end, y_end)
    if (
        z_st > 0
        and z_end < img_volume.shape[0]
        and x_st > 0
        and x_end < img_volume.shape[1]
        and y_st > 0
        and y_end < img_volume.shape[2]
    ):
        z_st = int(z_st)
        z_end = int(z_end)
        y_st = int(y_st)
        y_end = int(y_end)
        x_st = int(x_st)
        x_end = int(x_end)
        img = img_volume[z_st:z_end, y_st:y_end, x_st:x_end]
    else:
        img = np.zeros(patch_size)

    return img


class FilteredVolumeDataset(Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        bvols: List[np.ndarray],
        labels: List[List[int]],
        patch_size=(32, 32, 32),
        transform=None,
        plot_verbose=False,
    ):
        self.images, self.bvols, self.labels = images, bvols, labels
        self.transform = transform
        self.plot_verbose = plot_verbose
        self.patch_size = np.array(patch_size)
        print(f"Setting FilteredVolumeDataset patch size to {self.patch_size}")

    def __len__(self):
        return len(self.bvols)

    def __getitem__(self, idx):
        bvol = self.bvols[idx]
        label = self.labels[idx]
        samples = []

        for filtered_vol in self.images:
            # print(self.patch_size)
            sample = sample_bounding_volume(
                filtered_vol, bvol, patch_size=self.patch_size
            )
            samples.append(sample.astype(np.float32))

        if self.transform:
            sample = self.transform(sample)

        box_volume = sample.shape[0] * sample.shape[1] * sample.shape[2]

        target = {}
        target["boxes"] = bvol
        target["labels"] = label
        target["image_id"] = idx
        target["box_volume"] = box_volume

        return samples, target


class BoundingVolumeDataset(Dataset):
    def __init__(
        self,
        image: np.ndarray,
        bvols: List[np.ndarray],
        labels:List[List[int]],
        patch_size=(64, 64, 64),
        transform=None,
        plot_verbose=False,
    ):
        self.image, self.bvols, self.labels = image, bvols, labels
        self.transform = transform
        self.plot_verbose = plot_verbose
        self.patch_size = patch_size

    def __len__(self):
        return len(self.bvols)

    def __getitem__(self, idx):
        bvol = self.bvols[idx]
        label = self.labels[idx]
        image = sample_bounding_volume(self.image, bvol, patch_size=self.patch_size)

        if self.transform:
            image = self.transform(image)

        box_volume = image.shape[0] * image.shape[1] * image.shape[2]
        target = {}

        target["boxes"] = bvol
        target["labels"] = label
        target["image_id"] = idx
        target["box_volume"] = box_volume

        return image, target


# LabeledDataset
class SmallVolDataset(Dataset):
    def __init__(
        self, images, labels, class_names=None, slice_num=None, dim=3, transform=None
    ):

        self.input_images, self.target_labels = images, labels
        self.transform = transform
        self.class_names = class_names
        self.slice_num = slice_num
        self.dim = dim

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        image = self.input_images[idx]
        label = self.target_labels[idx]

        if self.dim == 2 and self.slice_num is not None:
            image = image[self.slice_num, :]
            image = np.stack((image, image, image)).T

        if self.transform:
            image = self.transform(image.T)
            label = self.transform(label.T)
        return image, label

