from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import data

from batchgenerators.dataloading.data_loader import DataLoaderBase

from batchgenerators.dataloading import MultiThreadedAugmenter, SlimDataLoaderBase
import numpy as np

class DataLoader(SlimDataLoaderBase):

    def __init__(self, data, BATCH_SIZE=2, num_batches=None, seed=False, num_threads_in_mt=8):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_threads_in_mt)
        self.BATCH_SIZE = BATCH_SIZE
        #__init__(data, BATCH_SIZE, num_batches, seed) 
        # data is now stored in self._data.
    
    def generate_train_batch(self):
        # usually you would now select random instances of your data. We only have one therefore we skip this
        img = self._data
        
        # The camera image has only one channel. Our batch layout must be (b, c, x, y). Let's fix that
        img = np.tile(img[None, None], (self.BATCH_SIZE, 1, 1, 1))
        
        # now construct the dictionary and return it. np.float32 cast because most networks take float
        return {'data':img.astype(np.float32), 'some_other_key':'some other value'}



class DummyDL(SlimDataLoaderBase):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDL, self).__init__(None, None, num_threads_in_mt)
        self._data = list(range(100))
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[idx]
        else:
            self.reset()
            raise StopIteration


class DummyDLWithShuffle(DummyDL):
    def __init__(self, num_threads_in_mt=8):
        super(DummyDLWithShuffle, self).__init__(num_threads_in_mt)
        self.num_restarted = 0
        self.data_order = np.arange(len(self._data))

    def reset(self):
        super(DummyDLWithShuffle, self).reset()
        rs = np.random.RandomState(self.num_restarted)
        rs.shuffle(self.data_order)
        self.num_restarted = self.num_restarted + 1

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        idx = self.current_position
        if idx < len(self._data):
            self.current_position = idx + self.number_of_threads_in_multithreaded
            return self._data[self.data_order[idx]]
        else:
            self.reset()
            raise StopIteration

def plot_batch(batch):
    batch_size = batch['data'].shape[0]
    plt.figure(figsize=(16, 10))
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(batch['data'][i, 0], cmap="gray") # only grayscale image here
    plt.show()
def main():
    batchgen = DataLoader(data.camera(), 4, None, False)
    batch = next(batchgen)

    my_transforms = []
    brightness_transform = ContrastAugmentationTransform((0.3, 3.), preserve_range=True)
    my_transforms.append(brightness_transform)
    mirror_transform = MirrorTransform(axes=(0, 1))
    my_transforms.append(mirror_transform)

    all_transforms = Compose(my_transforms)
    multithreaded_generator = MultiThreadedAugmenter(batchgen, all_transforms, 1, 1, seeds=None)
    #mt = MultiThreadedAugmenter(dl, None, 3, 1, None)
    plot_batch(multithreaded_generator.next())
    multithreaded_generator._finish() # kill the workers
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()

