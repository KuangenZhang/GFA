from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, data, label,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
        #print(self.data.shape)
        #print(self.labels.shape)
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.float32(img)
        return img, target
    def __len__(self):
        return len(self.data)
