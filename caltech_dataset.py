from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = "Caltech101/" + split + ".txt"
        with open(self.split) as f:
            self.indexes = [line.rstrip() for line in f if line.startswith("BAC")==False]
        self.categories = list(set([ind[:-15] for ind in self.indexes]))
        self.categories.sort(key=lambda v: v.upper())
        self.labels = [self.categories.index(ind[:-15]) for ind in self.indexes]
        self.images = [pil_loader("Caltech101/101_ObjectCategories/"+ind) for ind in self.indexes]

        #- Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)


    def __getitem__(self, index):
            
        #label = categories.index(index[:-15])

        #image = pil_loader(index)
        image, label = self.images[i], self.categories.index(self.indexes[i][:-15])

        # Applies preprocessing when accessing the image
        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images) # Provide a way to get the length (number of elements) of the dataset
        return length
