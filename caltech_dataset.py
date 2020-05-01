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

        self.split = split + ".txt"
        with open(split) as f:
            indexes = [line for line in f]
        categories = [set(index[:-15] for index in indexes)].sort()

        results = [__getitem__(index) for index in indexes]

        return results

        #- Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class)


    def __getitem__(self, index):
        int_index = int(index[-8:-4])
        label = categories.index(index[:-15])

        filepath = "101_ObjectCategories/" + index
        image = pil_loader(filepath)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = 42 # Provide a way to get the length (number of elements) of the dataset
        return length
