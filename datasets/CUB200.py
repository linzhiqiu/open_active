import os
import os.path
import sys

from torch.utils.data import Dataset
import torch
import pickle
import pandas as pd
import numpy as np

from datasets.image_utils import default_loader


class CUB200(Dataset):
    '''
    A Dataset for CUB200 

    Args:
        root (string): Root directory path. Should contain images/ and train_test_split.txt
                        from CUB200-2011
        loader (callable): A function to load a sample given its path. 
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    '''
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 is_valid_file=None, loader=default_loader):
        super(CUB200, self).__init__()

        self.transform = transform
        self.target_transform = target_transform
        self.root = root

        self.train = train

        classes = []
        with open(os.path.join(root, 'classes.txt'), 'r') as f:
            for i in f:
                cl = i.rstrip().split()[-1]
                classes.append(cl)
        
        self.classes = classes
        self.classes_to_idx = {i: ind for ind, i in enumerate(self.classes)}

        self.loader = loader

        self.samples, self.targets = self._make_dataset()

        return

    def _make_dataset(self):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes 
            are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        # image id -> filepath
        images = {}

        with open(os.path.join(self.root, 'images.txt')) as f:
            for i in f:
                l = i.rstrip().split()
                images[int(l[0])] = os.path.join('images', l[1])

        #  class_id to class_name 
        classes = {}
        with open(os.path.join(self.root, 'classes.txt')) as f:
            for i in f:
                l = i.rstrip().split()
                classes[int(l[0])] = l[1]

        # image_id -> class_name 
        all_labels = {}
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for i in f:
                l = i.rstrip().split()
                all_labels[int(l[0])] = classes[int(l[1])]

        samples = []
        targets = []

        if self.train:
            mode = 1
        else:
            mode = 0

        with open(os.path.join(self.root, 'train_test_split.txt')) as f:
            for i in f:
                l = i.rstrip().split()
                image_id = int(l[0])
                image_mode = int(l[1])

                if image_mode == mode:
                    samples.append(images[image_id])
                    targets.append(self.classes_to_idx[all_labels[image_id]])


        return samples, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        target = self.targets[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


# if __name__ == "__main__":
#     trainset = CUB200(
#         '/media/cheng/Samsung_T5/cub_200_2011/CUB_200_2011', train=True)
#     testset = CUB200(
#         '/media/cheng/Samsung_T5/cub_200_2011/CUB_200_2011', train=False)
#     print(trainset[0])
#     print(len(trainset))
#     print(len(testset))

#     from collections import Counter
#     s = (Counter(trainset.targets))
#     print(len(s))
#     print(s)
#     s = (Counter(testset.targets))
#     print(len(s))
#     print(s)
