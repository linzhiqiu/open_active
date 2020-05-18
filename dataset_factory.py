# Prepare dataset/loader
import os
import torch
import torchvision
import torchvision.datasets as datasets
from transform import get_transform_dict

from torch.utils.data import DataLoader

from torch.utils.data import SubsetRandomSampler
import numpy as np
import copy
import random

from global_setting import SUPPORTED_DATASETS, INIT_TRAIN_SET_CONFIG

class DatasetFactory(object):
    def __init__(self, data, data_path, init_mode):
        """Constructor of a facotry of pytorch dataset
            Args:
                data (str) : Name of datasets
                data_path (str) : Path where the dataset will be saved
                init_mode (str) : How the dataset will be splitted, including information about
                                      - Number of initial seen classes
                                      - Number of samples per seen classes
                                      - Number of open classes hold out of the unlabeled pool
        """
        super(DatasetFactory, self).__init__()
        self.data = data
        self.data_path = data_path
        self.init_mode = init_mode

        # Sanity Check
        assert data in SUPPORTED_DATASETS

        self.train_dataset, self.test_dataset = generate_dataset(self.data, self.data_path)
        self.train_samples, self.train_labels, self.classes = generate_dataset_info(self.data, 
                                                                                    self.train_dataset, 
                                                                                    mode='train')

        # Split the training set using the config
        self._split_train_set(self.data,
                              self.init_mode,
                              self.classes,
                              self.train_labels)

    def get_dataset(self):
        """Returns training set and test set
            Returns:
                train_dataset (torch.data.Dataset) : Training set
                test_dataset (torch.data.Dataset) : Test set
        """
        return self.train_dataset, self.test_dataset

    def get_class_info(self):
        """Return the class information, including
            Returns:
                classes (list of int) : List of class indices in the dataset (0-index based)
                                        including indices in unlabeled pool + open classes.
                open_classes (set) : The hold out open classes
        """
        return self.classes, self.open_classes

    def get_train_set_info(self):
        """Returns all information about training set
            Returns:
                train_samples (list of int) : List of sample indices (0-index based) 
                                              representing all available samples in trainset
                train_labels (list of int) : List of class indices in train set (0-index based)
        """
        return self.train_samples, self.train_labels
    
    def get_init_train_set(self):
        '''Return the initial training set
            Returns:
                discovered_samples (list of sample indices) : The initial labeled training set
                discovered_classes (set) : The initial discovered classes
        '''
        return self.discovered_samples, self.discovered_classes
    
    def get_open_samples_in_trainset(self):
        '''Return the open samples/class indices in training set
            Returns:
                open_samples (set of sample indices) : The samples in training set that belongs to open classes
        '''
        return self.open_samples
    
    def _split_train_set(self, data, init_mode, classes, labels):
        """Split the training set, and prepare the following class variables 
                self.discovered_samples (list of sample indices) : The initial labeled training set
                self.open_samples (set of sample indices) : The samples in training set that belongs to open classes
                self.discovered_classes (set) : The initial discovered classes
                self.open_classes (set) : The hold out open classes

            Args:
                data (str) : Name of dataset
                init_mode (str) : How to split the dataset. Same as the arg passed into constructor
                classes (lst of int) : The list of indices for all classes in dataset
                labels (lst of int) : The label of all samples in train set
        """
        init_conf = INIT_TRAIN_SET_CONFIG[data][init_mode]
        # Assert: num of class - num of open class - num of initial discovered classes >= 0
        assert len(classes) - init_conf['num_open_classes'] - init_conf['num_init_classes'] >= 0
        if data in ['CIFAR100', 'CIFAR10']:
            # class_to_indices[class_index] = list of sample indices (list of int)
            class_to_indices = {}
            for class_i in classes:
                class_to_indices[class_i] = []
            for idx, label_i in enumerate(labels):
                class_to_indices[label_i].append(idx)
                
            discovered_samples = []
            open_samples = []
            if init_conf['use_random_classes']:
                raise NotImplementedError()
            elif 'use_random_samples' in init_conf and init_conf['use_random_samples']:
                for class_i in range(init_conf['num_init_classes']):
                    discovered_samples += class_to_indices[class_i]

                random.shuffle(discovered_samples)
                total_size = init_conf['num_init_classes'] * init_conf['sample_per_class']
                discovered_samples = discovered_samples[:total_size]
            else:
                for class_i in range(init_conf['num_init_classes']):
                    discovered_samples += class_to_indices[class_i][:init_conf['sample_per_class']]

            # Caveat: The open classes are simply the last n-th classes.
            if init_conf['num_open_classes'] > 0:
                open_classes = range(len(classes))[-init_conf['num_open_classes']:]
            else:
                open_classes = []
            for open_class_i in open_classes:
                open_samples += class_to_indices[open_class_i]
            assert len(open_samples) == len(set(open_samples))
            
            self.discovered_samples = discovered_samples
            self.open_samples = set(open_samples)
            self.discovered_classes = set(range(init_conf['num_init_classes']))
            self.open_classes = set(open_classes)
        else:
            raise NotImplementedError()



def generate_dataset_info(data, dataset, mode=None):
    if data in ['CIFAR10', 'CIFAR100']:
        assert mode in ['train', 'test']
        samples = set(range(len(dataset)))
        labels = getattr(dataset, f"{mode}_labels")
        classes = set(labels)
        return samples, labels, classes
    else:
        raise NotImplementedError()

def generate_dataset(data, data_path):
    transforms_dict = get_transform_dict(data)
    ds_class = getattr(datasets, data)
    train_set = ds_class(root=data_path, 
                         train=True, 
                         download=True, 
                         transform=transforms_dict['train'])
    test_set = ds_class(root=data_path,
                        train=False,
                        download=True,
                        transform=transforms_dict['test'])
    return train_set, test_set
