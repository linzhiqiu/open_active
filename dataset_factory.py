"""Prepare dataset and dataloader for open set / active learning.
"""
import os
import random
import math

import torch
import torchvision
import torchvision.datasets
import transform
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass

import datasets


# If config.use_val_set: 20% of each initial discovered class will be used for validation.
VAL_RATIO = 0.2

@dataclass
class DatasetConfig():
    """Dataset configuration including class split and initial labeled samples
    Args:
        num_init_classes (int): Number of initial labeled (discovered) classes
        sample_per_class (int): Number of sample per discovered class
        num_open_classes (int): Number of classes hold out as open set
    """    
    num_init_classes: int=100
    sample_per_class: int=100
    num_open_classes: int=100


DATASET_CONFIG_DICT = {
    'CIFAR100': {
        # The setup in ICRA deep metric learning paper. 80 closed classes, 20 open classes.
        'regular': DatasetConfig(num_init_classes=40, sample_per_class=250, num_open_classes=20),
        # Use 1/5 of discovered classes, but same number of samples per discovered class
        'fewer_class': DatasetConfig(num_init_classes=8, sample_per_class=250, num_open_classes=20),
        # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
        'fewer_sample': DatasetConfig(num_init_classes=40, sample_per_class=50, num_open_classes=20),
        # For active learning closed set experiments
        'active': DatasetConfig(num_init_classes=100, sample_per_class=30, num_open_classes=0),
        # For open set learning with half of the classes being open set
        'open_set': DatasetConfig(num_init_classes=50, sample_per_class=500, num_open_classes=50),
    },
    'CIFAR10': {
        # The setup in ICRA deep metric learning paper. 8 closed classes, 2 open classes.
        'regular': DatasetConfig(num_init_classes=4, sample_per_class=500, num_open_classes=2),
        # Use 1/5 of discovered classes, but same number of samples per discovered class
        'fewer_class': DatasetConfig(num_init_classes=2, sample_per_class=500, num_open_classes=2),
        # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
        'fewer_sample': DatasetConfig(num_init_classes=4, sample_per_class=250, num_open_classes=2),
        # For active learning closed set experiments
        'active': DatasetConfig(num_init_classes=10, sample_per_class=100, num_open_classes=0),
        # For open set learning with half of the classes being open set
        'open_set': DatasetConfig(num_init_classes=5, sample_per_class=5000, num_open_classes=5),
    },

    'CUB200': {
        # The setup in ICRA deep metric learning paper.
        'regular': DatasetConfig(num_init_classes=90, sample_per_class=15, num_open_classes=20),
        # Use 1/5 of discovered classes, but same number of samples per discovered class
        'fewer_class': DatasetConfig(num_init_classes=18, sample_per_class=15, num_open_classes=20),
        # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
        'fewer_sample': DatasetConfig(num_init_classes=90, sample_per_class=3, num_open_classes=20),
    },

    'Cars': {
        # The setup in ICRA deep metric learning paper.
        'regular': DatasetConfig(num_init_classes=80, sample_per_class=20, num_open_classes=36),
        # Use 1/5 of discovered classes, but same number of samples per discovered class
        'fewer_class': DatasetConfig(num_init_classes=16, sample_per_class=20, num_open_classes=36),
        # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
        'fewer_sample': DatasetConfig(num_init_classes=80, sample_per_class=4, num_open_classes=36),
    },
}

@dataclass
class DatasetPaths():
    """Dataset download and saved paths
    Args:
        download_path (str): Where the raw image dataset will be saved
        parsed_path (str): After generate the initial labeled set, save the dataset information at this location
    """
    download_path: str="."
    parsed_path: str="./dataset"


@dataclass
class DatasetParams():
    """All dataset parameters
    Args:
        dataset_config (DatasetConfig)
        dataset_paths (DatasetPaths)
        dataset_rand_seed (int): If None, use first nth classes' first nth sample for initial labeled set.
                                 Otherwise, use it as the random seed to select random initial labeled and open classes, and pick random samples per initial labeled class.
    """
    dataset_config: DatasetConfig
    dataset_paths: DatasetPaths
    dataset_rand_seed: int=None


def get_dataset_params(config):
    """Return dataset configuration

    Args:
        config (ArgParse): Configuration from command line

    Returns:
        DatasetParams: Dataset Parameters
    """
    return DATASET_CONFIG_DICT[data][data_config]

class Dataset():
    pass
       

class DatasetFactory(object):
    def __init__(self, data, download_path, dataset_info_path, data_config, dataset_rand_seed=None, use_val_set=False):
        """Constructor of a factory of pytorch datasets

        Args:
            data (str): Name of dataset
            download_path (str): Path where the dataset will be downloaded
            dataset_info_path (str): Path where the dataset split information will be downloaded
            data_config (str): How the dataset will be splitted, including information about
                             - Number of initial seen classes
                             - Number of samples per seen classes
                             - Number of open classes hold out of the unlabeled pool
            dataset_rand_seed (int): If None, use first nth classes' first nth sample for initial labeled set.
                                     Otherwise, use it as the random seed to select random initial labeled and open classes, and pick random samples per initial labeled class.
            use_val_set (bool, optional): Whether to include a validation set. Defaults to False.
        """        
        super(DatasetFactory, self).__init__()
        self.data = data
        self.download_path = download_path
        self.dataset_info_path = dataset_info_path
        self.data_config = data_config
        self.dataset_rand_seed = dataset_rand_seed
        self.use_val_set = use_val_set

        self.train_dataset, self.test_dataset = generate_dataset(self.data, self.download_path)
        self.train_samples, self.train_labels, self.classes = generate_dataset_info(self.data, 
                                                                                    self.train_dataset)

        # Load the dataset split information, or generate a new split and save it
        if os.path.exists(self.dataset_info_path):
            print(f"Dataset file already generated at {self.dataset_info_path}.")
            self.dataset_info_dict = torch.load(self.dataset_info_path)
        else:
            print(f"Dataset file does not exist. Will be created at {self.dataset_info_path}")
            # Split the training set using the config
            self.dataset_info_dict = self._split_train_set(self.data,
                                                           self.data_config,
                                                           self.classes,
                                                           self.train_labels,
                                                           self.dataset_rand_seed,
                                                           use_val_set=self.use_val_set)
            torch.save(self.dataset_info_dict, self.dataset_info_path)

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
        return self.classes, self.dataset_info_dict['open_classes']

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
                discovered_samples (list of sample indices) : The initial labeled training set (including both train and val samples)
                discovered_classes (set) : The initial discovered classes
        '''
        return self.dataset_info_dict['discovered_samples'], self.dataset_info_dict['discovered_classes']
    
    def get_val_samples(self):
        '''Return the validation set
            Returns:
                val_samples (list of sample indices) : The validation set indices
        '''
        return self.dataset_info_dict['val_samples']
    
    def get_open_samples_in_trainset(self):
        '''Return the open samples/class indices in training set
            Returns:
                open_samples (set of sample indices) : The samples in training set that belongs to open classes
        '''
        return self.dataset_info_dict['open_samples']
    
    def _split_train_set(self, data, data_config, classes, labels, dataset_rand_seed, use_val_set=False):
        """Split the training set, and prepare the following class variables
            Returns:
                dataset_info (dict) : Contains the below key/values
                    discovered_samples -> (list of sample indices) : The initial labeled training set
                    open_samples -> (set of sample indices) : The samples in training set that belongs to open classes
                    discovered_classes -> (set) : The initial discovered classes
                    open_classes -> (set) : The hold out open classes

            Args:
                data (str) : Name of dataset
                data_config (str) : How to split the dataset. Same as the arg passed into constructor
                classes (lst of int) : The list of indices for all classes in dataset
                labels (lst of int) : The label of all samples in train set
                dataset_rand_seed (int or None) : The random seed. None if not randomized.
        """
        dataset_info = {} # Will be filled
        if dataset_rand_seed:
            random.seed(dataset_rand_seed)

        init_conf = DATASET_CONFIG_DICT[data][data_config]
        # Assert: num of class - num of open class - num of initial discovered classes >= 0
        assert len(classes) - init_conf['num_open_classes'] - init_conf['num_init_classes'] >= 0
        if data not in ['CIFAR100', 'CUB200', 'Cars', 'CIFAR10']:
            raise NotImplementedError()
        else:
            # class_to_indices[class_index] = list of sample indices (list of int)
            class_to_indices = {}
            for class_i in classes:
                class_to_indices[class_i] = []
            for idx, label_i in enumerate(labels):
                class_to_indices[label_i].append(idx)

            if dataset_rand_seed:
                discovered_classes = set(random.sample(classes, init_conf['num_init_classes']))
                remaining_classes = set(classes).difference(discovered_classes)
                open_classes = set(random.sample(remaining_classes, init_conf['num_open_classes']))
            else:
                discovered_classes = set(range(init_conf['num_init_classes']))
                if init_conf['num_open_classes'] > 0:
                    open_classes = set(range(len(classes))[-init_conf['num_open_classes']:])
                else:
                    open_classes = set()
            
            discovered_samples = []
            if dataset_rand_seed:
                all_samples = []
                for class_i in discovered_classes:
                    all_samples += class_to_indices[class_i]
                
                total_sample_size = init_conf['sample_per_class'] * len(discovered_classes)
                discovered_samples = list(random.sample(all_samples, total_sample_size))
            else:
                for class_i in discovered_classes:
                    discovered_samples += class_to_indices[class_i][:init_conf['sample_per_class']]

            open_samples = []
            for open_class_i in open_classes:
                open_samples += class_to_indices[open_class_i]

            # If you need a validation set, then use the below code
            # if use_val_set:
            #     print("Using a validation set")
            #     val_samples = []
            #     # VAL_RATIO of each discovered class, from smallest index
            #     discovered_class_to_indices = {}
            #     for sample_i in discovered_samples:
            #         for class_i in discovered_classes:
            #             discovered_class_to_indices[class_i] = []
            #         for idx in discovered_samples:
            #             label_i = labels[idx]
            #             discovered_class_to_indices[label_i].append(idx)
            #     for class_i in discovered_classes:
            #         sorted_indices_class_i = sorted(discovered_class_to_indices[class_i].copy())
            #         if len(sorted_indices_class_i) <= 1:
            #             continue
            #         class_i_size = math.ceil(len(sorted_indices_class_i) * VAL_RATIO)
            #         val_samples += sorted_indices_class_i[:class_i_size]
            # else:
            #     val_samples = []
            val_samples = []

            assert len(open_samples) == len(set(open_samples))
            assert len(discovered_samples) == len(set(discovered_samples))
            assert len(val_samples) == len(set(val_samples))
            
            dataset_info['discovered_samples'] = discovered_samples
            dataset_info['val_samples'] = val_samples
            dataset_info['open_samples'] = set(open_samples)
            dataset_info['discovered_classes'] = discovered_classes
            dataset_info['open_classes'] = open_classes
        return dataset_info

def generate_dataset_info(data, dataset):
    if data in ['CIFAR10', 'CIFAR100']:
        samples = set(range(len(dataset)))
        labels = getattr(dataset, "targets")
        classes = set(labels)
    elif data in ['CUB200', 'Cars']:
        samples = set(range(len(dataset)))
        labels = getattr(dataset, "targets")
        classes = set(range(len(dataset.classes)))
    else:
        raise NotImplementedError()
    return samples, labels, classes

def generate_dataset(data, download_path):
    transforms_dict = transform.get_transform_dict(data)
    if data in ['CIFAR10', 'CIFAR100']:
        ds_class = getattr(torchvision.datasets, data)
        train_set = ds_class(root=download_path, 
                             train=True, 
                             download=True, 
                             transform=transforms_dict['train'])
        test_set = ds_class(root=download_path,
                            train=False,
                            download=True,
                            transform=transforms_dict['test'])
    elif data in ['CUB200']:
        ds_class = getattr(datasets, data)
        train_set = ds_class(root=download_path,
                             train=True,
                             transform=transforms_dict['train'])
        test_set = ds_class(root=download_path,
                            train=False,
                            transform=transforms_dict['test'])
    elif data in ['Cars']:
        ds_class = getattr(datasets, data)
        train_set = ds_class(root=download_path,
                             train=True,
                             transform=transforms_dict['train'])
        test_set = ds_class(root=download_path,
                            train=False,
                            transform=transforms_dict['test'])
    else:
        raise NotImplementedError()
    return train_set, test_set
