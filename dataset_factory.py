# Prepare dataset/loader
import os
import torch
import torchvision
import torchvision.datasets as datasets
import transform

from torch.utils.data import DataLoader

from torch.utils.data import SubsetRandomSampler
import numpy as np
import copy
import random
import pickle

from global_setting import SUPPORTED_DATASETS, DATASET_CONFIG_DICT

class DatasetFactory(object):
    def __init__(self, data, download_path, save_path, init_mode, dataset_rand_seed=None):
        """Constructor of a facotry of pytorch dataset
            Args:
                data (str) : Name of datasets
                download_path (str) : Path where the dataset will be downloaded
                save_path (str) : Path will the dataset split information will be downloaded
                init_mode (str) : How the dataset will be splitted, including information about
                                      - Number of initial seen classes
                                      - Number of samples per seen classes
                                      - Number of open classes hold out of the unlabeled pool
                dataset_rand_seed (int or None) : If None, use first nth class's first nth samples.
                                                  Otherwise, use the random seed to select random classes + random samples per class.
        """
        super(DatasetFactory, self).__init__()
        self.data = data
        self.download_path = download_path
        self.save_path = save_path
        self.init_mode = init_mode
        self.dataset_rand_seed = dataset_rand_seed

        # Sanity Check
        assert data in SUPPORTED_DATASETS

        self.train_dataset, self.test_dataset = generate_dataset(self.data, self.download_path)
        self.train_samples, self.train_labels, self.classes = generate_dataset_info(self.data, 
                                                                                    self.train_dataset)

        # Load the dataset split information, or generate a new split and save it
        dataset_info_dir = os.path.join(self.save_path,
                                        self.data,
                                        self.init_mode)
        if not os.path.exists(dataset_info_dir):
            input(f"{dataset_info_dir} does not exists. Press anything to create it >> ")
            os.makedirs(dataset_info_dir)
        
        dataset_info_path = os.path.join(dataset_info_dir, f"seed_{self.dataset_rand_seed}.pickle")
        
        if os.path.exists(dataset_info_path):
            print(f"Dataset file already generated at {dataset_info_path}.")
            self.dataset_info_dict = pickle.load(open(dataset_info_path, 'rb'))
        else:
            input(f"Dataset file does not exist. Will be created at {dataset_info_path}? >> ")
            # Split the training set using the config
            self.dataset_info_dict = self._split_train_set(self.data,
                                                           self.init_mode,
                                                           self.classes,
                                                           self.train_labels,
                                                           self.dataset_rand_seed)
            pickle.dump(self.dataset_info_dict, open(dataset_info_path, 'wb+'))

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
                discovered_samples (list of sample indices) : The initial labeled training set
                discovered_classes (set) : The initial discovered classes
        '''
        return self.dataset_info_dict['discovered_samples'], self.dataset_info_dict['discovered_classes']
    
    def get_open_samples_in_trainset(self):
        '''Return the open samples/class indices in training set
            Returns:
                open_samples (set of sample indices) : The samples in training set that belongs to open classes
        '''
        return self.dataset_info_dict['open_samples']
    
    def _split_train_set(self, data, init_mode, classes, labels, dataset_rand_seed):
        """Split the training set, and prepare the following class variables
            Returns:
                dataset_info (dict) : Contains the below key/values
                    discovered_samples -> (list of sample indices) : The initial labeled training set
                    open_samples -> (set of sample indices) : The samples in training set that belongs to open classes
                    discovered_classes -> (set) : The initial discovered classes
                    open_classes -> (set) : The hold out open classes

            Args:
                data (str) : Name of dataset
                init_mode (str) : How to split the dataset. Same as the arg passed into constructor
                classes (lst of int) : The list of indices for all classes in dataset
                labels (lst of int) : The label of all samples in train set
                dataset_rand_seed (int or None) : The random seed. None if not randomized.
        """
        dataset_info = {} # Will be filled
        if dataset_rand_seed:
            random.seed(dataset_rand_seed)

        init_conf = DATASET_CONFIG_DICT[data][init_mode]
        # Assert: num of class - num of open class - num of initial discovered classes >= 0
        assert len(classes) - init_conf['num_open_classes'] - init_conf['num_init_classes'] >= 0
        if data not in ['CIFAR100']:
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
                open_classes = set(range(len(classes))[-init_conf['num_open_classes']:])
            
            discovered_samples = []
            for class_i in discovered_classes:
                if dataset_rand_seed:
                    discovered_samples += random.sample(class_to_indices[class_i], init_conf['sample_per_class'])
                else:
                    discovered_samples += class_to_indices[class_i][:init_conf['sample_per_class']]

            open_samples = []
            for open_class_i in open_classes:
                open_samples += class_to_indices[open_class_i]
            
            assert len(open_samples) == len(set(open_samples))
            assert len(discovered_samples) == len(set(discovered_samples))
            
            dataset_info['discovered_samples'] = discovered_samples
            dataset_info['open_samples'] = set(open_samples)
            dataset_info['discovered_classes'] = discovered_classes
            dataset_info['open_classes'] = open_classes
        return dataset_info

def generate_dataset_info(data, dataset):
    if data in ['CIFAR10', 'CIFAR100']:
        samples = set(range(len(dataset)))
        labels = getattr(dataset, "targets")
        classes = set(labels)
        return samples, labels, classes
    else:
        raise NotImplementedError()

def generate_dataset(data, download_path):
    transforms_dict = transform.get_transform_dict(data)
    ds_class = getattr(datasets, data)
    train_set = ds_class(root=download_path, 
                         train=True, 
                         download=True, 
                         transform=transforms_dict['train'])
    test_set = ds_class(root=download_path,
                        train=False,
                        download=True,
                        transform=transforms_dict['test'])
    return train_set, test_set
