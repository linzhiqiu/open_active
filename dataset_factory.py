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



# Below are gadgets never used. Please ignore or delete.
def get_imagenet12(config, transforms_dict):
    train_path = os.path.join(config.data_path, config.dataset.lower(), 'train')
    test_path = os.path.join(config.data_path, config.dataset.lower(), 'val')
    train_set = datasets.ImageFolder(train_path, transforms_dict["train"])
    test_set = datasets.ImageFolder(test_path, transforms_dict["test"])
    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'val': None,
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'train_size': len(train_set),
        'val_size': 0,
        'test_size': len(test_set),
        'num_classes': 1000
    }

def get_dataset_from_dir(dir_name, config, transforms_dict):
    train_set = datasets.ImageFolder(dir_name, transforms_dict["train"])
    val_set = datasets.ImageFolder(dir_name, transforms_dict["test"])
    train_sampler = None
    val_sampler = None
    train_size = len(train_set)
    val_size = 0
    if config.val_ratio != 0:
        indices = list(range(train_size))
        val_size = int(config.val_ratio*train_size)
        np.random.shuffle(indices)
        val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
        train_size = train_size-val_size
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

    loaders = {
        'train' : DataLoader(
                      train_set,
                      batch_size=config.batch_size,
                      sampler=train_sampler,
                      num_workers=config.num_workers,
                      pin_memory=True
                  ),
        'val': DataLoader(
            val_set,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            val_set,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        ), # TODO: Fix
        'train_size': train_size,
        'val_size': val_size,
        'test_size': val_size, # TODO: Fix
        'num_classes' : get_class_num(config)
    }
    config.num_classes = loaders['num_classes']
    return loaders

def get_tiny_imagnet(config, transforms_dict):
    train_path = os.path.join("tiny-imagenet-200", "train")
    return get_dataset_from_dir(train_path, config, transforms_dict)

def get_splitted_tiny_imgnet(config, transforms_dict):
    FIRST_100_CLASSES_DIR = os.path.join("tiny-imagenet-200", "train_first_100")
    SECOND_100_CLASSES_DIR = os.path.join("tiny-imagenet-200", "train_second_100")
    return [get_dataset_from_dir(FIRST_100_CLASSES_DIR, config, transforms_dict), 
            get_dataset_from_dir(SECOND_100_CLASSES_DIR, config, transforms_dict)]

def get_cifar10(config, transforms_dict):
    # load dataset
    ds = getattr(datasets, config.dataset)
    path = os.path.join(config.data_path, config.dataset.lower())

    transform_train = transforms_dict["train"]
    transform_test = transforms_dict["test"]
    train_set = ds(path, train=True, download=True, transform=transform_train)
    val_set = ds(path, train=True, download=True, transform=transform_test)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    train_sampler = None
    val_sampler = None
    train_size = len(train_set)
    val_size = 0
    if config.val_ratio != 0:
        indices = list(range(train_size))
        val_size = int(config.val_ratio*train_size)
        print("train set size {}, validation set size {}".format(train_size-val_size, val_size))
        np.random.shuffle(indices)
        val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
        train_size = train_size-val_size
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
    loaders = {
        'train': DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_set,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        ),
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(test_set),
        'num_classes': get_class_num(config)
    }
    # TODO: Move this to somewhere else
    config.num_classes = loaders['num_classes']
    return loaders

def get_dataset_with_class_indices(ds, class_indices, shuffle=False):
    # Only work for CIFAR10
    if ds.train:
        data_name = "train_data"
        labels_name = 'train_labels'
    else:
        data_name = "test_data"
        labels_name = 'test_labels'

    if not hasattr(ds, data_name) or not hasattr(ds, labels_name):
        raise AttributeError(f"No attribute {data_name} or {labels_name} is founded.")

    new_ds = copy.deepcopy(ds)
    data = getattr(new_ds, data_name)
    labels = getattr(new_ds, labels_name)
    class_example_indices = {i : np.where(np.array(labels) == i)[0] for i in class_indices}
    class_examples = [data[class_example_indices[i]] for i in class_indices]
    # labels = 1
    new_data = np.concatenate(class_examples, axis=0)
    new_labels = [[idx] * len(class_arr) for idx, class_arr in zip(class_indices, class_examples)]
    new_labels = [idx for sublist in new_labels for idx in sublist]
    setattr(new_ds, data_name, new_data)
    setattr(new_ds, labels_name, new_labels)
    return new_ds

def get_loaders_with_num_examples(config, loaders, num_example_per_class):
    val_ratio = config.val_ratio
    batch_size = config.batch_size

    if config.dataset == "CIFAR10":
        # Only work for CIFAR10


        data_name = "train_data"
        labels_name = 'train_labels'

        if not hasattr(loaders['train'].dataset, data_name) or not hasattr(loaders['train'].dataset, labels_name):
            raise AttributeError(f"No attribute {data_name} or {labels_name} is founded.")

        new_loaders = copy.deepcopy(loaders)
        data = getattr(new_loaders['train'].dataset, data_name)
        labels = getattr(new_loaders['train'].dataset, labels_name)

        class_indices = np.unique(np.array(labels))
        class_example_indices = {i : (np.where(np.array(labels) == i)[0])[:num_example_per_class] for i in class_indices}
        class_examples = [data[class_example_indices[i]] for i in class_indices]
        # class_examples = [examples[:num_example_per_class] for examples in class_examples]
        new_data = np.concatenate(class_examples, axis=0)
        new_labels = [[idx] * len(class_arr) for idx, class_arr in zip(class_indices, class_examples)]
        new_labels = [idx for sublist in new_labels for idx in sublist]
        setattr(new_loaders['train'].dataset, data_name, new_data)
        setattr(new_loaders['train'].dataset, labels_name, new_labels)
        

        if val_ratio != 0:
            train_size = len(new_loaders['train'].dataset)
            indices = list(range(train_size))
            val_size = int(val_ratio*train_size)
            print("train set size {}, validation set size {}".format(train_size-val_size, val_size))
            np.random.shuffle(indices)
            val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
            train_size = train_size-val_size
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
        new_loaders['train'] = DataLoader(
                                   new_loaders['train'].dataset,
                                   batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=0,
                                   pin_memory=True
                               )
        new_loaders['val'] = DataLoader(
                                 new_loaders['train'].dataset,
                                 batch_size=batch_size,
                                 shuffle=(val_sampler is None),
                                 sampler=val_sampler,
                                 num_workers=0,
                                 pin_memory=True
                             )
        return new_loaders
    elif config.dataset == "TINY-IMAGENET":
        train_set = loaders['train'].dataset
        num_classes = train_set.classes.__len__()
        old_num_per_class = train_set.samples.__len__() / num_classes
        indexes = []
        for idx in range(num_classes):
            start = int(idx*old_num_per_class)
            end = start + num_example_per_class
            indexes += list(range(start, end))
        indexes = np.array(indexes)
        if val_ratio != 0:
            train_size = len(indexes)
            val_size = int(val_ratio*train_size)
            print("train set size {}, validation set size {}".format(train_size-val_size, val_size))
            np.random.shuffle(indexes)
            val_idx, train_idx = indexes[train_size-val_size:], indexes[:train_size-val_size]
            train_size = train_size-val_size
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
        else:
            train_sampler = SubsetRandomSampler(indexes)
            val_sampler = SubsetRandomSampler(indexes)
        new_loaders = {}
        new_loaders['train'] = DataLoader(
                                   train_set,
                                   batch_size=batch_size,
                                   sampler=train_sampler,
                                   num_workers=0,
                                   pin_memory=True
                               )
        new_loaders['val'] = DataLoader(
                                 train_set,
                                 batch_size=batch_size,
                                 sampler=val_sampler,
                                 num_workers=0,
                                 pin_memory=True
                             )
        new_loaders['test'] = loaders['test']
        return new_loaders


def get_splitted_cifar_10(config, transforms_dict):
    ds = getattr(datasets, config.dataset)
    path = os.path.join(config.data_path, config.dataset.lower())

    transform_train = transforms_dict["train"]
    transform_test = transforms_dict["test"]
    train_set = ds(path, train=True, download=True, transform=transform_train)
    val_set = ds(path, train=True, download=True, transform=transform_test)
    test_set = ds(path, train=False, download=True, transform=transform_test)

    # Determine the classes to split
    num_classes = get_class_num(config)
    class_indices = np.arange(num_classes)
    if config.split_method in ['split_half', "split_random_half"]:
        if config.split_method == "split_random_half":
            class_indices = np.random.shuffle(class_indices)
        first_class_indices = class_indices[:int(num_classes/2)]
        second_class_indices = class_indices[int(num_classes/2):]
    elif config.split_method in ["split_ratio", 'split_random_ratio']:
        assert config.split_ratio <= 1 and config.split_ratio >= 0
        index_to_split = num_classes * config.split_ratio
        print(f"Split into 1: {index_to_split} classes and 2: {num_classes-index_to_split} classes.")
        if config.split_method == 'split_random_ratio':
            class_indices = np.random.shuffle(class_indices)
        first_class_indices = class_indices[:index_to_split]
        second_class_indices = class_indices[index_to_split:]
    else:
        raise ValueError("Splitting method not supported")

    print(f"Splitting into: \n {str(first_class_indices)} and \n {str(second_class_indices)}")
    # Generate two datasets
    train_sets = [get_dataset_with_class_indices(train_set, first_class_indices),
                  get_dataset_with_class_indices(train_set, second_class_indices)]
    val_sets = [get_dataset_with_class_indices(val_set, first_class_indices),
                get_dataset_with_class_indices(val_set, second_class_indices)]                  
    test_sets = [get_dataset_with_class_indices(test_set, first_class_indices),
                 get_dataset_with_class_indices(test_set, second_class_indices)]

    loaders = []
    for i, (train_set, val_set, test_set) in enumerate(zip(train_sets, val_sets, test_sets)):
        if config.val_ratio != 0:
            train_size = len(train_set)
            indices = list(range(train_size))
            val_size = int(config.val_ratio*train_size)
            print("{}: train set size {}, validation set size {}".format(i, train_size-val_size, val_size))
            np.random.shuffle(indices)
            val_idx, train_idx = indices[train_size-val_size:], indices[:train_size-val_size]
            train_size = train_size-val_size
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
        loaders += [{
            'train': DataLoader(
                train_set,
                batch_size=config.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=config.num_workers,
                pin_memory=True
            ),
            'val': torch.utils.data.DataLoader(
                val_set,
                batch_size=config.batch_size,
                sampler=val_sampler,
                num_workers=config.num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            ),
            'train_size': train_size,
            'val_size': val_size,
            'test_size': len(test_set),
            'num_classes': get_class_num(config),
            'train_sampler' : train_sampler,
            'val_sampler' : val_sampler
        }]
    # TODO: Move this to somewhere else
    config.num_classes = [loaders[0]['num_classes'], loaders[1]['num_classes']]
    return loaders

if __name__ == "__main__":
    from config import get_config
    from transform import get_transform
    config, _ = get_config()
    transforms_dict = get_transform(config)
    # get_imagenet12(config, transforms_dict)
    get_splitted_cifar_10(config, transforms_dict)
