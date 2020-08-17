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
    num_init_classes: int = 100
    sample_per_class: int = 100
    num_open_classes: int = 100


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
        save_path (str): After generate the initial labeled set, save the dataset information at this location
    """
    download_path: str = "."
    save_path: str = "./dataset"


@dataclass
class DatasetParams():
    """All dataset parameters
    Args:
        data (str): Dataset name
        dataset_config (DatasetConfig): - Number of initial seen classes
                                        - Number of samples per seen classes
                                        - Number of open classes hold out of the unlabeled pool
        dataset_paths (DatasetPaths): Dataset save location
        data_rand_seed (int): If None, use first nth classes' first nth sample for initial labeled set.
                              Otherwise, use it as the random seed to select random initial labeled and open classes, and pick random samples per initial labeled class.
        use_val_set (bool): Whether to split a portion of train set as validation set. Defaults to False.
    """
    data: str
    dataset_config: DatasetConfig
    dataset_paths: DatasetPaths
    data_rand_seed: int = None
    use_val_set: bool = False


def _get_dataset_params(config, dataset_paths, use_val_set=False):
    """Return dataset configuration

    Args:
        config (ArgParse): Configuration from command line
        data_paths (DatasetPaths): Paths to where dataset will be saved after generated

    Returns:
        DatasetParams: Dataset Parameters
    """
    dataset_config = DATASET_CONFIG_DICT[config.data][config.data_config]
    return DatasetParams(
        config.data,
        dataset_config,
        dataset_paths,
        config.data_rand_seed,
        use_val_set=False
    )


@dataclass
class TrainsetInfo(object):
    """A data class holding all information for training set
    Args:
        train_samples (list[int]) : All sample indices (0-index based) representing all available samples in training set
        train_labels (list[int]) : Class indices for all samples in train set
        open_samples (set[int]): All sample indices (0-index based) representing training set samples belonging to hold out open class
        query_samples (set[int]): All sample indices (0-index based) representing the initial unlabeled pool
        val_samples (list[int]) : All sample indices (0-index based) for validation set (a subset of train_sample)
    """
    train_samples: list
    train_labels: list
    open_samples: set
    query_samples: set
    val_samples: list


@dataclass
class ClassInfo(object):
    """A data class holding all information for class split information
    Args:
        classes (list[int]) : All class indices in the dataset (0-index based) including both closed set and open set classes.
        open_classes (set[int]) : Hold out open class indices
        query_classes (set[int]) : All class indices in dataset other than hold out open classes
    """
    classes: list
    open_classes: set
    query_classes: set


@dataclass
class DatasetInfo():
    """The generated train+test dataset for open active learning
    Args:
        train_dataset (torch.utils.data.Dataset): Training set
        test_dataset (torch.utils.data.Dataset): Testing set
        class_info (ClassInfo): Class information
        trainset_info (TrainsetInfo): Training set information
        discovered_samples (list[int]) : Sample indices of initial labeled training set (including both train and val samples)
        discovered_classes (set[int]) : Initial discovered class indices
    """
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    class_info: ClassInfo
    trainset_info: TrainsetInfo
    discovered_samples: list
    discovered_classes: set


def _get_dataset_from_params(dataset_params: DatasetParams) -> DatasetInfo:
    train_dataset, test_dataset = _generate_dataset(
        dataset_params.data,
        dataset_params.dataset_paths.download_path
    )
    train_samples, train_labels, classes = _generate_dataset_info(
        dataset_params.data,
        train_dataset
    )
    # Load the dataset split information, or generate a new split and save it
    if os.path.exists(dataset_params.dataset_paths.save_path):
        print(
            f"Dataset file already generated at {dataset_params.dataset_paths.save_path}.")
        dataset_save_dict = torch.load(dataset_params.dataset_paths.save_path)
    else:
        print(
            f"Dataset file does not exist. Will be created at {dataset_params.dataset_paths.save_path}")
        # Split the training set using the config
        dataset_save_dict = _split_train_set(dataset_params.data,
                                             dataset_params.dataset_config,
                                             classes,
                                             train_labels,
                                             dataset_params.data_rand_seed,
                                             use_val_set=dataset_params.use_val_set)
        torch.save(dataset_save_dict, dataset_params.dataset_paths.save_path)

    open_classes = dataset_save_dict['open_classes']
    query_classes = classes.difference(open_classes)
    class_info = ClassInfo(
        classes,
        open_classes,
        query_classes
    )

    val_samples = dataset_save_dict['val_samples']
    open_samples_in_trainset = dataset_save_dict['open_samples']
    query_samples_in_trainset = train_samples.difference(open_samples_in_trainset)
    trainset_info = TrainsetInfo(
        train_samples,
        train_labels,
        open_samples_in_trainset,
        query_samples_in_trainset,
        val_samples
    )

    discovered_samples = dataset_save_dict['discovered_samples']
    discovered_classes = dataset_save_dict['discovered_classes']

    return DatasetInfo(
        train_dataset,
        test_dataset,
        class_info,
        trainset_info,
        discovered_samples,
        discovered_classes,
    )


def prepare_dataset_from_config(config, data_download_path, data_save_path):
    """Prepare a Dataset object from argparse configuration

    Args:
        config (ArgParse object): The argparse arguments

    Returns:
        DatasetInfo: All dataset information
    """
    dataset_paths = DatasetPaths(download_path=data_download_path,
                                 save_path=data_save_path)
    dataset_params = _get_dataset_params(config, dataset_paths)
    return _get_dataset_from_params(dataset_params)


def _split_train_set(data, dataset_config, classes, labels, data_rand_seed, use_val_set=False):
    """Split the training set, and prepare the following class variables

    Args:
        data (str) : Name of dataset
        dataset_config (DatasetConfig) : How to split the dataset. Same as the arg passed into constructor
        classes (lst of int) : The list of indices for all classes in dataset
        labels (lst of int) : The label of all samples in train set
        data_rand_seed (int or None) : The random seed. None if not randomized.

    Returns:
        dataset_info (dict): Contains the below key/values
        discovered_samples (list[int]): The initial labeled training set
        open_samples (set[int]): The samples in training set that belongs to open classes
        discovered_classes (set[int]): The initial discovered classes
        open_classes (set[int]): The hold out open classes
    """
    if data_rand_seed:
        random.seed(data_rand_seed)

    # Assert: num of class - num of open class - num of initial discovered classes >= 0
    assert len(classes) - dataset_config.num_open_classes - dataset_config.num_init_classes >= 0
    # class_to_sample_indices[class_index] = list of sample indices (list of int)
    class_to_sample_indices = {}
    for class_i in classes:
        class_to_sample_indices[class_i] = []
    for idx, label_i in enumerate(labels):
        class_to_sample_indices[label_i].append(idx)

    if data_rand_seed:
        discovered_classes = set(random.sample(
            classes, dataset_config.num_init_classes))
        remaining_classes = set(classes).difference(discovered_classes)
        open_classes = set(random.sample(remaining_classes, dataset_config.num_open_classes))
    else:
        discovered_classes = set(range(dataset_config.num_init_classes))
        if dataset_config.num_open_classes > 0:
            open_classes = set(range(len(classes))[-dataset_config.num_open_classes:])
        else:
            open_classes = set()

    discovered_samples = []
    if data_rand_seed:
        all_samples = []
        for class_i in discovered_classes:
            all_samples += class_to_sample_indices[class_i]

        total_sample_size = dataset_config.sample_per_class * len(discovered_classes)
        discovered_samples = list(random.sample(all_samples, total_sample_size))
    else:
        for class_i in discovered_classes:
            discovered_samples += class_to_sample_indices[class_i][:dataset_config.sample_per_class]

    open_samples = []
    for open_class_i in open_classes:
        open_samples += class_to_sample_indices[open_class_i]

    if use_val_set:
        print("Using a validation set")
        val_samples = []
        # VAL_RATIO of each discovered class, from smallest index
        discovered_class_to_sample_indices = {}
        for sample_i in discovered_samples:
            for class_i in discovered_classes:
                discovered_class_to_sample_indices[class_i] = []
            for idx in discovered_samples:
                label_i = labels[idx]
                discovered_class_to_sample_indices[label_i].append(idx)
        for class_i in discovered_classes:
            sorted_indices_class_i = sorted(
                discovered_class_to_sample_indices[class_i].copy())
            if len(sorted_indices_class_i) <= 1:
                continue
            class_i_size = math.ceil(len(sorted_indices_class_i) * VAL_RATIO)
            val_samples += sorted_indices_class_i[:class_i_size]
    else:
        val_samples = []

    assert len(open_samples) == len(set(open_samples))
    assert len(discovered_samples) == len(set(discovered_samples))
    assert len(val_samples) == len(set(val_samples))

    dataset_save_dict = {}  # Will be filled
    dataset_save_dict['discovered_samples'] = discovered_samples
    dataset_save_dict['val_samples'] = val_samples
    dataset_save_dict['open_samples'] = set(open_samples)
    dataset_save_dict['discovered_classes'] = discovered_classes
    dataset_save_dict['open_classes'] = open_classes
    return dataset_save_dict


def _generate_dataset_info(data, dataset):
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


def _generate_dataset(data, download_path):
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
