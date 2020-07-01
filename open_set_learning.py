import torch
import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import DatasetFactory

from trainer import OpenTrainer, TrainsetInfo
from open_config import get_open_set_learning_config
import utils
from utils import makedirs
import logging_helper
import json
import random

from utils import prepare_open_set_learning_dir_from_config

def main():
    config = get_config()
    if config.use_random_seed:
        print("Using random random seed")
    else:
        print("Use random seed 1")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    # It contains all directory/save_paths that will be used
    paths_dict = prepare_open_set_learning_dir_from_config(config)
    
    dataset_factory = DatasetFactory(config.data,
                                     paths_dict['data_download_path'], # Where to download the images
                                     paths_dict['dataset_info_path'], # Where to save the dataset information
                                     config.open_init_mode,
                                     dataset_rand_seed=config.dataset_rand_seed)
    train_dataset, test_dataset = dataset_factory.get_dataset() # The pytorch datasets
    train_samples, train_labels = dataset_factory.get_train_set_info() # List of indices/labels
    classes, open_classes = dataset_factory.get_class_info() # Set of indices
    
    time_stamp = time.strftime("%Y-%m-%d %H:%M")

    # Begin from scratch
    discovered_samples, discovered_classes = dataset_factory.get_init_train_set() # Get initial training set, discovered classes
    open_samples = dataset_factory.get_open_samples_in_trainset() # Get open samples and classes in train set
    
    # The train set details
    trainset_info = TrainsetInfo(train_dataset,
                                 train_samples,
                                 open_samples,
                                 train_labels,
                                 classes,
                                 open_classes)
    if not os.path.exists(paths_dict['trainset_info_path']):
        torch.save(trainset_info, paths_dict['trainset_info_path'])

    # The training details including arch, lr, batch size..
    open_set_config = get_open_set_learning_config(config.data,
                                                   config.training_method,
                                                   config.open_set_train_mode)

    trainer = OpenTrainer(
        config.training_method,
        config.open_set_train_mode,
        open_set_config,
        trainset_info,
        config.open_set_val_mode,
    )

    trainer.train(discovered_samples,
                  discovered_classes,
                  paths_dict['open_set_ckpt_results'],
                  verbose=config.verbose)

    closed_set_test_acc = trainer.eval_closed_set(discovered_classes,
                                                  test_dataset,
                                                  paths_dict['open_set_test_result'],
                                                  verbose=config.verbose)
    
    trainer.eval_open_set(discovered_samples, discovered_classes, test_dataset, verbose=config.verbose)
    

if __name__ == '__main__':
    main()
