import torch
import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import DatasetFactory

from trainer import Trainer, TrainsetInfo
from trainer_config import get_trainer_config
import utils
from utils import makedirs
import logging_helper
import json
import random

from utils import prepare_save_dir_from_config
import global_setting

def main():
    config = get_config()
    if config.use_random_seed:
        print("Using random random seed")
    else:
        print("PyTorch use random seed 1")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    # It contains all directory/save_paths that will be used
    paths_dict = prepare_save_dir_from_config(config)
    
    dataset_factory = DatasetFactory(config.data,
                                     paths_dict['data_download_path'], # Where to download the images
                                     paths_dict['dataset_info_path'], # Where to save the dataset information
                                     config.init_mode,
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
    trainer_config = get_trainer_config(config.data,
                                        config.training_method,
                                        config.train_mode)

    # Trainer is the main class for training and querying
    # It contains train() query() finetune() functions
    trainer = Trainer(
        config.training_method,
        config.train_mode,
        trainer_config,
        trainset_info,
        config.query_method,
        config.budget,
        global_setting.OPEN_SET_METHOD_DICT[config.training_method],
        paths_dict=paths_dict,
    )

    if config.training_method == 'deep_metric':
        print("Skip softmax network train phase, directly go to deep metric learning for train phase")
        pretrained_softmax_path = os.path.join(config.deep_metric_softmax_pretrained_folder, config.data, config.init_mode+".pt")
        trainer.trainer_machine.load_backbone(pretrained_softmax_path) 
    trainer.train(discovered_samples, discovered_classes, verbose=config.verbose)

    discovered_samples, discovered_classes = trainer.query(discovered_samples, discovered_classes, verbose=config.verbose)
    
    trainer.finetune(discovered_samples, discovered_classes, verbose=config.verbose)
    

    closed_set_test_acc = trainer.eval_closed_set(discovered_classes, test_dataset, verbose=config.verbose)
    
    trainer.eval_open_set(discovered_samples, discovered_classes, test_dataset, verbose=config.verbose)
    

if __name__ == '__main__':
    main()
