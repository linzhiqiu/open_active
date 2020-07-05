import torch
import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import DatasetFactory

from trainer import ActiveTrainer, TrainsetInfo
from active_config import get_active_learning_config
import utils
from utils import makedirs
import logging_helper
import json
import random

from utils import prepare_active_learning_dir_from_config, get_budget_list_from_config

def main():
    config = get_config()
    if config.use_random_seed:
        print("Using random random seed")
    else:
        print("Use random seed 1")
        torch.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # It contains all directory/save_paths that will be used
    budget_list = get_budget_list_from_config(config) 
    paths_dict = prepare_active_learning_dir_from_config(config, budget_list)
    
    dataset_factory = DatasetFactory(config.data,
                                     paths_dict['data_download_path'], # Where to download the images
                                     paths_dict['dataset_info_path'], # Where to save the dataset information
                                     config.active_init_mode,
                                     dataset_rand_seed=config.dataset_rand_seed)
    train_dataset, test_dataset = dataset_factory.get_dataset() # The pytorch datasets
    train_samples, train_labels = dataset_factory.get_train_set_info() # List of indices/labels
    classes, open_classes = dataset_factory.get_class_info() # Set of indices
    
    time_stamp = time.strftime("%Y-%m-%d %H:%M")

    # Begin from scratch
    discovered_samples, discovered_classes = dataset_factory.get_init_train_set() # Get initial training set, discovered classes
    val_samples = dataset_factory.get_val_samples() # Val samples is a subset of discovered_samples, and will be excluded in the network training.
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
    active_config = get_active_learning_config(config.data,
                                               config.training_method,
                                               config.active_train_mode)

    # Trainer is the main class for training and querying
    # It contains train() query() finetune() functions
    trainer = ActiveTrainer(
        config.training_method,
        config.active_train_mode,
        active_config,
        trainset_info,
        config.query_method,
        test_dataset,
        val_samples=val_samples,
        active_test_val_diff=config.active_test_val_diff
    )

    for i, b in enumerate(budget_list): 
        # b is the budget for independent mode, need to adjust it for sequential mode
        if config.active_query_scheme == 'sequential':
            if i > 0:
                budget = b - budget_list[i-1]
            else:
                budget = b
        else:
            budget = b
        
        # if config.training_method == 'deep_metric':
        #     print("Skip softmax network train phase, directly go to deep metric learning for train phase")
        #     pretrained_softmax_path = os.path.join(config.deep_metric_softmax_pretrained_folder, config.data, config.init_mode+".pt")
        #     trainer.trainer_machine.load_backbone(pretrained_softmax_path) 
        new_discovered_samples, new_discovered_classes = trainer.query(budget,
                                                                       discovered_samples,
                                                                       discovered_classes,
                                                                       paths_dict['active_query_results'][b],
                                                                       verbose=config.verbose)

        if config.active_query_scheme == 'sequential':
            print("using sequential mode, we updated the discovered samples")
            discovered_samples, discovered_classes = new_discovered_samples, new_discovered_classes
        else:
            print("using independent mode, we do not update the initial labeled pool.")

        trainer.train(new_discovered_samples,
                      new_discovered_classes,
                      paths_dict['active_ckpt_results'][b],
                      verbose=config.verbose)

        closed_set_test_acc = trainer.eval_closed_set(new_discovered_classes,
                                                    #   test_dataset,
                                                      paths_dict['active_test_results'][b],
                                                      verbose=config.verbose)

        # trainer.eval_open_set(discovered_samples, discovered_classes, test_dataset, verbose=config.verbose)
    

if __name__ == '__main__':
    main()
