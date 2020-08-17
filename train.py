import torch
import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import prepare_dataset_from_config

from trainer import Trainer, TrainsetInfo
from trainer_config import get_trainer_config
import utils
from utils import makedirs
import json
import random

from utils import prepare_save_dir_from_config
import global_setting

def main():
    config = get_config()

    if config.use_random_seed:
        print("Using random random seed")
    else:
        print("PyTorch will use random seed 1")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    
    # It contains all directory that will be used for saving datasets/checkpoints
    paths_dict = prepare_save_dir_from_config(config)
    
    dataset_info = prepare_dataset_from_config(
        config,
        paths_dict['data_download_path'],
        paths_dict['data_save_path']
    )
    
    time_stamp = time.strftime("%Y-%m-%d %H:%M")

    # Save the train set details for later analysis
    if not os.path.exists(paths_dict['trainset_info_path']):
        torch.save(dataset_info.trainset_info, paths_dict['trainset_info_path'])

    # The training details including arch, lr, batch size..
    trainer_config = get_trainer_config(config.data,
                                        config.training_method,
                                        config.train_mode)

    discovered_samples = dataset_info.discovered_samples
    discovered_classes = dataset_info.discovered_classes

    # Trainer is the main class for training and querying
    # It contains train() query() finetune() functions
    trainer = Trainer(
        config.training_method,
        config.train_mode,
        trainer_config,
        dataset_info,
        config.query_method,
        config.budget,
        global_setting.OPEN_SET_METHOD_DICT[config.training_method],
        paths_dict,
        dataset_info.test_dataset,
        val_samples=dataset_info.trainset_info.val_samples
    )

    trainer.train(discovered_samples, discovered_classes, verbose=config.verbose)

    discovered_samples, discovered_classes = trainer.query(discovered_samples, discovered_classes, verbose=config.verbose)
    
    trainer.finetune(discovered_samples, discovered_classes, verbose=config.verbose)
    
    closed_set_test_acc = trainer.eval_closed_set(discovered_classes, verbose=config.verbose)
    
    trainer.eval_open_set(discovered_samples, discovered_classes, verbose=config.verbose)
    

if __name__ == '__main__':
    main()
