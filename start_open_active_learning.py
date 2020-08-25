import torch
import numpy as np

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import prepare_dataset_from_config

from trainer import Trainer
from trainer_config import get_trainer_config
import utils
from utils import makedirs
import json
import random

from utils import prepare_save_dir_from_config, set_random_seed
import global_setting


def main():
    config = get_config()

    if not config.use_random_seed:
        set_random_seed(1)

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
        torch.save(
            dataset_info.trainset_info,
            paths_dict['trainset_info_path']
        )

    # The training configurations including backbone architecture, lr, batch size..
    trainer_config = get_trainer_config(
        config.data,
        config.training_method,
        config.train_mode
    )

    discovered_samples = dataset_info.discovered_samples
    discovered_classes = dataset_info.discovered_classes

    # Trainer is the main class for training and querying
    # It contains train() query() finetune() functions
    trainer = Trainer(
        training_method=config.training_method,
        trainer_config=trainer_config,
        dataset_info=dataset_info,
        query_method=config.query_method,
    )

    # First time training
    trainer.train(
        discovered_samples,
        discovered_classes,
        ckpt_path=paths_dict['trained_ckpt_path'],
        verbose=config.verbose
    )

    # Perform active querying
    discovered_samples, discovered_classes = trainer.query(
        discovered_samples,
        discovered_classes,
        budget=config.budget,
        query_result_path=paths_dict['query_result_path'],
        verbose=config.verbose
    )

    # Perform second round of training
    trainer.train(
        discovered_samples,
        discovered_classes,
        ckpt_path=paths_dict['finetuned_ckpt_path'],
        verbose=config.verbose
    )

    closed_set_test_acc = trainer.eval_closed_set(
        discovered_classes,
        result_path=paths_dict['test_result_path'],
        verbose=config.verbose
    )

    trainer.eval_open_set(
        discovered_samples,
        discovered_classes,
        open_set_methods=global_setting.OPEN_SET_METHOD_DICT[config.training_method],
        result_paths=paths_dict['open_result_paths'],
        roc_paths=paths_dict['open_result_roc_paths'],
        goscr_paths=paths_dict['open_result_goscr_paths'],
        verbose=config.verbose
    )


if __name__ == '__main__':
    main()
