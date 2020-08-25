"""Start a open set learning experiment
The network is first trained on initial labeled set, then evaluate on a test set which may contain
samples not in initially labeled set of classes.
"""

import time
import os

import torch

from config import get_config
from dataset_factory import prepare_dataset_from_config
from trainer import Trainer
from trainer_config import get_trainer_config
from utils import prepare_open_set_learning_dir_from_config, set_random_seed
import global_setting


def main():
    config = get_config()

    if not config.use_random_seed:
        set_random_seed(1)

    # It contains all directory/save_paths that will be used
    paths_dict = prepare_open_set_learning_dir_from_config(config)

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
    trainer = Trainer(
        training_method=config.training_method,
        trainer_config=trainer_config,
        dataset_info=dataset_info
    )

    trainer.train(
        discovered_samples,
        discovered_classes,
        ckpt_path=paths_dict['trained_ckpt_path'],
        verbose=config.verbose
    )

    closed_set_test_acc = trainer.eval_closed_set(
        discovered_classes,
        result_path=paths_dict['test_result_path'],
        verbose=config.verbose
    )

    # Evaluate on all open set methods
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
