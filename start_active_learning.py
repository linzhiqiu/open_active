"""Start a closed set active learning experiment (no open set classes)
The network is first trained on initial labeled set, then an active learning method will be used to
query (i.e., select) new samples to label. The network will then be trained on the new labeled set.
This process will repeat for different numbers of query.

Two active learning scheme:
    Suppose the size of initial labeled pool is 100.
    If the list of budget is [0, 100, 200, 300].
        (a) Sequential mode (common adopted by active learning literature):
            The network is first trained on 100 samples.
            Then it query for 100 new labeled samples, add them to labeled pool, and train on total 100+100=200 samples.
            Next it query for another 100 new labeled samples, add them to labeled pool, and train again on these total 300 samples.
            And go on..

        (b) Independent mode:
            The network is first trained on 100 samples.
            Then it query for 100 new labeled samples, and train on the 200 samples.
            However, it does not add those 100 new labeled samples to labeled pool.
            Instead, it used the trained network to query another 200 new samples, and train again on these total 100+200=300 samples.
            Only the initial labeled pool is always used for training.
            Caveat: Not a common way to do active learning.
"""
import time
import os

import torch

from config import get_config
from dataset_factory import prepare_dataset_from_config
from trainer import Trainer
from trainer_config import get_trainer_config
from utils import prepare_active_learning_dir_from_config, get_budget_list_from_config, set_random_seed


def main():
    config = get_config()

    if not config.use_random_seed:
        set_random_seed(1)

    # A list of budgets to query, e.g. [100, 200, 300].
    # Then the labeled pool will obtain 100 new samples each round, until 300 budgets are all used.
    budget_list = get_budget_list_from_config(config)

    # It contains all directory/save_paths that will be used
    paths_dict = prepare_active_learning_dir_from_config(config, budget_list)

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

    for i, b in enumerate(budget_list):
        # b is the budget for independent mode, need to adjust it for sequential mode
        if config.active_query_scheme == 'sequential':
            if i > 0:
                budget = b - budget_list[i-1]
            else:
                budget = b
        else:
            budget = b

        new_discovered_samples, new_discovered_classes = trainer.query(
            discovered_samples,
            discovered_classes,
            budget=budget,
            query_method=config.query_method,
            query_result_path=paths_dict['active_query_results'][b],
            verbose=config.verbose
        )

        if config.active_query_scheme == 'sequential':
            print("Using sequential mode, we updated the discovered samples")
            discovered_samples, discovered_classes = new_discovered_samples, new_discovered_classes
        else:
            print("Using independent mode, we do not update the initial labeled pool.")

        trainer.train(
            new_discovered_samples,
            new_discovered_classes,
            ckpt_path=paths_dict['active_ckpt_results'][b],
            verbose=config.verbose
        )

        closed_set_test_acc = trainer.eval_closed_set(
            new_discovered_classes,
            result_path=paths_dict['active_test_results'][b],
            verbose=config.verbose
        )


if __name__ == '__main__':
    main()
