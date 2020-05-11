import torch

import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import DatasetFactory

from tensorboardX import SummaryWriter
from trainer import get_trainer # Contains different ways to train model
from logger import get_logger
import utils
import json
import random


def main():
    config, _ = get_config()
    if config.use_random_seed:
        print("Using random random seed")
    else:
        print("Random seed use 30")
        random.seed(30)
        torch.manual_seed(30)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    dataset_factory = DatasetFactory(config.data, config.data_path, config.init_mode)
    train_dataset, test_dataset = dataset_factory.get_dataset()
    train_samples, train_labels = dataset_factory.get_train_set_info()
    classes, open_classes = dataset_factory.get_class_info()
    
    time_stamp = time.strftime("%Y-%m-%d %H:%M")
    if not config.resume:
        # Begin from scratch
        start_round = 0
        discovered_samples, discovered_classes = dataset_factory.get_init_train_set() # Get initial training set, seen classes
        open_samples = dataset_factory.get_open_samples_in_trainset() # Get open samples and classes in train set

        # trainer contains train() eval() load_checkpoint() functions
        trainer = get_trainer(config,
                              train_dataset, 
                              train_samples,
                              open_samples,
                              train_labels,
                              classes,
                              open_classes)

        log_name = "{}{}{}".format(utils.get_experiment_name(config),
                                   os.sep,
                                   time_stamp)
        ckpt_dir = os.path.join(config.ckpt_dir, log_name)
        if config.save_ckpt:
            utils.makedirs(ckpt_dir)
            print(ckpt_dir)
        else:
            print("Warning: Not saving.")

        if config.writer:
            writer = SummaryWriter(log_dir=os.path.join(".", "runs", ckpt_dir))
        else:
            writer = None
            print("Not writing to tensorboard")
        # logger save all performance data, and write to tensorboard every epoch/round
        logger = get_logger(log_name=log_name,
                            ckpt_dir=ckpt_dir,
                            writer=writer)
        
        logger.init_round(discovered_samples, open_samples, discovered_classes, open_classes)

        checkpoint = utils.get_checkpoint(start_round, discovered_samples, open_samples, discovered_classes, open_classes, trainer, logger)
                    
        if config.save_ckpt: utils.save_checkpoint(ckpt_dir, checkpoint)
    else:
        # Begin from the checkpoint
        if not os.path.isfile(config.resume):
            raise FileNotFoundError(f"No checkpoint file found at {config.resume}")
        
        print("=> loading checkpoint '{}'".format(config.resume))
        checkpoint = torch.load(config.resume)

        start_round = checkpoint['round']
        discovered_samples, open_samples, discovered_classes, open_classes = checkpoint['discovered_samples'], checkpoint['open_samples'], checkpoint['discovered_classes'], checkpoint['open_classes']
        # trainer contains train() eval() load_checkpoint() functions
        trainer = get_trainer(config,
                              train_dataset,
                              train_samples,
                              open_samples,
                              train_labels,
                              classes,
                              open_classes)

        trainer.load_checkpoint(checkpoint['trainer_checkpoint'])

        log_name = checkpoint['logger_checkpoint']['log_name']
        ckpt_dir = checkpoint['logger_checkpoint']['ckpt_dir']

        writer = SummaryWriter(log_dir=os.path.join(".", "runs", ckpt_dir))
        logger = get_logger(log_name=log_name,
                            ckpt_dir=ckpt_dir,
                            writer=writer)
        logger.load_checkpoint(checkpoint['logger_checkpoint'])

        
    # The main loop
    for round_i in range(start_round, config.max_rounds):
        print(f"Round [{round_i}]")

        # Test if log dir is short enough
        # log_strs = utils.get_experiment_name(config).split(os.sep)
        # dataset_str = utils.get_data_param(config)
        # method_str = utils.get_method_param(config)
        # training_str = '_'.join(log_strs[2:])
        # save_dir = os.path.join('first_round_thresholds', dataset_str, method_str, training_str)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        assert not (config.log_first_round_model and config.save_first_round_model)

        if config.log_everything:
            log_strs = utils.get_experiment_name(config).split(os.sep)
            dataset_str = utils.get_data_active_param(config)
            method_str = utils.get_method_param(config)
            active_str = utils.get_active_param(config)
            training_str = '_'.join(log_strs[2:])
            save_dir = os.path.join('open_active_results_new', dataset_str, "_".join([method_str, active_str]), training_str)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if config.log_first_round_model:
            trainer.trainer_machine.model = torch.load(f"{config.init_mode}.pt")
            print("Didn't train. Load model.")
            assert config.epochs == 0

        train_loss, train_acc, eval_results = trainer.train_then_eval(discovered_samples, discovered_classes, test_dataset, eval_verbose=True)
        
        if config.log_everything:
            if round_i == 0:
                results = {} # A dictionary, key is round number
                # log_strs = utils.get_experiment_name(config).split(os.sep)
                # dataset_str = utils.get_data_active_param(config)
                # method_str = utils.get_method_param(config)
                # active_str = utils.get_active_param(config)
                # training_str = '_'.join(log_strs[2:])
                # save_dir = os.path.join('open_active_results', dataset_str, "_".join([method_str, active_str]), training_str)
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                log_filename = os.path.join(save_dir, time_stamp+'.json')

            results[round_i] = {
                'discovered_samples' : discovered_samples,
                'exemplar_set' : trainer.get_exemplar_set(),
                'train_labels' : train_labels,
                'num_discovered_samples' : len(discovered_samples),
                'num_discovered_classes' : len(discovered_classes),
                'num_undiscovered_classes' : len(classes.difference(discovered_classes))-len(open_classes),
                'num_open_classes' : len(open_classes),
                'discovered_classes' : list(discovered_classes),
                'undiscovered_classes' : list(classes.difference(discovered_classes)),
                'open_classes' : list(open_classes),
                'eval_results' : eval_results,
                'thresholds' : trainer.get_thresholds_checkpoints()[round_i+1],
            }

            # if round_i == config.max_rounds - 1:
            json_dict = json.dumps(results)
            if round_i == config.max_rounds - 1:
                print(f"Writing all results to {log_filename}")
            else:
                print(f"Writing intermediate results to {log_filename}")
            f = open(log_filename, "w+")
            f.write(json_dict)
            f.close()

        if round_i == 0:
            if config.save_first_round_model:
                print("save model!!!")
                torch.save(trainer.trainer_machine.model, f"{config.init_mode}.pt")

            if config.log_first_round_thresholds:
                log_strs = utils.get_experiment_name(config).split(os.sep)
                dataset_str = utils.get_data_param(config)
                method_str = utils.get_method_param(config)
                training_str = '_'.join(log_strs[2:])
                save_dir = os.path.join('first_round_thresholds', dataset_str, method_str, training_str)
                first_round_thresholds = trainer.get_thresholds_checkpoints()[1]
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                json_dict = json.dumps(first_round_thresholds)
                f = open(os.path.join(save_dir, time_stamp+'.json'),"w+")
                f.write(json_dict)
                f.close()
                print(f"Writing results to {os.path.join(save_dir, time_stamp+'.json')}")
                # trainer.trainer_machine.draw_histogram()
                exit(0)

            if config.log_first_round:
                log_strs = utils.get_experiment_name(config).split(os.sep)
                setting_str = log_strs[0]
                training_str = '_'.join(log_strs[2:])
                save_path = os.path.join('first_round', setting_str+".txt")
                if not os.path.isfile(save_path):
                    with open(save_path, 'w+') as file:
                        title_str = "|".join(["overall", "closed_set", "open_set", "details"])
                        file.write(title_str + "\n")
                with open(save_path, "a") as file:
                    closed_acc = eval_results['seen_closed_acc']
                    open_acc = eval_results['holdout_open_acc']
                    overall_acc = (closed_acc + open_acc) / 2.
                    detail_str = "|".join([f"{overall_acc:.5f}", f"{closed_acc:.5f}", f"{open_acc:.5f}", training_str])
                    file.write(detail_str + "\n")
                print(f"Check {save_path}. Now exiting.")
                exit(0)

        if config.log_test_accuracy:
            if round_i == 0:
                test_accs = {}

                log_strs = utils.get_experiment_name(config).split(os.sep)
                dataset_str = utils.get_data_active_param(config)
                method_str = utils.get_method_param(config)
                active_str = utils.get_active_param(config)
                training_str = '_'.join(log_strs[2:])
                save_dir = os.path.join('learning_loss', dataset_str, "_".join([method_str, active_str]), training_str)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                active_learning_filename = os.path.join(save_dir, time_stamp+'.json')

            test_accs[len(discovered_samples)] = eval_results['seen_closed_acc']
            
            if round_i == config.max_rounds - 1:
                json_dict = json.dumps(test_accs)
                print(f"Writing test accuracy to {active_learning_filename}")
                f = open(active_learning_filename, "w+")
                f.write(json_dict)
                f.close()
    

        t_train, t_classes = trainer.select_new_data(discovered_samples, discovered_classes)

        new_discovered_classes = discovered_classes.union(t_classes)
        classes_diff = new_discovered_classes.difference(discovered_classes)
        discovered_classes = new_discovered_classes

        print(f"Recognized class from {len(discovered_classes)-len(classes_diff)} to {len(discovered_classes)}")

        if config.writer: 
            writer.add_scalar("/train_acc", train_acc, round_i)
            for acc_key in eval_results.keys():
                if isinstance(eval_results[acc_key], float):
                    writer.add_scalar("/"+acc_key, eval_results[acc_key], round_i)
            writer.add_scalar("/discovered_classes", len(discovered_classes), round_i)
        

        assert len(set(discovered_samples)) == len(discovered_samples)
        assert len(set(t_train)) == len(t_train)
        discovered_samples = list(set(discovered_samples).union(set(t_train)))

        logger.log_round(round_i, discovered_samples, discovered_classes, eval_results)
        

        if round_i % 20 == 0 and config.save_ckpt:
            # Save every 20 rounds
            checkpoint = utils.get_checkpoint(round_i, discovered_samples, open_samples, discovered_classes, open_classes, trainer, logger)
            utils.save_checkpoint(ckpt_dir, checkpoint)

    logger.finish()

if __name__ == '__main__':
    main()
