import torch

import numpy as np 

import time
import os
import copy
from tqdm import tqdm
from config import get_config

from dataset_factory import get_dataset_factory

from tensorboardX import SummaryWriter
from trainer import get_trainer # Contains different ways to train model
from logger import get_logger
import utils
# from tools_training import get_device, get_criterion, get_optimizer, get_scheduler, get_tensorboard_logger

def main():
    config, _ = get_config()
    
    dataset_factory = get_dataset_factory(config.data, config.data_path, config.init_mode)
    train_dataset, test_dataset = dataset_factory.get_dataset()
    train_samples, train_labels, classes = dataset_factory.get_train_set_info()
    
    # contains train() eval() load_checkpoint() functions
    trainer = get_trainer(config, 
                          train_dataset, 
                          train_samples, 
                          train_labels, 
                          classes)
    
    if not config.resume:
        # Begin from scratch
        start_round = 0
        s_train, seen_classes = dataset_factory.get_init_train_set() # Get initial training set, seen classes

        log_name = "{}_{}".format(utils.get_experiment_name(config), 
                                  time.strftime("%Y-%m-%d %H:%M"))
        ckpt_dir = os.path.join(config.ckpt_dir, log_name)
        utils.makedirs(ckpt_dir)

        writer = SummaryWriter(log_dir=os.path.join(".", "runs", ckpt_dir))
        # logger save all performance data, and write to tensorboard every epoch/round
        logger = get_logger(log_name=log_name,
                            ckpt_dir=ckpt_dir,
                            writer=writer)
        
        logger.init_round(s_train, seen_classes)

        checkpoint = utils.get_checkpoint(start_round, s_train, seen_classes, trainer, logger)
                    
        utils.save_checkpoint(ckpt_dir, checkpoint)
    else:
        # Begin from the checkpoint
        if not os.path.isfile(config.resume):
            raise FileNotFoundError(f"No checkpoint file found at {config.resume}")
        
        print("=> loading checkpoint '{}'".format(config.resume))
        checkpoint = torch.load(config.resume)

        start_round = checkpoint['round']
        s_train, seen_classes = checkpoint['s_train'], checkpoint['seen_classes']

        trainer.load_checkpoint(checkpoint['trainer_checkpoint'])

        log_name, ckpt_dir = checkpoint['log_name'], checkpoint['ckpt_dir']

        writer = SummaryWriter(log_dir=os.path.join(".", "runs", ckpt_dir))
        logger.load_checkpoint(checkpoint['logger_checkpoint'])

        
    # The main loop
    for round_i in range(start_round, config.max_rounds):
        trainer.train_new_round(s_train, seen_classes)

        multi_class_acc, open_set_acc = trainer.eval(test_dataset, seen_classes)
        
        
        t_train, t_classes = trainer.select_new_data(s_train, seen_classes)


        new_seen_classes = seen_classes.union(t_classes)
        classes_diff = new_seen_classes.difference(seen_classes)
        seen_classes = new_seen_classes
        
        print(f"Recognized class from {len(seen_classes)-len(classes_diff)} to {len(seen_classes)}")
        writer.add_scalar("/multi_class_acc", multi_class_acc, round_i)
        writer.add_scalar("/open_set_acc", open_set_acc, round_i)
        writer.add_scalar("/seen_classes", len(seen_classes), round_i)
        
        assert len(set(s_train)) == len(s_train)
        assert len(set(t_train)) == len(t_train)
        s_train = set(s_train).union(set(t_train))

        logger.log_round(round_i, s_train, seen_classes, multi_class_acc, open_set_acc)
        
        checkpoint = utils.get_checkpoint(round_i, s_train, seen_classes, trainer, logger)
        utils.save_checkpoint(ckpt_dir, checkpoint)

    logger.finish()

if __name__ == '__main__':
    main()
