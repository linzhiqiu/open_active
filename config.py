import argparse

arg_lists = []
parser = argparse.ArgumentParser(
    description='Config for Open World Active Learning')


def str2bool(v):
    return v.lower() in ('true', '1')


def str2lower(s):
    return s.lower()


def str2NoneInt(s):
    return None if s == "None" else int(s)


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


parser = argparse.ArgumentParser(
    description='PyTorch Implementation of Open Active Algorithm')

dataset_args = add_argument_group('Dataset Param.')
dataset_args.add_argument('data',
                          default="CIFAR100",
                          choices=["CIFAR10", # Regular CIFAR10
                                   "OPEN_CIFAR10", # CIFAR10 for open set learning
                                   "CIFAR100",
                                   'CUB200',
                                   'Cars'],
                          help='Dataset for training and testing. Dataset details can be found in dataset_factory.py')
dataset_args.add_argument('--data_config',
                          default='regular',
                          choices=['regular', 'fewer_class', 'fewer_sample'],
                          help="This parameter decides how to select the initial labeled set and hold-out open set classes on given dataset")
dataset_args.add_argument('--data_download_path',
                          default="./",
                          help='The location for downloaded datasets.')
dataset_args.add_argument('--data_save_path',
                          default='./datasets',
                          help='path to where the dataset information will be saved after initialized. If already exist, use existing dataset.')
dataset_args.add_argument('--data_rand_seed',
                          type=str2NoneInt,
                          default=None,
                          help='None then use first nth classes nth samples. Otherwise use the rand seed to select random classes and random samples.')
dataset_args.add_argument('--use_val_set',
                          type=str2bool,
                          default=False,
                          help='None then not using val set.')


setting_arg = add_argument_group('Setting Param.')
setting_arg.add_argument('--val_mode',
                         default=None,
                         choices=['randomized', 'balanced',
                                  ],
                         help="How to select the validation set (for train.py)")

trainer_args = add_argument_group('Trainer Param.')
trainer_args.add_argument('--training_method',
                          default='softmax_network',
                          choices=['softmax_network',
                                   'cosine_network',  # Use cosine similarity as final layer in place of fc layer
                                   ],
                          help='The training method determines how to train/finetune using labeled samples.'
                          )
trainer_args.add_argument('--trainer_save_dir',  # The direction hierachy will be: {dataset}/{data_config}/{trainer}/{train/query/finetune/result}
                          default='./trainers',
                          help='path to where the trainer checkpoints will be saved after training/finetuning. If already exist, use existing dataset.')


trainer_args.add_argument('--query_method',
                          default='entropy',
                          choices=['random',  # random query
                                   'entropy',
                                   'softmax',  # Max score of softmax as an uncertainty measure
                                   'uldr',  # Unlabeled to labeled density
                                   'uldr_norm_cosine',  # ULDR With features normalized and then use cosine distance
                                   'learnloss',  # Learning Loss paper - might be a variant of it
                                   'coreset',
                                   'coreset_norm_cosine',  # Coreset With features normalized and then use cosine distance
                                   ],
                          help='The active learning method determines how to query from unlabeled pool.'
                          )
trainer_args.add_argument('--budget',
                          default=10,
                          type=int,
                          help='The budget to query from unlabeled pool.'
                          )

# Warning not used anymore
trainer_args.add_argument('--open_set_method',
                          default='entropy',
                          choices=['entropy',
                                   'softmax',  # Max score of softmax as an uncertainty measure
                                   'nn',  # Nearest Neighbor with eu distance
                                   'nn_cosine',  # Nearest Neighbor with cosine distance
                                   'openmax',
                                   'c2ae',
                                   ],
                          help='The open set method determines how evaluate the open set score of a test sample.'
                          )

# Configs for analaysis.py
analysis_arg = add_argument_group('Analysis.')
analysis_arg.add_argument('--analysis_save_dir', default="/share/coecis/open_active/analysis",
                          help='The directory to save all the plots.')
analysis_arg.add_argument('--analysis_trainer', choices=['softmax_network', 'cosine_network', 'deep_metric'], default='same_sample',
                          help='For the budget constraint, whether to evaluate based on same query size, or same sample size.')
analysis_arg.add_argument('--budget_mode', choices=['1_5_10_20_50_100'], default='1_5_10_20_50_100',
                          help='For the budget constraint, whether to evaluate based on same query size, or same sample size.')

active_analysis_arg = add_argument_group('Active Learning Analysis.')
active_analysis_arg.add_argument('--active_analysis_save_dir', default="/share/coecis/open_active/active_analysis",
                                 help='The directory to save all the plots for closed set active learning')


training_arg = add_argument_group('Training and Finetuning Param.')
training_arg.add_argument('--train_mode', type=str,
                          choices=['retrain'], default='retrain',
                          help='The training and finetuning mode. Check TRAIN_CONFIG_DICT in trainer_machine.py.')
training_arg.add_argument('--workers', default=4, type=int,
                          help='number of data loading workers (default: 4)')
training_arg.add_argument('--seed', type=int, default=31,
                          help='random seed (default: 31)')


active_arg = add_argument_group('Active Learning Param.')
active_arg.add_argument('--active_save_path',
                        default='/share/coecis/open_active/active_datasets',
                        help='path to where the dataset information will be saved after initialized. If already exist, use existing dataset.')
active_arg.add_argument('--active_query_scheme', type=str,
                        choices=['sequential', 'independent'], default='sequential',
                        help='How the query is selected etc.')
active_arg.add_argument('--active_train_mode', type=str,
                        choices=['retrain'], default='retrain',
                        help='How many epochs etc.')
active_arg.add_argument('--active_save_dir',  # The direction hierachy will be: {dataset}/{data_config}/{training_method}/{active_query_scheme}/{active_train_mode}/{seed}/{round}
                        default='/share/coecis/open_active/active_learners',
                        help='path to where the active learning checkpoints will be saved after training/finetuning. If already exist, use existing dataset.')
active_arg.add_argument('--active_budget_mode',
                        default='default',
                        choices=['default',  # the default. Detail in global_settings.py
                                 ],
                        help="How to select the budget to query and evaluate")

open_arg = add_argument_group("Open set Learning Param")
open_arg.add_argument('--open_set_save_path',
                      default='/share/coecis/open_active/open_datasets',
                      help='path to where the dataset information will be saved after initialized. If already exist, use existing dataset.')
open_arg.add_argument('--open_set_train_mode', type=str,
                      choices=['default'], default='default',
                      help='The training mode for open set recognition.')
open_arg.add_argument('--open_set_save_dir',  # The direction hierachy will be: {dataset}/{data_config}/{training_method}/{open_set_train_mode}/{seed}/{open_set_method}
                      default='/share/coecis/open_active/open_learners',
                      help='path to where the open set learning checkpoints will be saved after training/eval. If already exist, use existing dataset.')


misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--device', type=str, default='cuda',
                      help='Which device to use.')
misc_arg.add_argument('--debug', action="store_true",
                      help="Whether to use the debug mode")
misc_arg.add_argument('--verbose', type=str2bool, default=True,
                      help='chatty')
misc_arg.add_argument('--use_random_seed', action='store_true', default=False,
                      help='If true, use a random random seed. Otherwise, use 30 as the seed.')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config
