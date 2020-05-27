import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Config for Open World Active Learning')

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

parser = argparse.ArgumentParser(description='PyTorch Implementation of Open Active Algorithm')

dataset_args = add_argument_group('Dataset Param.')
dataset_args.add_argument('data',
                          default="CIFAR100",
                          choices=["CIFAR10", "CIFAR100", 'CUB200'],
                          help='Choice of dataset + preprocessing method.')
dataset_args.add_argument('--download_path', 
                          default="/scratch", metavar='PATH',
                          help='path to datasets download location default :%(default)')
dataset_args.add_argument('--save_path',
                          default='/share/coecis/open_active/datasets',
                          help='path to where the dataset information will be saved after initialized. If already exist, use existing dataset.')
dataset_args.add_argument('--dataset_rand_seed',
                          type=str2NoneInt,
                          default=None,
                          help='None then use first nth classes nth samples. Otherwise use the rand seed to select random classes and random samples.')

setting_arg = add_argument_group('Setting Param.')
setting_arg.add_argument('--init_mode',
                         default='regular',
                         choices=['regular', 'fewer_class', 'fewer_sample',
                                  ],
                         help="How to select the initial training/hold-out open set")

trainer_args = add_argument_group('Trainer Param.')
trainer_args.add_argument('--training_method',
                          default='softmax_network',
                          choices=['softmax_network',
                                   'sigmoid_network', # Use 1 v.s. rest sigmoid network instead of softmax.
                                   'cosine_network', # Use cosine similarity as final layer in place of fc layer
                                   'deep_metric',
                          ],
                          help='The training method determines how to train/finetune using labeled samples.'
                          )
trainer_args.add_argument('--trainer_save_dir', # The direction hierachy will be: {dataset}/{init_mode}/{trainer}/{train/query/finetune/result}
                          default='/share/coecis/open_active/trainers',
                          help='path to where the trainer checkpoints will be saved after training/finetuning. If already exist, use existing dataset.')

trainer_args.add_argument('--query_method',
                          default='entropy',
                          choices=['random', # random query
                                   'entropy',
                                   'softmax', # Max score of softmax as an uncertainty measure
                                   'uldr', # Unlabeled to labeled density
                                   'uldr_norm_cosine', # ULDR With features normalized and then use cosine distance
                                   'learnloss', # Learning Loss paper - might be a variant of it
                                   'coreset',
                                   'coreset_norm_cosine', # Coreset With features normalized and then use cosine distance
                          ],
                          help='The active learning method determines how to query from unlabeled pool.'
                          )
trainer_args.add_argument('--budget',
                          default=10,
                          type=int,
                          help='The budget to query from unlabeled pool.'
                          )

trainer_args.add_argument('--open_set_method',
                          default='entropy',
                          choices=['entropy',
                                   'softmax', # Max score of softmax as an uncertainty measure
                                   'nn', # Nearest Neighbor
                                   'openmax',
                                   'c2ae',
                          ],
                          help='The open set method determines how evaluate the open set score of a test sample.'
                          )

# Configs for analaysis.py
analysis_arg = add_argument_group('Analysis.')
analysis_arg.add_argument('--analysis_save_dir', default="/share/coecis/open_active/analysis",
                          help='The directory to save all the plots.')
analysis_arg.add_argument('--analysis_trainer', choices=['softmax_network', 'cosine_network'], default='same_sample',
                          help='For the budget constraint, whether to evaluate based on same query size, or same sample size.')
analysis_arg.add_argument('--budget_mode', choices=['1_5_10_20_50_100'], default='1_5_10_20_50_100',
                          help='For the budget constraint, whether to evaluate based on same query size, or same sample size.')


sigmoid_args = add_argument_group('Sigmoid Trainer Machine Param.')
sigmoid_args.add_argument('--sigmoid_train_mode',
                          default='mean',
                          choices=['sum', 'mean'], 
                          help="Sum is to add all class score (e.g. logsigmoid or 1-logsigmoid). Mean is to take average of all classes for each example")

# Need to clean out
c2ae_args = add_argument_group('C2AE Trainer Machine Param.')
c2ae_args.add_argument('--c2ae_train_mode',
                          default='default',
                          choices=['default', 'a_minus_1', 'default_mse', 'a_minus_1_mse', 'default_bce', 'a_minus_1_bce', 
                                   "debug_no_label", 'debug_no_label_mse', 'debug_no_label_bce', 'debug_no_label_dcgan',
                                   'debug_no_label_dcgan_mse', 'debug_no_label_not_frozen_dcgan_mse', 'a_minus_1_dcgan_mse', 'a_minus_1_dcgan',
                                   'a_minus_1_instancenorm_dcgan_mse', 'a_minus_1_instancenorm_dcgan', 'a_minus_1_instancenorm_affine_dcgan_mse', 'a_minus_1_instancenorm_affine_dcgan',
                                   'debug_no_label_not_frozen', 'debug_no_label_not_frozen_dcgan', 'debug_no_label_simple_autoencoder', 'debug_no_label_simple_autoencoder_bce',
                                   'debug_simple_autoencoder_bce', 'debug_simple_autoencoder_mse', 'debug_simple_autoencoder',
                                   'a_minus_1_dcgan_mse_not_frozen', 'a_minus_1_dcgan_not_frozen', 'UNet_mse', 'UNet'],
                          help="C2AE config")
c2ae_args.add_argument('--c2ae_alpha',
                          default=0.9, type=float,
                          help="C2AE alpha")
c2ae_args.add_argument('--c2ae_train_in_eval_mode',
                          default=True, type=str2bool,
                          help="C2AE whether to train model in eval mode")


learnloss_args = add_argument_group('Learning Loss Param.')
learnloss_args.add_argument('--learning_loss_train_mode',
                            default='default',
                            choices=['default', 'mse'],
                            help="Learning loss config")
learnloss_args.add_argument('--learning_loss_lambda',
                            default=1.0, type=float,
                            help="Learning loss lambda")
learnloss_args.add_argument('--learning_loss_margin',
                            default=1.0, type=float,
                            help="Learning loss margin")
learnloss_args.add_argument('--learning_loss_stop_epoch',
                            default=120, type=int,
                            help="Learning loss stop propagation (to target model) epoch")
learnloss_args.add_argument('--learning_loss_start_epoch',
                            default=0, type=int,
                            help="Learning loss start working epoch")


osdn_args = add_argument_group('OSDN Trainer Machine Param.')
osdn_args.add_argument('--weibull_tail_size',
                       default='fixed_20',
                       choices=['fixed_20','fixed_50'],
                       help='How to fit the weibull distribution. Default is using the largest 20 (or fewer) per category, as in official OSDN repo.'
                       )
osdn_args.add_argument('--alpha_rank',
                       default='fixed_10',
                       choices=['fixed_5', 'fixed_2', 'fixed_10','fixed_20', 'fixed_40'],
                       help='The alpha rank. Default is using the largest 10 (or fewer) per category as in official OSDN repo.'
                       )
osdn_args.add_argument('--osdn_eval_threshold',
                       default=0.5,
                       type=float,
                       help='(OSDN setting) If max class probability < threshold, then classify new example to unseen class')
osdn_args.add_argument('--mav_features_selection',
                       default='correct',
                       choices=['correct', 'all', 'none_correct_then_all'],
                       help='Determine how features are selected to generate the mean activation vector.')

disc_args = add_argument_group('Distance Metric Param.')
disc_args.add_argument('--distance_metric',
                       default='eucos',
                       choices=['eu', 'cos', 'eucos'],
                       help='How to measure the distance between two examples in the feature space. EU distance is always divided by eu_divby.'
                       )
disc_args.add_argument('--div_eu',
                       default=200.,
                       type=float,
                       help='EU distance will be divided by this parameter.'
                       )


training_arg = add_argument_group('Training and Finetuning Param.')
training_arg.add_argument('--train_mode', type=str,
                          choices=['default', 'no_finetune', 'default_lr01_200eps', 'fix_feature_extractor'], default='default',
                          help='The training and finetuning mode. Check TRAIN_CONFIG_DICT in trainer_machine.py.')
training_arg.add_argument('--workers', default=4, type=int,
                          help='number of data loading workers (default: 4)')
training_arg.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')

uncertainty_sampling_arg = add_argument_group('Uncertainty Measure Param.')
uncertainty_sampling_arg.add_argument('--uncertainty_measure',
                                      default='least_confident',
                                      choices=['least_confident',
                                               'most_confident',
                                               'random_query',
                                               'entropy',
                                               'highest_loss',
                                               'lowest_loss'],
                                      )
uncertainty_sampling_arg.add_argument('--active_random_sampling',
                                      default='none',
                                      type=str,
                                      choices=['fixed_10K', # As in learning the loss paper
                                               '1_out_of_5',  # 1/5 of the unlabled pool. Or budget if smaller than budget.
                                               'none',
                                              ],
                                      help='If true, use the random sampling scheme in "Learning Loss Active Learning" paper',
                                      )
uncertainty_sampling_arg.add_argument('--coreset_measure',
                                      default='greedy',
                                      choices=['greedy'],
                                      )
uncertainty_sampling_arg.add_argument('--coreset_feature',
                                      default='before_fc',
                                      choices=['before_fc', 'after_fc'],
                                      )

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