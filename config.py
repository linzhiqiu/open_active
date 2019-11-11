import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='Config for Open World Active Learning')

def str2bool(v):
    return v.lower() in ('true', '1')

def str2lower(s):
    return s.lower()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

parser = argparse.ArgumentParser(description='PyTorch Implementation of Open Active Algorithm')

graphite_args = add_argument_group('Graphite Usage')
graphite_args.add_argument('--graphite',
                           default=False, type=str2bool,
                           help='Enable Graphite.')

dataset_args = add_argument_group('Dataset Param.')
dataset_args.add_argument('data', 
                          default="CIFAR100",
                          choices=["CIFAR100", "MNIST", "IMAGENET12", "TINY-IMAGENET", "CIFAR10"],
                          help='Choice of dataset')
dataset_args.add_argument('--data_path', 
                          default="./data", metavar='PATH',
                          help='path to datasets location default :%(default)')


preprocess_arg = add_argument_group('Image Preprocessing Param.')
preprocess_arg.add_argument('--default_prep', default=True, type=bool,
                            help="Whether to use default preprocessing scheme of the dataset\
                                  Otherwise, use customized param. as below.")
preprocess_arg.add_argument('--img_size', type=int, default=0,
                            help='Size of the image (patch) - square (size x size). \
                                  If 0, then no resizing/cropping of images.')
preprocess_arg.add_argument('--channel_normalize', action="store_true", default=True,
                            help="whether or not to normalize the input image (patchs) channel-wise.\
                                  i.e. subtract the mean and divide by standard deviation per channel.")
preprocess_arg.add_argument('--center_crop', 
                            type=str2bool,
                            default=False,
                            help='whether or not to center crop the images')
preprocess_arg.add_argument('--random_crop',
                            type=str2bool,
                            default=True,
                            help='whether or not to random crop the images')
preprocess_arg.add_argument('--random_flip',
                            type=str2bool,
                            default=True,
                            help='whether or not to random horizontal flip the images')

trainer_args = add_argument_group('Trainer Param.')
trainer_args.add_argument('--trainer',
                          default='network',
                          choices=['network',
                                   'binary_softmax', # Softmax network train but use sigmoid function
                                   'icalr',
                                   'icalr_learning_loss',
                                   'icalr_binary_softmax',
                                   'sigmoid', # Use 1 v.s. rest sigmoid network instead of softmax.
                                   'c2ae',
                                   'gan',
                                   'cluster', # Will be using the distance_metric
                                   'osdn', # Open Set Deep Network
                                   'osdn_modified', # Open Set Deep Network. Without modifying the seen class score.
                                   'icalr_osdn_neg', # Open Set Deep Network negative scores
                                   'icalr_osdn_modified_neg', # Open Set Deep Network. Without modifying the seen class score. Negative score
                                   'icalr_osdn',
                                   'icalr_osdn_modified',
                                   'network_learning_loss'
                          ],
                          )

trainer_args = add_argument_group('Imbalanced dataset Param.')
trainer_args.add_argument('--class_weight',
                          default='uniform',
                          choices=['uniform', 'imbal' # Weight of each class is Total_Num/Total_Num_of_class
                          ],
                          )

network_args = add_argument_group('Network Trainer Machine Param.')
network_args.add_argument('--threshold_metric',
                          default='softmax',
                          choices=['entropy', 'softmax', 'gaussian'],
                          help="Use which score to measure the open set threshold")
network_args.add_argument('--network_eval_mode',
                          default='threshold',
                          choices=['threshold',
                                   'dynamic_threshold', # dynamic threshold adjust the open set threshold based on information of new instances
                                   'pseuopen_threshold', # adjust the open set threshold based on pseudo open class examples
                                  ],
                          help='How to perform open set recognition with Network Trainer'
                          )
network_args.add_argument('--network_eval_threshold',
                          default=0.5,
                          type=float,
                          help='If max class probility < threshold, then classify to unseen class')

sigmoid_args = add_argument_group('Sigmoid Trainer Machine Param.')
sigmoid_args.add_argument('--sigmoid_train_mode',
                          default='mean',
                          choices=['sum', 'mean'], 
                          help="Sum is to add all class score (e.g. logsigmoid or 1-logsigmoid). Mean is to take average of all classes for each example")


icalr_binary_softmax_args = add_argument_group('icalr_binary_softmax_train_mode Trainer Machine Param.')
icalr_binary_softmax_args.add_argument('--icalr_binary_softmax_train_mode',
                                        default='default',
                                        choices=['default', 'fixed_variance'], 
                                        help="Default is to use variance of distances to each class mean. Fixed variance use 1.")


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
                       choices=['fixed_20'],
                       help='How to fit the weibull distribution. Default is using the largest 20 (or fewer) per category, as in official OSDN repo.'
                       )
osdn_args.add_argument('--alpha_rank',
                       default='fixed_10',
                       choices=['fixed_5', 'fixed_10','fixed_20', 'fixed_40'],
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
# osdn_args.add_argument('--osdn_eval_mode',
#                        default='threshold',
#                        choices=['threshold'],
#                        help='How to perform open set recognition with Network Trainer'
#                        )

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

gan_args = add_argument_group('GAN Param.')
gan_args.add_argument('--gan_player',
                       default='single',
                       choices=['single', # All classes trained together, one GAN
                                'multiple', # #class GANs, use all D
                                'background', # #class discriminators, use background classes
                                'background_noise', # #class discriminators, use background classes + noisy distribution
                                ],
                       help='How many GANs are trained'
                       )
gan_args.add_argument('--gan_mode',
                       default='ImageLevelGAN',
                       choices=['ImageLevelGAN', 'MultiLabelImageLevelGAN', 'FeatureLevelGAN'],
                       help='GAN mode. MultiLabel version only available with single player'
                       )
gan_args.add_argument('--gan_setup',
                       default='standard',
                       choices=['standard', '20_epochs', '100_epochs'],
                       help='For image level it is a standard DCGAN.'
                     )
gan_args.add_argument('--gan_multi',
                       default='all',
                       choices=['all', 'highest', 'lowest'],
                       help='In multi GAN setting, which discriminator to use. Score based on softmax',
                     )

cluster_args = add_argument_group('Cluster Network Trainer Machine Param.')
cluster_args.add_argument('--clustering',
                          # default=['rbf_train','train_means'],
                          default=['rbf_train'],
                          help='The clustering algorithm to use. train_means use train examples to compute the cluster. rbf_train use rbf training objective')
cluster_args.add_argument('--rbf_gamma',
                          default=1.0, type=float,
                          help='The temperature parameter in K(x,x") = exp(-gamma*disc(x,x")).')
cluster_args.add_argument('--cluster_eval_threshold',
                          default=0.5, type=float,
                          help='The threshold to reject open set example.')
cluster_args.add_argument('--cluster_level',
                          default='after_fc', choices=['after_fc', 'before_fc'],
                          help='Where to take the feature')

icalr_args = add_argument_group('ICaLR Network Trainer Machine Param.')
icalr_args.add_argument('--icalr_mode',
                          default='default',
                          choices=['default'],
                          help='The ICALR training mode.')
icalr_args.add_argument('--icalr_exemplar_size',
                          default=None, type=int,
                          help='Size of exemplar set for each class. None then no limit.')
icalr_args.add_argument('--icalr_retrain_criterion',
                          default='round', choices=['round', 'sample', 'class'],
                          help='Retrain network criterion.')
icalr_args.add_argument('--icalr_strategy',
                          default=None, choices=['naive', 'proto', 'smooth'],
                          help='Naive just simply trains new weight vector. Proto use prototype as activation score.')
icalr_args.add_argument('--icalr_naive_strategy',
                          default='fixed', choices=['fixed', 'finetune'],
                          help='For naive strategy: Fixed is fixing weight of network representation. Finetune just try the entire network.')
icalr_args.add_argument('--icalr_proto_strategy',
                          default=None, choices=['default'],
                          help='For proto strategy: Only default mode available. Strictly follows the ICALR paper if there is budget limit.')
icalr_args.add_argument('--icalr_retrain_threshold',
                          default=10, type=int,
                          help='Retrain network threshold (e.g. 10 classes, 10 rounds, 200 samples.)')

training_arg = add_argument_group('Network Training Param.')
training_arg.add_argument('--arch', '-a', type=str, metavar='ARCH',
                          choices=['alexnet', 'vgg16', 'resnet101', 'ResNet50', 
                          'ResNet18', 'classifier32', 'classifier32_instancenorm', 'classifier32_instancenorm_affine'], default='ResNet50',
                          help='CNN architecture (default: ResNet50)')
training_arg.add_argument('--pretrained',
                          choices=[None, 'CIFAR10'],
                          default=None,
                          help='Whether the model is pretrained.')
training_arg.add_argument('--lr', default=0.1, type=float,
                          help='learning rate (default: 0.1)')
training_arg.add_argument('--optim', default="sgd", choices=["sgd", "adam"], type=str,
                          help='optimizer (default: sgd)')
training_arg.add_argument('--momentum', default=0.9, type=float, 
                          help='momentum (default: 0.9)')
training_arg.add_argument('--wd', default=5e-4, type=float,
                          help='weight decay pow (default: 5e-4)')
training_arg.add_argument("--amsgrad", 
                          type=str2bool,
                          default=True, 
                          help="If using adam whether to use amsgrad ")
# training_arg.add_argument("--shuffle", 
#                           type=str2bool,
#                           default=True, 
#                           help="Whether to shuffle the dataset each epoch")
training_arg.add_argument('--workers', default=4, type=int,
                          help='number of data loading workers (default: 4)')
training_arg.add_argument('--epochs', type=int, default=50,
                          help='number of total epochs to run (default: 50)')
training_arg.add_argument('--smooth_epochs', type=int, default=10,
                          help='number of smooth epochs to run (default: 10)')
training_arg.add_argument('--start_epoch', default=0, type=int,
                          help='manual epoch number (useful on restarts) (default: 0)')
training_arg.add_argument('--batch', default=128, type=int,
                          help='mini-batch size (default: 128)')
training_arg.add_argument("--lr_decay_step", type=int, default=None,
                          help="After [lr_decay_step] epochs, decay the learning rate by [lr_decay_ratio].")
training_arg.add_argument("--lr_decay_ratio", "--lr_decay_ratio", 
                          type=float, default=0.1,
                          help="After [lr_decay_step] epochs, decay the learning rate by this ratio. Default is 0.1")
training_arg.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
training_arg.add_argument('--resume', default='', type=str, metavar='PATH',
                          help='path to checkpoint (default: None)')


open_act_arg = add_argument_group('Open Active Learning Param.')
# Should be different according to datasets
open_act_arg.add_argument('--max_rounds',
                          default=500,
                          type=int,
                          help='The number of rounds to play.')
open_act_arg.add_argument('--budget',
                          default=20,
                          type=int,
                          help='The budget constraint for each round.')

setting_arg = add_argument_group('Setting Param.')
setting_arg.add_argument('--init_mode',
                         default='default',
                         choices=['default', 'no_learning_5K_5_open_classes', 'no_learning_5_open_classes', 'no_learning_5K_50_open_classes', 'no_learning_50_open_classes',  'open_set_leave_one_out', 'open_set_leave_one_out_new', 'open_active_1', 'open_active_2', 'many_shot_1','many_shot_2','many_shot_3', 'few_shot_1', 'few_shot_3', 'no_learning', 'no_learning_10K', 'learning_loss', 'learning_loss_start_random', 'learning_loss_start_random_tuning',
                                  'cifar100_open_50', 'cifar100_open_80', 'cifar100_open_20',
                                  'no_learning_9K_randomsample', 'no_learning_9K'],
                         help="How to select the initial training/hold-out open set")

exp_vs_acc_arg = add_argument_group('Exploitation v.s. accuracy Param.')
exp_vs_acc_arg.add_argument('--label_picker',
                            default='uncertainty_measure',
                            choices=['uncertainty_measure', 'coreset_measure', 'open_active'],
                            )
exp_vs_acc_arg.add_argument('--open_active_setup', # If label_picker is open_active
                            default='active',
                            choices=['half', 'active', 'open'], # half is half of each score (normalize to [0..1] first).
                            )

uncertainty_sampling_arg = add_argument_group('Uncertainty Measure Param.')
uncertainty_sampling_arg.add_argument('--uncertainty_measure',
                                      default='least_confident',
                                      choices=['least_confident',
                                               'most_confident',
                                               'random_query',
                                               'margin_sampling',
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

pseudo_open_arg = add_argument_group('Pseudo-open set hyper tuning Param.')
pseudo_open_arg.add_argument('--pseudo_open_set',
                             default=None,
                             choices=[None,
                                      1,
                                      5, 4, 6,
                                      10],
                             type=int,
                             help='The number of pseudo-open set class (from training class). None if not using any.'
                            )
pseudo_open_arg.add_argument('--pseudo_open_set_rounds',
                             default=0,
                             type=int,
                             help='How many rounds that we perform pseudo open set hyper-tuning.'
                            )
pseudo_open_arg.add_argument('--pseudo_open_set_metric',
                             default='weighted',
                             choices=['weighted', 'average', '7_3'],
                             help='What is a good hyper? (a) weighted by example size. (b) average of closed and open set. Currently only support OSDNish methods.'
                            )
pseudo_open_arg.add_argument('--pseudo_same_network',
                             default=False,
                             type=str2bool,
                             help='For debugging purpose. Use same network for pseudo training and actual training.',
                            )
pseudo_open_arg.add_argument('--openmax_meta_learn',
                             default=None,
                             choices=['default', 'advanced', 'morealpha', 'open_set', 'toy', 'open_set_more'],
                             help='The meta learning setting for OpenMax/Modified OpenMax algorithm when using pseudo-open classes'
                            )


misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--log_everything', type=str2bool, default=False, 
                      help='Log all results.')
misc_arg.add_argument('--log_first_round', type=str2bool, default=False, #default='first_round.txt'
                      help='If True, log first round results to first_round/')
misc_arg.add_argument('--log_first_round_thresholds', type=str2bool, default=False, #default='first_round.txt'
                      help='If True, log first round threshold results to first_round_thresholds/')
misc_arg.add_argument('--log_test_accuracy', type=str2bool, default=True, #default='first_round.txt'
                      help='If True, log test acc results to learning_loss/')
misc_arg.add_argument('--writer', type=str2bool, default=True,
                      help='Whether or not to use writer')
misc_arg.add_argument('--save_ckpt', type=str2bool, default=True,
                      help='Whether or not to save ckpt')
misc_arg.add_argument('--device', type=str, default='cuda',
                      help='Which device to use.')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpts/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--out_dir', type=str, default='./outputs/',
                      help='Directory to store any output files')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory to store Tensorboard logs')
misc_arg.add_argument('--debug', action="store_true",
                      help="Whether to use the debug mode")
misc_arg.add_argument('--verbose', action='store_true', default=False, 
                      help='chatty')
misc_arg.add_argument('--save_gan_output', action='store_true', default=False, 
                      help='If true, save to gan_output/')
misc_arg.add_argument('--use_random_seed', action='store_true', default=False,
                      help='If true, use a random random seed')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed