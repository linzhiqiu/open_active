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

dataset_args = add_argument_group('Dataset Param.')
dataset_args.add_argument('data', 
                          default="CIFAR100",
                          choices=["CIFAR10", "CIFAR100", "MNIST", "IMAGENET12", "TINY-IMAGENET"],
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
                          choices=['network', 'open_set_network'],
                          )

network_args = add_argument_group('Network Trainer Machine Param.')
network_args.add_argument('--network_eval_mode',
                          default='threshold',
                          choices=['threshold'],
                          help='How to perform open set recognition with Network Trainer'
                          )
network_args.add_argument('--network_eval_threshold',
                          default=0.5,
                          type=float,
                          help='If max class probility < threshold, then classify to unseen class')


training_arg = add_argument_group('Network Training Param.')
training_arg.add_argument('--arch', '-a', type=str, metavar='ARCH',
                          choices=['alexnet', 'vgg16', 'resnet101', 'ResNet50'], default='ResNet50',
                          help='CNN architecture (default: ResNet50)')
training_arg.add_argument('--pretrained',
                          choices=[None, 'CIFAR10'],
                          default=None,
                          help='Whether the model is pretrained.')
training_arg.add_argument('--lr', default=0.1, type=float,
                          help='learning rate (default: 0.01)')
training_arg.add_argument('--optim', default="sgd", choices=["sgd", "adam"], type=str,
                          help='optimizer (default: sgd)')
training_arg.add_argument('--momentum', default=0.9, type=float, 
                          help='momentum (default: 0.9)')
training_arg.add_argument('--wd', default=-5, type=float,
                          help='weight decay pow (default: -5)')
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
training_arg.add_argument('--start_epoch', default=0, type=int,
                          help='manual epoch number (useful on restarts) (default: 0)')
training_arg.add_argument('--batch', default=128, type=int,
                          help='mini-batch size (default: 128)')
training_arg.add_argument("--lr_decay_step", type=int, default=None,
                          nargs="+",
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
                          default=50,
                          type=int,
                          help='The number of rounds to play.')
open_act_arg.add_argument('--budget',
                          default=500,
                          type=int,
                          help='The budget constraint for each round.')

init_s_arg = add_argument_group('Initial Training Set Param.')
init_s_arg.add_argument('--init_mode',
                        default='default',
                        choices=['default'],
                        help="How to select the initial training set")

exp_vs_acc_arg = add_argument_group('Exploitation v.s. accuracy Param.')
exp_vs_acc_arg.add_argument('--label_picker',
                            default='uncertainty_measure',
                            choices=['uncertainty_measure'],
                            )

uncertainty_sampling_arg = add_argument_group('Uncertainty Measure Param.')
uncertainty_sampling_arg.add_argument('--uncertainty_measure',
                                      default='least_confident',
                                      choices=['least_confident',
                                               'most_confident', 
                                               'margin_sampling',
                                               'entropy'],
                                      )

misc_arg = add_argument_group('Misc.')
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


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed