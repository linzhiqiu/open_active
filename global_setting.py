OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNSEEN_CLASS_INDEX = -1 # The index for unseen open set class examples

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}

OPENMAX_META_LEARN = {
    'default' : {
        'weibull_tail_size' : [20],
        'alpha_rank' : [2, 3, 5, 10],
        'osdn_eval_threshold' : [0.05, 0.1, 0.15, 0.2],
    },
    'advanced' : {
        'weibull_tail_size' : [10, 20],
        'alpha_rank' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'osdn_eval_threshold' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]
    },
    'morealpha' : {
        'weibull_tail_size' : [10, 20, 50],
        'alpha_rank' : [2, 5, 10, 15, 20, 25, 30],
        'osdn_eval_threshold' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5]
    },
    'open_set' : {
        'weibull_tail_size' : [10, 20, 50],
        'alpha_rank' : [2, 3, 4, 5],
        'osdn_eval_threshold' : [0.01, 0.1, 0.2, 0.5]
    },
}

SUPPORTED_DATASETS = ['CIFAR10', 'CIFAR100']

INIT_TRAIN_SET_CONFIG = {
'CIFAR100' : {
        'default' : {
            'num_init_classes' : 40,
            'sample_per_class' : 12, # initially 40 x 12 = 480 samples
            'num_open_classes' : 20, # So num_unseen_classes = 100 - 30 - 20 = 50
            'use_random_classes' : False
        },
        'open_set_leave_one_out' : {
            'num_init_classes' : 10,
            'sample_per_class' : 500,
            'num_open_classes' : 10,
            'use_random_classes' : False
        },
    }
}

GAN_STANDARD_SETUP = {
                         'nc' : 3,
                         'nz' : 100,
                         'ngf' : 64,
                         'ndf' : 64,
                         'optim' : 'Adam',
                         'lr' : 0.0002,
                         'beta1' : 0.5,
                         'num_epochs' : 5,
                     }

FEATURE_GAN_STANDARD_SETUP = {
                                 'nc' : 100,
                                 'nz' : 2048,
                                 'ngf' : 512,
                                 'ndf' : 512,
                                 'optim' : 'Adam',
                                 'lr' : 0.0002,
                                 'beta1' : 0.5,
                                 'num_epochs' : 5,
                             }

# Below are param for GAN training
GAN_SETUP_DICT = {
    'single' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    },
    'multiple' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    }
}
