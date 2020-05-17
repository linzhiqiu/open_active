OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNDISCOVERED_CLASS_INDEX = -1 # The index for unseen open set class examples

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}


uncertainty_type_dict = {
    'osdn' : ['osdn', 'osdn_modified', 'icalr_osdn', 'icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg'],
    'icalr' : ['icalr', 'icalr_learning_loss', 'icalr_osdn', 'icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg'],
    'cluster' : ['cluster'],
    'learning_loss' : ['network_learning_loss', 'icalr_learning_loss'],
    'network' : ['network', 'icalr', 'icalr_binary_softmax'],
    'sigmoid' : ['sigmoid', 'binary_softmax'],
}

# Only working for ICALR based modules
open_type_dict = {
    'osdn' : {'module' : ['icalr_osdn', 'icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg'],
              'type' : ['osdn']}, # Use osdn the open set probability
    # 'icalr' : ['icalr', 'icalr_learning_loss', 'icalr_osdn', 'icalr_osdn_modified', 'icalr_osdn_neg', 'icalr_osdn_modified_neg'],
    # 'cluster' : ['cluster'],
    'learning_loss' : { 'module': ['icalr_learning_loss'],
                        'type' : ['entropy', 'softmax'],},
    'network' : { 'module' : ['icalr'],
                  'type' : ['entropy', 'softmax']},
    'binary_softmax' : {'module' : ['icalr_binary_softmax'],
                        'type' : ['rbf'],},
    # 'sigmoid' : ['sigmoid', 'binary_softmax'],
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
    'open_set_more' : {
        'weibull_tail_size' : [10, 20, 50, 100],
        'alpha_rank' : [2, 3, 4, 5, 10, 15, 20, 40],
        'osdn_eval_threshold' : [0.01, 0.1, 0.2, 0.5]
    },
    'toy' : {
        'weibull_tail_size' : [10],
        'alpha_rank' : [2],
        'osdn_eval_threshold' : [0.01]
    },
}

SUPPORTED_DATASETS = ['CIFAR10', 'CIFAR100']

INIT_TRAIN_SET_CONFIG = {
'CIFAR100' : {
        'default' : {
            'num_init_classes' : 40,
            'sample_per_class' : 12, # initially 40 x 12 = 480 samples
            'num_open_classes' : 20, # So num_undiscovered_classes = 100 - 30 - 20 = 50
            'use_random_classes' : False
        },
        'open_set_leave_one_out' : {
            'num_init_classes' : 10,
            'sample_per_class' : 500,
            'num_open_classes' : 10,
            'use_random_classes' : False
        },
        'open_set_leave_one_out_new' : {
            'num_init_classes' : 10,
            'sample_per_class' : 500,
            'num_open_classes' : 10,
            'use_random_classes' : False
        },
        'learning_loss' : {
            'num_init_classes' : 100,
            'sample_per_class' : 10,
            'num_open_classes' : 0,
            'use_random_classes' : False
        },
        'learning_loss_start_random' : {
            'num_init_classes' : 100,
            'sample_per_class' : 10,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : True, # sample size at first is num_init_classes * sample_per_class
        },
        'learning_loss_start_random_tuning' : {
            'num_init_classes' : 100,
            'sample_per_class' : 10,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : True, # sample size at first is num_init_classes * sample_per_class
        },
        'few_shot_1' : {
            'num_init_classes' : 40,
            'sample_per_class' : 25,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : False,
        },
        'few_shot_3' : {
            'num_init_classes' : 100,
            'sample_per_class' : 20,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        },
        'many_shot_1' : {
            'num_init_classes' : 100,
            'sample_per_class' : 50,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        },
        'many_shot_2' : {
            'num_init_classes' : 100,
            'sample_per_class' : 100,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        },
        'many_shot_3' : {
            'num_init_classes' : 100,
            'sample_per_class' : 150,
            'num_open_classes' : 0,
            'use_random_classes' : False,
            'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        },
        'open_active_0' : {
            'num_init_classes' : 40,
            'sample_per_class' : 25,
            'num_open_classes' : 10,
            'use_random_classes' : False,
            'use_random_samples' : False,
        },
        'open_active_1' : {
            'num_init_classes' : 40,
            'sample_per_class' : 25,
            'num_open_classes' : 10,
            'use_random_classes' : False,
            'use_random_samples' : False,
        },
        'open_active_2' : {
            'num_init_classes' : 40,
            'sample_per_class' : 100,
            'num_open_classes' : 10,
            'use_random_classes' : False,
            'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        },
        # 'open_active_3' : {
        #     'num_init_classes' : 40,
        #     'sample_per_class' : 200,
        #     'num_open_classes' : 10,
        #     'use_random_classes' : False,
        #     'use_random_samples' : False, # sample size at first is num_init_classes * sample_per_class
        # },
        'no_learning' : {
            'num_init_classes' : 100,
            'sample_per_class' : 500,
            'num_open_classes' : 0,
            'use_random_classes' : False
        },
        'no_learning_10K' : {
            'num_init_classes' : 100,
            'sample_per_class' : 100,
            'num_open_classes' : 0,
            'use_random_classes' : False
        },
        'no_learning_9K' : {
            'num_init_classes' : 90,
            'sample_per_class' : 100,
            'num_open_classes' : 10,
            'use_random_classes' : False
        },
        'no_learning_9K_randomsample' : {
            'num_init_classes' : 90,
            'sample_per_class' : 100,
            'num_open_classes' : 10,
            'use_random_classes' : False,
            'use_random_samples' : True,
        },
        'no_learning_8280' : {
            'num_init_classes' : 90,
            'sample_per_class' : 92,
            'num_open_classes' : 10,
            'use_random_classes' : False
        },
        'no_learning_8280_randomsample' : {
            'num_init_classes' : 90,
            'sample_per_class' : 92,
            'num_open_classes' : 10,
            'use_random_classes' : False,
            'use_random_samples' : True,
        },
        'no_learning_50_open_classes' : {
            'num_init_classes' : 50,
            'sample_per_class' : 500,
            'num_open_classes' : 50,
            'use_random_classes' : False
        },
        'no_learning_5K_50_open_classes' : {
            'num_init_classes' : 50,
            'sample_per_class' : 100,
            'num_open_classes' : 50,
            'use_random_classes' : False
        },
        'cifar100_open_50': {
            'num_init_classes' : 50,
            'sample_per_class' : 500,
            'num_open_classes' : 50,
            'use_random_classes' : False,
            'use_random_samples' : False,
        },
        'cifar100_open_80': {
            'num_init_classes' : 80,
            'sample_per_class' : 500,
            'num_open_classes' : 20,
            'use_random_classes' : False,
            'use_random_samples' : False,
        },
        'cifar100_open_20': {
            'num_init_classes' : 20,
            'sample_per_class' : 500,
            'num_open_classes' : 80,
            'use_random_classes' : False,
            'use_random_samples' : False,
        }
    },
'CIFAR10' : {
        'learning_loss' : {
            'num_init_classes' : 10,
            'sample_per_class' : 100, # initially 40 x 12 = 480 samples
            'num_open_classes' : 0, # So num_undiscovered_classes = 10 - 10 = 0
            'use_random_classes' : False
        },
        'learning_loss_start_random' : {
            'num_init_classes' : 10,
            'sample_per_class' : 100,
            'num_open_classes' : 0, # So num_undiscovered_classes = 10 - 10 = 0
            'use_random_classes' : False,
            'use_random_samples' : True, # sample size at first is num_init_classes * sample_per_class
        },
        'learning_loss_start_random_tuning' : {
            'num_init_classes' : 10,
            'sample_per_class' : 100,
            'num_open_classes' : 0, # So num_undiscovered_classes = 10 - 10 = 0
            'use_random_classes' : False,
            'use_random_samples' : True, # sample size at first is num_init_classes * sample_per_class
        },
        'no_learning' : {
            'num_init_classes' : 10,
            'sample_per_class' : 5000,
            'num_open_classes' : 0,
            'use_random_classes' : False
        },
        'no_learning_10K' : {
            'num_init_classes' : 10,
            'sample_per_class' : 1000,
            'num_open_classes' : 0,
            'use_random_classes' : False
        },
        'no_learning_5_open_classes' : {
            'num_init_classes' : 5,
            'sample_per_class' : 5000,
            'num_open_classes' : 5,
            'use_random_classes' : False
        },
        'no_learning_5K_5_open_classes' : {
            'num_init_classes' : 5,
            'sample_per_class' : 1000,
            'num_open_classes' : 5,
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

GAN_STANDARD_SETUP_20 = {
                           'nc' : 3,
                           'nz' : 100,
                           'ngf' : 64,
                           'ndf' : 64,
                           'optim' : 'Adam',
                           'lr' : 0.0002,
                           'beta1' : 0.5,
                           'num_epochs' : 20,
                       }

GAN_STANDARD_SETUP_100 = {
                           'nc' : 3,
                           'nz' : 100,
                           'ngf' : 64,
                           'ndf' : 64,
                           'optim' : 'Adam',
                           'lr' : 0.0002,
                           'beta1' : 0.5,
                           'num_epochs' : 100,
                       }

FEATURE_GAN_STANDARD_SETUP = {
                                 'nc' : 100,
                                 'nz' : 2048,
                                 'ngf' : 512,
                                 'ndf' : 512,
                                 'optim' : 'Adam',
                                 'lr' : 0.0002,
                                 'beta1' : 0.5,
                                 'num_epochs' : 50,
                             }

# Below are param for GAN training
GAN_SETUP_DICT = {
    'single' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP,
            '20_epochs' : GAN_STANDARD_SETUP_20,
            '100_epochs' : GAN_STANDARD_SETUP_100,
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    },
    'multiple' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP,
            '20_epochs' : GAN_STANDARD_SETUP_20,
            '100_epochs' : GAN_STANDARD_SETUP_100,
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    },
    'background' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP,
            '20_epochs' : GAN_STANDARD_SETUP_20,
            '100_epochs' : GAN_STANDARD_SETUP_100,
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    },
    'background_noise' : {
        'ImageLevelGAN' : {
            'standard' : GAN_STANDARD_SETUP,
            '20_epochs' : GAN_STANDARD_SETUP_20,
            '100_epochs' : GAN_STANDARD_SETUP_100,
        },
        'FeatureLevelGAN' : {
            'standard' : FEATURE_GAN_STANDARD_SETUP
        }
    }
}
