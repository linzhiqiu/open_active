OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNDISCOVERED_CLASS_INDEX = -1 # The index for unseen open set class examples

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}

DATASET_CONFIG_DICT = {
    'CIFAR100' : {
        'regular': { # The setup in ICRA deep metric learning paper. 80 closed classes, 20 open classes.
            'num_init_classes' : 40, # Number of initial discovered classes
            'sample_per_class' : 250, # Number of samples per discovered class
            'num_open_classes' : 20, # Number of open classes hold out
        },
        'fewer_class': { # Use 1/5 of discovered classes, but same number of samples per discovered class
            'num_init_classes' : 8, # Number of initial discovered classes
            'sample_per_class' : 250, # Number of samples per discovered class
            'num_open_classes' : 20, 
        },
        'fewer_sample': { # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
            'num_init_classes' : 40, # Number of initial discovered classes
            'sample_per_class' : 50, # Number of samples per discovered class
            'num_open_classes' : 20, 
        },
     
    },
    'CIFAR10' : {}
}

SUPPORTED_DATASETS = list(DATASET_CONFIG_DICT.keys())

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