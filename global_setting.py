OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNDISCOVERED_CLASS_INDEX = -1 # The index for unseen open set class examples

OPEN_SET_METHOD_DICT = {
    'softmax_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
    'cosine_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
    'deep_metric' : ['nn','nn_cosine', 'softmax', 'entropy'],
}

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
        # For active learning closed set experiments
        'active': {
            'num_init_classes' : 100, # Number of initial discovered classes
            'sample_per_class' : 10, # Number of samples per discovered class
            'num_open_classes' : 0, 
        },
    },
    'CIFAR10' : {
        'active': {
            'num_init_classes' : 10, # Number of initial discovered classes
            'sample_per_class' : 100, # Number of samples per discovered class
            'num_open_classes' : 0, 
        },
    },

    'CUB200': {
        'regular': {  # The setup in ICRA deep metric learning paper. 80 closed classes, 20 open classes.
            'num_init_classes': 90,  # Number of initial discovered classes
            'sample_per_class': 15,  # Number of samples per discovered class
            'num_open_classes': 20,  # Number of open classes hold out
        },
        'fewer_class': {  # Use 1/5 of discovered classes, but same number of samples per discovered class
            'num_init_classes': 18,  # Number of initial discovered classes
            'sample_per_class': 15,  # Number of samples per discovered class
            'num_open_classes': 20,
        },
        'fewer_sample': {  # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
            'num_init_classes': 90,  # Number of initial discovered classes
            'sample_per_class': 3,  # Number of samples per discovered class
            'num_open_classes': 20,
        },

    },

    'Cars': {
        'regular': {  # 160 closed, 36 hold out open
            'num_init_classes': 80,  # Number of initial discovered classes
            'sample_per_class': 20,  # Number of samples per discovered class
            'num_open_classes': 36,  # Number of open classes hold out
        },
        'fewer_class': {  # Use 1/5 of discovered classes, but same number of samples per discovered class
            'num_init_classes': 16,  # Number of initial discovered classes
            'sample_per_class': 20,  # Number of samples per discovered class
            'num_open_classes': 36,
        },
        'fewer_sample': {  # Use 1/5 of samples per discovered class, but keep 40 initial discovered classes
            'num_init_classes': 80,  # Number of initial discovered classes
            'sample_per_class': 4,  # Number of samples per discovered class
            'num_open_classes': 36,
        },

    },
}

SUPPORTED_DATASETS = list(DATASET_CONFIG_DICT.keys())

