OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNDISCOVERED_CLASS_INDEX = -1 # The index for unseen open set class examples

OPEN_SET_METHOD_DICT = {
    # 'softmax_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax', 'c2ae'],
    # 'cosine_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax', 'c2ae'],
    'softmax_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
    'cosine_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
    'deep_metric' : ['nn','nn_cosine', 'softmax', 'entropy'],
}

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}

VAL_RATIO = 0.2 # 0.2 of each initial discovered class will be used for validation.

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
            'sample_per_class' : 30, # Number of samples per discovered class
            'num_open_classes' : 0,
        },
        # For open set learning
        'default_open_set' : {
            'num_init_classes' : 50,
            'sample_per_class' : 500,
            'num_open_classes' : 50
        },
    },
    'CIFAR10' : {
        'regular': { # The setup in ICRA deep metric learning paper. 8 closed classes, 2 open classes.
            'num_init_classes' : 4, # Number of initial discovered classes
            'sample_per_class' : 500, # Number of samples per discovered class
            'num_open_classes' : 2, # Number of open classes hold out
        },
        'fewer_class': { # Use 1/2 of discovered classes, but same number of samples per discovered class
            'num_init_classes' : 2, # Number of initial discovered classes
            'sample_per_class' : 500, # Number of samples per discovered class
            'num_open_classes' : 2, 
        },
        'fewer_sample': { # Use 1/2 of samples per discovered class, but keep 40 initial discovered classes
            'num_init_classes' : 4, # Number of initial discovered classes
            'sample_per_class' : 250, # Number of samples per discovered class
            'num_open_classes' : 2, 
        },
        'active': {
            'num_init_classes' : 10, # Number of initial discovered classes
            'sample_per_class' : 100, # Number of samples per discovered class
            'num_open_classes' : 0, 
        },
        # For open set learning
        'default_open_set' : {
            'num_init_classes' : 5,
            'sample_per_class' : 5000,
            'num_open_classes' : 5
        },
        'default_open_set_1' : {
            'num_init_classes' : 6,
            'sample_per_class' : 5000,
            'num_open_classes' : 4
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

