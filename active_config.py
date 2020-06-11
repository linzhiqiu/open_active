class TrainConfig(object):
    """Every dict[key] can be visited by dict.key
    """
    def __init__(self, dict):
        self.dict = dict
    
    def __getattr__(self, name):
        return self.dict[name]


CIFAR100_DEFAULT_CONFIG = {
    'backbone' : 'ResNet18',
    'feature_dim' : 512,
    'batch' : 128,
    'workers' : 4,
    'device' : 'cuda',
    'train' : TrainConfig({# Just a place holder. Won't be used.
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    }),
    'finetune' : TrainConfig({ 
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 200,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    }),
}

CIFAR10_DEFAULT_CONFIG = CIFAR100_DEFAULT_CONFIG.copy()


TRAIN_CONFIG_DICT = {
    'CIFAR100' : {
        'softmax_network' : {
            'retrain' : CIFAR100_DEFAULT_CONFIG,
        },
        'cosine_network' : {
            'retrain' : CIFAR100_DEFAULT_CONFIG,
        },
        'deep_metric' : {
            'retrain' : None,
        },
    },
    'CIFAR10' : {
        'softmax_network' : {
            'retrain' : CIFAR10_DEFAULT_CONFIG,
        },
        'cosine_network' : {
            'retrain' : CIFAR10_DEFAULT_CONFIG,
        },
        'deep_metric' : {
            'retrain' : None,
        },
    },
}

def get_active_learning_config(data, training_method, active_train_mode):
    return TRAIN_CONFIG_DICT[data][training_method][active_train_mode]