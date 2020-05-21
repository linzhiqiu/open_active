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
    'train' : TrainConfig({
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
        'lr' : 0.01,
        'epochs' : 140,
        'decay_epochs': 60,
        'decay_by' : 0.1,
    }),
}

TRAIN_CONFIG_DICT = {
    'CIFAR100' : {
        'softmax_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
        },
        'cosine_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
        },
        'sigmoid_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
        },
    }
}

def get_trainer_config(data, training_method, train_mode):
    return TRAIN_CONFIG_DICT[data][training_method][train_mode]