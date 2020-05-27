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

CIFAR100_NO_FINETUNE_CONFIG = CIFAR100_DEFAULT_CONFIG.copy()
CIFAR100_NO_FINETUNE_CONFIG['finetune'] = TrainConfig({
                                              'optim' : 'sgd',
                                              'weight_decay' : 0.0005,
                                              'momentum' : 0.9,
                                              'lr' : 0.1,
                                              'epochs' : 200,
                                              'decay_epochs' : 60,
                                              'decay_by' : 0.1
                                          })

CUB200_DEFAULT_CONFIG = {
    'backbone' : 'ResNet18HighRes',
    'feature_dim' : 512,
    'batch' : 128,
    'workers' : 4,
    'device' : 'cuda',
    'train' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 400,
        'decay_epochs': 120,
        'decay_by' : 0.1,
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.01,
        'epochs' : 280,
        'decay_epochs': 120,
        'decay_by' : 0.1,
    }),
}

CUB200_DEFAULT_CONFIG_FOR_DEEPMETRIC = {
    'backbone' : 'ResNet18HighRes',
    'feature_dim' : 512,
    'batch' : 128,
    'workers' : 4,
    'device' : 'cuda',
    'train' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 400,
        'decay_epochs': 120,
        'decay_by' : 0.1,
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.01,
        'epochs' : 280,
        'decay_epochs': 120,
        'decay_by' : 0.1,
    }),
}

TRAIN_CONFIG_DICT = {
    'CIFAR100' : {
        'softmax_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'no_finetune': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
        },
        'cosine_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'no_finetune': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
        },
        'sigmoid_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'no_finetune': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
        },
    },
    'CUB200' : {
        'softmax_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
        },
        'cosine_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
        },
        'sigmoid_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
        },
        'deep_metric' : {
            'default' : CUB200_DEFAULT_CONFIG_FOR_DEEPMETRIC,
        },
    }
}

def get_trainer_config(data, training_method, train_mode):
    return TRAIN_CONFIG_DICT[data][training_method][train_mode]