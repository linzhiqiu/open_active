class TrainConfig(object):
    """Every dict[key] can be visited by dict.key
    """
    def __init__(self, dict):
        self.dict = dict
    
    def __getattr__(self, name):
        return self.dict[name]

CARS_DEFAULT_300EPS_CONFIG = {
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
        'epochs' : 300,
        'decay_epochs': 100,
        'decay_by' : 0.1,
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        'epochs' : 300,
        'decay_epochs': 100,
        'decay_by' : 0.1,
    }),
}

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
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
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

CIFAR10_NO_FINETUNE_CONFIG = CIFAR100_NO_FINETUNE_CONFIG.copy()

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
        'lr' : 0.01,
        # 'epochs' : 400,
        'epochs' : 1200,
        'decay_epochs': 240,
        'decay_by' : 0.5,
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.01,
        # 'epochs' : 400,
        'epochs' : 1200,
        'decay_epochs': 240,
        'decay_by' : 0.5,
    }),
}

CUB200_FEWER_EPOCH_CONFIG = {
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
        # 'epochs' : 400,
        'epochs' : 300,
        'decay_epochs': 100,
        'decay_by' : 0.1,
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.1,
        # 'epochs' : 400,
        'epochs' : 300,
        'decay_epochs': 100,
        'decay_by' : 0.1,
    }),
}

CIFAR100_FINETUNE_CONFIG_FOR_DEEPMETRIC = {
    'backbone' : 'ResNet18',
    'feature_dim' : 512,
    'batch' : 128,
    'workers' : 4,
    'device' : 'cuda',
    'num_neighbours': 200,
    'sigma': 10,
    'interval': 5,
    'train' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 1e-4,
        'epochs' : 20,
        'decay_epochs': 30,
        'decay_by' : 0.1,
        'softmax_optim' : 'sgd',
        'softmax_epochs': 0,
        'softmax_lr': 1e-1,
        'softmax_decay_epochs':0,
        'softmax_decay_by': 0.1,
        'softmax_weight_decay' : 0.0005,
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 1e-4,
        'epochs' : 20,
        'decay_epochs': 30,
        'decay_by' : 0.1,
        'softmax_optim' : 'sgd',
        'softmax_epochs': 0,
        'softmax_lr': 1e-1,
        'softmax_decay_epochs':60,
        'softmax_decay_by': 0.1,
        'softmax_weight_decay' : 0.0005,
    }),
}

CIFAR100_DEFAULT_CONFIG_FOR_DEEPMETRIC = {
    'backbone' : 'ResNet18',
    'feature_dim' : 512,
    'batch' : 128,
    'workers' : 4,
    'device' : 'cuda',
    'num_neighbours': 200,
    'sigma': 10,
    'interval': 5,
    'train' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 1e-4,
        'epochs' : 20,
        'decay_epochs': 30,
        'decay_by' : 0.1,
        'softmax_optim' : 'sgd',
        'softmax_epochs': 0,
        'softmax_lr': 1e-1,
        'softmax_decay_epochs':0,
        'softmax_decay_by': 0.1,
        'softmax_weight_decay' : 0.0005,
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
    }),
    'finetune' : TrainConfig({
        'optim' : 'sgd',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 1e-4,
        'epochs' : 20,
        'decay_epochs': 30,
        'decay_by' : 0.1,
        'softmax_optim' : 'sgd',
        'softmax_epochs': 200,
        'softmax_lr': 1e-1,
        'softmax_decay_epochs':60,
        'softmax_decay_by': 0.1,
        'softmax_weight_decay' : 0.0005,
    }),
}

CUB200_DEFAULT_CONFIG_FOR_DEEPMETRIC = {
    'backbone' : 'ResNet18HighRes',
    'feature_dim' : 512,
    'batch' : 20,
    'workers' : 4,
    'device' : 'cuda',
    'num_neighbours': 200,
    'sigma': 10,
    'interval': 5,
    'train' : TrainConfig({
        'optim' : 'adam',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.00001,
        'epochs' : 400,
        'decay_epochs': 120,
        'decay_by' : 0.1,
        'random_restart' : False, # For each retrain round, whether or not to use a fresh random initialization. If False, then use the same initialization each round (loading from the same checkpoint location).
    }),
    'finetune' : TrainConfig({
        'optim' : 'adam',
        'weight_decay' : 0.0005,
        'momentum' : 0.9,
        'lr' : 0.00001,
        'epochs' : 280,
        'decay_epochs': 120,
        'decay_by' : 0.1,
    }),
}

TRAIN_CONFIG_DICT = {
    'CIFAR100' : {
        'softmax_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'retrain': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
            'fix_feature_extractor' : CIFAR100_NO_FINETUNE_CONFIG,
        },
        'cosine_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'retrain': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
            'fix_feature_extractor' : CIFAR100_NO_FINETUNE_CONFIG,
        },
        'sigmoid_network' : {
            'default' : CIFAR100_DEFAULT_CONFIG,
            'retrain': CIFAR100_NO_FINETUNE_CONFIG,
            'default_lr01_200eps' : CIFAR100_NO_FINETUNE_CONFIG,
        },
        'deep_metric' : {
            'default_lr01_200eps' : CIFAR100_DEFAULT_CONFIG_FOR_DEEPMETRIC,
            'default' : CIFAR100_FINETUNE_CONFIG_FOR_DEEPMETRIC,
        },
    },
    'CIFAR10' : {
        'softmax_network' : {
            'retrain': CIFAR10_NO_FINETUNE_CONFIG,
        },
        'cosine_network' : {
            'retrain': CIFAR10_NO_FINETUNE_CONFIG,
        },
        'sigmoid_network' : {
            'retrain': CIFAR10_NO_FINETUNE_CONFIG,
        },
        # 'deep_metric' : {
        #     'default_lr01_200eps' : CIFAR100_DEFAULT_CONFIG_FOR_DEEPMETRIC,
        #     'default' : CIFAR100_FINETUNE_CONFIG_FOR_DEEPMETRIC,
        # },
    },
    'CUB200' : {
        'softmax_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
            'default_300eps' : CUB200_FEWER_EPOCH_CONFIG,
        },
        'cosine_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
            'default_300eps' : CUB200_FEWER_EPOCH_CONFIG,
        },
        'sigmoid_network' : {
            'default' : CUB200_DEFAULT_CONFIG,
            'default_300eps' : CUB200_FEWER_EPOCH_CONFIG,
        },
        'deep_metric' : {
            'default' : CUB200_DEFAULT_CONFIG_FOR_DEEPMETRIC,
        },
    },
    'Cars' : {
        'softmax_network' : {
            'default_300eps' : CARS_DEFAULT_300EPS_CONFIG,
        },
        'cosine_network' : {
            'default_300eps' : CARS_DEFAULT_300EPS_CONFIG,
        },
    }
}

def get_trainer_config(data, training_method, train_mode):
    return TRAIN_CONFIG_DICT[data][training_method][train_mode]