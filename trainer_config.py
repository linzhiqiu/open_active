from dataclasses import dataclass


class TrainConfig(object):
    """Every dict[key] can be visited by dict.key
    """

    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]


@dataclass
class OptimConfig():
    """Configuration for optimzation
    Args:
        optim (str): Optimizer choose from ['sgd', 'adam']
        weight_decay (float): L2 regularization
        momentum (float): SGD momentum
        lr (float): Learning rate 
        epochs (int): Total epochs
        decay_epochs (int): Learning rate decay after every [decay_epochs]
        decay_by (float): Learning rate decay ratio
    """
    optim: str = 'sgd'
    weight_decay: float = 0.0005
    momentum: float = 0.9
    lr: float = 0.1
    epochs: int = 200
    decay_epochs: int = 60
    decay_by: float = 0.1


@dataclass
class TrainerConfig():
    """All configuration for an experiment
    Args:
        optim_config (OptimConfig): Configuration for optimization
        backbone (str): Network architecture choose from ['ResNet18HighRes', "ResNet18']
        feature_dim (int): Final output feature dimension
        batch (int): Batch size
        workers (int): Number of workers for fetching a batch
        device (str): Device choose from ['cuda', 'cpu']
    """
    optim_config: OptimConfig
    backbone: str = 'ResNet18HighRes'
    feature_dim: int = 512
    batch: int = 128
    workers: int = 4
    device: str = 'cuda'


CIFAR_OPTIM_CONFIG = OptimConfig(
    optim='sgd',
    weight_decay=0.0005,
    momentum=0.9,
    lr=0.1,
    epochs=200,
    decay_epochs=60,
    decay_by=0.1,
    random_restart=False
)

CIFAR_TRAINER_CONFIG = TrainerConfig(
    optim_config=CIFAR_OPTIM_CONFIG,
    backbone='ResNet18',
    feature_dim=512,
    batch=128,
    workers=4,
    device='cuda'
)

CARS_OPTIM_CONFIG = OptimConfig(
    optim='sgd',
    weight_decay=0.0005,
    momentum=0.9,
    lr=0.1,
    epochs=300,
    decay_epochs=100,
    decay_by=0.1,
    random_restart=False
)

CARS_TRAINER_CONFIG = TrainerConfig(
    optim_config=CARS_OPTIM_CONFIG,
    backbone='ResNet18HighRes',
    feature_dim=512,
    batch=128,
    workers=4,
    device='cuda'
)

CUB200_OPTIM_CONFIG = OptimConfig(
    optim='sgd',
    weight_decay=0.0005,
    momentum=0.9,
    lr=0.1,
    epochs=1200,
    decay_epochs=240,
    decay_by=0.5,
    random_restart=False
)

CUB200_TRAINER_CONFIG = TrainerConfig(
    optim_config=CUB200_OPTIM_CONFIG,
    backbone='ResNet18HighRes',
    feature_dim=512,
    batch=128,
    workers=4,
    device='cuda'
)


TRAIN_CONFIG_DICT = {
    'CIFAR100': {
        'softmax_network': {
            'retrain': CIFAR_TRAINER_CONFIG,
        },
        'cosine_network': {
            'retrain': CIFAR_TRAINER_CONFIG,
        },
    },
    'CIFAR10': {
        'softmax_network': {
            'retrain': CIFAR_TRAINER_CONFIG,
        },
        'cosine_network': {
            'retrain': CIFAR_TRAINER_CONFIG,
        },
    },
    'CUB200': {
        'softmax_network': {
            'retrain': CUB200_TRAINER_CONFIG,
        },
        'cosine_network': {
            'retrain': CUB200_TRAINER_CONFIG,
        },
    },
    'Cars': {
        'softmax_network': {
            'retrain': CARS_TRAINER_CONFIG,
        },
        'cosine_network': {
            'retrain': CARS_TRAINER_CONFIG,
        },
    }
}


def get_trainer_config(data, training_method, train_mode):
    return TRAIN_CONFIG_DICT[data][training_method][train_mode]
