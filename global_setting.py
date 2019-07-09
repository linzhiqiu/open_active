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
    }
}