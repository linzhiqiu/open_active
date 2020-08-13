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


