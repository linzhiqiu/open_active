OPEN_CLASS_INDEX = -2 # The index for hold out open set class examples
UNDISCOVERED_CLASS_INDEX = -1 # The index for unseen open set class examples

# Open set methods applicable to each training method
OPEN_SET_METHOD_DICT = {
    # 'softmax_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax', 'c2ae'],
    # 'cosine_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax', 'c2ae'],
    'softmax_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
    'cosine_network' : ['nn','nn_cosine', 'softmax', 'entropy', 'openmax'],
}

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}


def get_budget_list_from_config(config, makedir=True):
    return get_budget_list(config.data)


def get_budget_list(data):
    """Return the list of budget candidates for active learning experiments.
        Args:
            data (str) - dataset name
        Returns:
            budget_list (list) : List of int
    """
    if data in ['CIFAR10']:
        return [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    elif data in ['CIFAR100']:
        return [0, 3000, 6000, 9000, 12000, 15000]
    else:
        raise NotImplementedError()
