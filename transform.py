import torchvision
import torchvision.transforms as transforms


def get_transform_dict(data):
    """Return a dictionary with values being a torchvision.transforms object

    Args:
        data (str): Dataset

    Raises:
        ValueError: If dataset is not yet implemented

    Returns:
        dict: A dict with two keys 'train' and 'test' with corresponding value being the transform object.
    """    
    if data in ["CIFAR10", "CIFAR100"]:
        transform_train, transform_test = _get_cifar_transform()
    elif data in ['CUB200', 'Cars']:
        transform_train, transform_test = _get_imagenet_transform()
    else:
        raise ValueError("Dataset not supported")

    transforms_dict = {'train': transform_train,
                       'test': transform_test}
    return transforms_dict


def _get_imagenet_transform():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


def _get_cifar_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test
