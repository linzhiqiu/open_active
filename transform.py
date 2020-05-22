import torchvision
import torchvision.transforms as transforms

def get_transform_dict(data):
    if data in ["CIFAR10", "CIFAR100"]:
        transform_train, transform_test = get_cifar_transform()
    elif data in ['CUB200']:
        transform_train, transform_test = get_ImageNet_transform()
    else:
        raise ValueError("Dataset not supported")

    transforms_dict = {'train' : transform_train,
                       'test' : transform_test}
    return transforms_dict

def get_ImageNet_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    return transform_train, transform_test


def get_cifar_transform():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test

def get_dcgan_transform():
    # transform = transforms.Compose([
    #     transforms.Resize(64),
    #     transforms.CenterCrop(64),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    # ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform



