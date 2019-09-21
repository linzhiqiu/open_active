import torchvision
import torchvision.transforms as transforms

def get_transform_dict(data):
    assert data in ["CIFAR10", "CIFAR100"]
    if data in ["CIFAR10", "CIFAR100"]:
        transform_train, transform_test = get_cifar_transform()
    else:
        raise ValueError("Dataset not supported")

    transforms_dict = {'train' : transform_train,
                       'test' : transform_test}
    return transforms_dict


def get_cifar_transform():
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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



