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