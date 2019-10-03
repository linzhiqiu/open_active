import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from trainer_machine import Network, train_epochs
import argparse
from dataset_factory import get_dataset_factory
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func, get_target_unmapping_dict


NUM_CLASSES = 100
BATCH_SIZE = 32
ALPHA = 0.1

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ll_gap1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ll_fc1 = nn.Linear(64, 128)
        self.ll_relu1 = nn.ReLU()
        self.ll_gap2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ll_fc2 = nn.Linear(128, 128)
        self.ll_relu2 = nn.ReLU()
        self.ll_gap3 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ll_fc3 = nn.Linear(256, 128)
        self.ll_relu3 = nn.ReLU()
        self.ll_gap4 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.ll_fc4 = nn.Linear(512, 128)
        self.ll_relu4 = nn.ReLU()

        self.ll_fc = nn.Linear(128*4, 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(x.shape)
        ll1 = self.ll_gap1(x)
        ll1 = ll1.squeeze()
        # print(ll1.shape)
        ll1 = self.ll_fc1(ll1)
        ll1 = self.ll_relu1(ll1)

        x = self.layer2(x)
        ll2 = self.ll_gap2(x)
        ll2 = ll2.squeeze()
        # print(ll2.shape)
        ll2 = self.ll_fc2(ll2)
        ll2 = self.ll_relu2(ll2)

        x = self.layer3(x)
        ll3 = self.ll_gap3(x)
        ll3 = ll3.squeeze()
        # print(ll3.shape)
        ll3 = self.ll_fc3(ll3)
        ll3 = self.ll_relu3(ll3)

        x = self.layer4(x)
        ll4 = self.ll_gap4(x)
        ll4 = ll4.squeeze()
        # print(ll4.shape)
        ll4 = self.ll_fc4(ll4)
        ll4 = self.ll_relu4(ll4)

        # print(ll1.shape,ll2.shape,ll3.shape,ll4.shape)
        ll = torch.cat((ll1, ll2, ll3, ll4), dim=1)
        # print(ll.shape)
        ll = self.ll_fc(ll)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, ll


def _resnet(arch, block, layers, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)


def resnet34(progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], progress,
                   **kwargs)


def resnet50(progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], progress,
                   **kwargs)


def resnet101(progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], progress,
                   **kwargs)


def resnet152(progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], progress,
                   **kwargs)


def resnext50_32x4d(progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   progress, **kwargs)


def resnext101_32x8d(progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   progress, **kwargs)


def wide_resnet50_2(progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   progress, **kwargs)


def wide_resnet101_2(progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   progress, **kwargs)

class LearningLoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.resnet = resnet18(num_classes=NUM_CLASSES)

    def forward(self, x):
        x, ll = self.resnet(x, ll)


def learning_loss_function(loss_truth, loss_pred, epsilon=1):
    """
    Loss function for the loss prediction module
    Equation 2 in paper Learning Loss for Active Learning
    """
    # TODO: Implement the loss
    loss = nn.MSELoss()
    return loss(loss_truth, loss_pred)

def evaluation(eval_dataset, model, criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    eval_data = torch.utils.data.DataLoader(eval_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    with torch.no_grad():
        for batch_data, batch_label in eval_data:
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            output, ll = model(batch_data)
            pred = F.log_softmax(output, dim=1)
            eval_loss += F.cross_entropy(pred, batch_label, reduction='sum').item()
            pred = pred.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(batch_label.view_as(pred)).sum().item()
    eval_loss /= len(eval_dataset)
    accuracy = correct / len(eval_dataset)

    print(f"Eval accuracy={accuracy}, loss={eval_loss}")

def trainer(train_dataset, model, criterion):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_data = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    # Initial 10000 training samples
    for epoch in range(300):
        for batch_idx, (batch_data, batch_label) in enumerate(train_data):
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            optimizer.zero_grad()

            output, ll = model(batch_data)
            # print(output.shape)
            # print(ll.shape)
            pred = F.log_softmax(output, dim=1)
            loss = criterion(pred, batch_label)
            # print(loss, loss.shape)
            # print(loss, loss.shape)
            learning_loss = learning_loss_function(loss, ll)
            # print(learning_loss, learning_loss.shape)

            total_loss = ALPHA * learning_loss + (1-ALPHA) * loss
            
            total_loss.backward()
            optimizer.step()

            if batch_idx >= int(10000/BATCH_SIZE):
                break
        print(f"Train epoch={epoch}, loss={loss}, learning_loss={learning_loss}")

        # TODO

if __name__ == "__main__":
    dataset_factory = get_dataset_factory("CIFAR100", "./data", "default")
    train_dataset, test_dataset = dataset_factory.get_dataset()
    train_samples, train_labels, classes = dataset_factory.get_train_set_info()

    device = torch.device("cuda")
    model = resnet18()
    criterion = nn.CrossEntropyLoss().cuda()
    model.cuda()
    trainer(train_dataset, model, criterion)
    evaluation(test_dataset, model, criterion)
