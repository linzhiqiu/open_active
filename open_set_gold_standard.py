from utils import get_subset_dataloaders, get_target_mapping_func, get_subset_loader
from dataset_factory import get_dataset_factory, select_init_train_set
import torch
from torch import nn
from models.resnet import ResNet50
import time


DATA = 'CIFAR100'
DATA_PATH = './data/'
INIT_MODE = 'open_set_leave_one_out'
BATCH_SIZE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.1
NUM_EPOCHS = 50


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        [acc1] = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    progress.display(i)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            [acc1] = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=top1))

    return top1.avg


def main():
    dataset_factory = get_dataset_factory(DATA, DATA_PATH, INIT_MODE)
    train_dataset, test_dataset = dataset_factory.get_dataset()
    train_samples, train_labels, classes = dataset_factory.get_train_set_info()
    test_samples, test_labels, _ = dataset_factory.get_test_set_info()
    s_train, open_samples, seen_classes, open_classes = dataset_factory.get_init_train_set()
    s_test, _, _, _ = select_init_train_set(
        DATA, INIT_MODE, test_labels, classes)

    train_loader = get_subset_loader(train_dataset,
                                     s_train,
                                     get_target_mapping_func(
                                         classes, seen_classes, open_classes),
                                     batch_size=BATCH_SIZE,
                                     workers=4,
                                     shuffle=True)

    test_loader = get_subset_loader(test_dataset,
                                    s_test,
                                    get_target_mapping_func(
                                        classes, seen_classes, open_classes),
                                    batch_size=BATCH_SIZE,
                                    workers=4,
                                    shuffle=True)



    model = ResNet50().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, criterion, optimizer, epoch)
        validate(test_loader, model, criterion)


if __name__ == "__main__":
    main()
