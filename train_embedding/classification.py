import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms, datasets

from tqdm import tqdm
import argparse

import os
import numpy as np

import utils
import utils_local

import models
import time

import wandb
import warnings

from dataset_factory import DatasetFactory
import global_setting


def get_target_mapping_func_for_tensor(classes,
                                       discovered_classes,
                                       open_classes,
                                       OPEN_CLASS_INDEX=global_setting.OPEN_CLASS_INDEX,
                                       UNDISCOVERED_CLASS_INDEX=global_setting.UNDISCOVERED_CLASS_INDEX,
                                       device='cuda'):
    """Exactly the same as get_target_mapping_func but the returning function operates on tensor level
        Returns:
            target_mapping_func (fun tensor -> tensor) : As specified.
    """
    discovered_classes = sorted(list(discovered_classes))
    open_classes = sorted(list(open_classes))
    mapping = {idx: global_setting.OPEN_CLASS_INDEX if idx in open_classes else
               global_setting.UNDISCOVERED_CLASS_INDEX if idx not in discovered_classes else
               discovered_classes.index(idx)
               for idx in classes}
    index_tensor = torch.zeros((len(classes))).long().to(device)
    for idx in classes:
        index_tensor[idx] = mapping[idx]

    def mapp_func(real_labels):
        return index_tensor[real_labels]
    return mapp_func, mapping


def main(args):
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    logger = utils_local.create_logger(os.path.join(
        args.dir, 'checkpoint.log'), __name__)
    trainlog = utils_local.savelog(args.dir, 'train')
    # vallog = utils_local.savelog(args.dir, 'val')
    # testlog = utils_local.savelog(args.dir, 'test')

    wandb.init(project='open set active learning',
               group=__file__,
               name=f'{__file__}_{args.dir}')

    wandb.config.update(args)

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    ###########################
    # Create DataLoader
    ###########################
    dataset_factory = DatasetFactory(args.data, args.download_path, args.init_mode)
    train_dataset, test_dataset = dataset_factory.get_dataset()  # The pytorch datasets
    # List of indices/labels
    train_samples, train_labels = dataset_factory.get_train_set_info()
    classes, open_classes = dataset_factory.get_class_info()  # Set of indices

    # Begin from scratch
    # Get initial training set, discovered classes
    discovered_samples, discovered_classes = dataset_factory.get_init_train_set()
    num_discovered_classes = len(discovered_classes)
    open_samples = dataset_factory.get_open_samples_in_trainset()

    target_mapping, target_dict = get_target_mapping_func_for_tensor(classes,
                                       discovered_classes,
                                       open_classes)
    trainloader = utils.get_subset_loader(train_dataset, discovered_samples,
                    None, args.bsize, workers=args.num_workers, shuffle=True)

    ############################

    ###########################
    # Create Models
    ###########################
    if args.model == 'resnet18':
        backbone = models.ResNet18(
            last_relu=(not args.remove_last_relu))
        feature_dim = 512
    else:
        raise ValueError('Invalid backbone model')

    backbone = backbone.cuda()

    if args.use_cosine_clf:
        if args.model=='resnet18' and not args.remove_last_relu:
            warnings.warn(
                "Using cosine classifier without the last relu activation removed!")
        clf = models.cosine_clf(feature_dim, num_discovered_classes).cuda()
    else:
        if args.model == 'resnet18' and args.remove_last_relu:
            warnings.warn(
                "Using linear classifier with the last relu activation removed!")
        clf = nn.Linear(feature_dim, num_discovered_classes).cuda()
    ############################

    ###########################
    # Create Optimizer
    ###########################
    if args.no_clf_wd:
        clf_wd = 0
    else:
        clf_wd = args.wd

    optimizer = torch.optim.SGD([
        {'params': backbone.parameters()},
        {'params': clf.parameters(), 'weight_decay': clf_wd}
    ],
        lr=args.lr, momentum=0.9,
        weight_decay=args.wd,
        nesterov=False)
    ############################

    ############################
    # Specify Loss Function
    ############################
    criterion = nn.NLLLoss(reduction='mean')
    ############################

    starting_epoch = 0

    if args.load_path is not None:
        print('Loading model from {}'.format(args.load_path))
        logger.info('Loading model from {}'.format(args.load_path))
        starting_epoch = load_checkpoint(
            backbone, clf, optimizer, args.load_path)

    if args.resume_latest:
        # Only works if model is saved as checkpoint_(/d+).pkl
        import re
        pattern = "checkpoint_(\d+).pkl"
        candidate = []
        for i in os.listdir(args.dir):
            match = re.search(pattern, i)
            if match:
                candidate.append(int(match.group(1)))

        # if nothing found, then start from scratch
        if len(candidate) == 0:
            print('No latest candidate found to resume!')
            logger.info('No latest candidate found to resume!')
        else:
            latest = np.amax(candidate)
            load_path = os.path.join(args.dir, f'checkpoint_{latest}.pkl')
            if latest >= args.epochs:
                print('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                logger.info('The latest checkpoint found ({}) is after the number of epochs (={}) specified! Exiting!'.format(
                    load_path, args.epochs))
                import sys
                sys.exit(0)
            else:
                print('Resuming from the latest checkpoint: {}'.format(load_path))
                logger.info(
                    'Resuming from the latest checkpoint: {}'.format(load_path))
                if args.load_path:
                    logger.info(
                        'Overwriting model loaded from {}'.format(args.load_path))
                starting_epoch = load_checkpoint(
                    backbone, clf, optimizer, load_path)

    # save the initialization
    checkpoint(backbone, clf, optimizer, target_dict, discovered_samples,
    os.path.join(
        args.dir, f'checkpoint_{starting_epoch}.pkl'), starting_epoch)

    sd = torch.load(os.path.join(
        args.dir, f'checkpoint_{starting_epoch}.pkl'))

    try:
        for epoch in tqdm(range(starting_epoch, args.epochs)):
            perf = train(backbone, clf, optimizer, trainloader, target_mapping, criterion,
                         epoch, args.epochs, logger, trainlog, args)

            # Always checkpoint after first epoch of training
            if (epoch == starting_epoch) or ((epoch + 1) % args.save_freq == 0):
                checkpoint(backbone, clf, optimizer,  target_dict, discovered_samples,
                os.path.join(
                    args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)

            if args.fail_epoch is not None and (epoch+1) == args.fail_epoch:
                top1 = perf['top1/avg']

                if top1 < (args.fail_threshold):
                    logger.info('Top1 Performance at epoch {:d}: {:.4f} is less than {:.4f}'.format
                                (args.fail_epoch, top1, args.fail_threshold))
                    raise RuntimeError('Top1 Performance at epoch {:d}: {:.4f} is less than {:.4f}'.format
                                       (args.fail_epoch, top1, args.fail_threshold))

        if (epoch + 1) % args.save_freq != 0:
            checkpoint(backbone, clf, optimizer, target_dict, discovered_samples, os.path.join(
                args.dir, f'checkpoint_{epoch + 1}.pkl'), epoch + 1)
    finally:
        trainlog.save()
    return


def checkpoint(model, clf, optimizer, target_dict, discovered_samples, save_path, epoch):
    '''
    epoch: the number of epochs of training that has been done
    Should resume from epoch
    '''
    torch.save({
        'model': model.state_dict(),
        'clf': clf.state_dict(),
        'opt': optimizer.state_dict(),
        'target_mapping': target_dict,  
        'discovered_samples': discovered_samples,
        'epoch': epoch
    }, save_path)
    return


def load_checkpoint(model, clf, optimizer, load_path):
    '''
    Load model and optimizer from load path 
    Return the epoch to continue the checkpoint
    '''
    sd = torch.load(load_path)
    model.load_state_dict(sd['model'])
    clf.load_state_dict(sd['clf'])
    optimizer.load_state_dict(sd['opt'])
    return sd['epoch']


def lr_schedule_step(optimizer, epoch, step_in_epoch, total_steps_in_epochs, args):
    def step_rampdown(epoch):
        return 0.1 ** (epoch // args.lr_decay_freq)

    lr = step_rampdown(epoch) * args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, clf, optimizer, trainloader, target_mapping, criterion, epoch, num_epochs, logger, trainlog, args):
    meters = utils_local.AverageMeterSet()
    model.train()
    clf.train()

    end = time.time()
    for i, (X, y) in enumerate(trainloader):
        meters.update('Data_time', time.time() - end)

        lr_schedule_step(optimizer, epoch, i, len(trainloader), args)
        
        X = X.cuda()
        y = y.cuda()
        y = target_mapping(y)

        optimizer.zero_grad()
        features = model(X)
        logits = clf(features)
        # print('Logits.shape', logits.shape)
        log_probability = F.log_softmax(logits, dim=1)

        loss = criterion(log_probability, y)

        loss.backward()
        optimizer.step()

        meters.update('Loss', loss.item(), 1)

        perf = utils_local.accuracy(logits.data,
                              y.data, topk=(1, 5))

        meters.update('top1', perf['average'][0].item(), len(X))
        meters.update('top5', perf['average'][1].item(), len(X))

        meters.update('top1_per_class', perf['per_class_average'][0].item(), 1)
        meters.update('top5_per_class', perf['per_class_average'][1].item(), 1)

        meters.update('Batch_time', time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            values = meters.values()
            averages = meters.averages()
            sums = meters.sums()

            logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step} / {steps}] Batch Time: {meters[Batch_time]:.4f} '
                             'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                             'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
                             'Top1_per_class: {meters[top1_per_class]:.4f} '
                             'Top5_per_class: {meters[top5_per_class]:.4f} ').format(
                epoch=epoch, epochs=num_epochs, step=i+1, steps=len(trainloader), meters=meters)

            logger.info(logger_string)

        if (args.iteration_bp is not None) and (i+1) == args.iteration_bp:
            break

    logger_string = ('Training Epoch: [{epoch}/{epochs}] Step: [{step}] Batch Time: {meters[Batch_time]:.4f} '
                     'Data Time: {meters[Data_time]:.4f} Average Loss: {meters[Loss]:.4f} '
                     'Top1: {meters[top1]:.4f} Top5: {meters[top5]:.4f} '
                     'Top1_per_class: {meters[top1_per_class]:.4f} '
                     'Top5_per_class: {meters[top5_per_class]:.4f} ').format(
        epoch=epoch+1, epochs=num_epochs, step=0, meters=meters)

    logger.info(logger_string)

    values = meters.values()
    averages = meters.averages()
    sums = meters.sums()

    trainlog.record(epoch, {
        **values,
        **averages,
        **sums
    })

    wandb.log({'loss': averages['Loss/avg']}, step=epoch+1)
    wandb.log({'top1': averages['top1/avg'],
               'top5': averages['top5/avg'],
               'top1_per_class': averages['top1_per_class/avg'],
               'top5_per_class': averages['top5_per_class/avg']}, step=epoch+1)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to Training a Image Classification Model')
    parser.add_argument('--dir', type=str, default='.',
                        help='directory to save the checkpoints')
    parser.add_argument('--download_path',
                        default="./data", metavar='PATH',
                        help='path to datasets location default :%(default)')
    parser.add_argument('--data',
                        default="CIFAR100",
                        choices=["CIFAR100", "MNIST", "IMAGENET12",
                                 "TINY-IMAGENET", "CIFAR10"],
                        help='Choice of dataset')
    parser.add_argument('--init_mode',
                        default='regular',
                        type=str,
                        help='Data Factory mode')
    parser.add_argument('--bsize', type=int, default=32,
                        help='batch_size for labeled data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Frequency (in epoch) to save')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Frequency (in epoch) to evaluate on testset')
    parser.add_argument('--print_freq', type=int, default=5,
                        help='Frequency (in step per epoch) to print training stats')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to the checkpoint to be loaded')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for randomness')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for model')
    parser.add_argument('--lr_decay_freq', type=int, default=10,
                        help='learning rate decay frequency (in epochs)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay for the model')
    parser.add_argument('--resume_latest', action='store_true',
                        help='resume from the latest model in args.dir')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--fail_epoch', type=int, default=None,
                        help='Number of epochs to quit the training. This is useful when tuning hyperparameter!')
    parser.add_argument('--fail_threshold', type=float, default=3,
                        help='threshold for determining failure')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Backbone model')
    parser.add_argument('--width', type=int, default=64,
                        help='Width of the 4-layer')
    parser.add_argument('--remove_last_relu', action='store_true', 
                        help='whether to remove the last relu activation for resnet')
    parser.add_argument('--use_cosine_clf', action='store_true',
                        help='whether to use cosine classifier')
    parser.add_argument('--no_clf_wd', action='store_true',
                        help='turn off the weight decay for classifier')
    parser.add_argument('--iteration_bp', type=int,
                        help='number of iteration to break the training loop. Useful for debugging')
    args = parser.parse_args()
    main(args)
