import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm

import models
from utils import get_subset_dataloaders, get_test_loader, SetPrintMode

PRETRAINED_MODEL_PATH = {
    'CIFAR10' : {
        'ResNet50' : "./downloaded_models/resnet101/ckpt.t7", # Run pytorch_cifar.py for 200 epochs
    }
}

def get_target_mapping_func(classes, seen_classes):
    """ Return a function that map seen_classes indices to 0-len(seen_classes)
    """
    seen_classes = list(seen_classes)
    mapping = {idx : -1 if idx not in seen_classes else seen_classes.index(idx)
               for idx in classes}
    return lambda idx : mapping[idx]

class TrainerMachine(object):
    """Abstract class"""
    def __init__(self, config, train_instance):
        super(TrainerMachine, self).__init__()
        self.config = config
        self.train_instance = train_instance

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def train_new_round(self, s_train, seen_classes):
        raise NotImplementedError()

    def eval(self, test_dataset, seen_classes):
        raise NotImplementedError()

    def train_mode(self):
        raise NotImplementedError()

    def eval_mode(self):
        raise NotImplementedError()


class Network(TrainerMachine):
    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.epoch = 0
        self.max_epochs = self.config.epochs
        self.device = self.config.device
        self.model = self._get_network_model()
        self.optimizer = self._get_network_optimizer()
        self.scheduler = self._get_network_scheduler()

    def get_checkpoint(self):
        return {
            'epoch' : self.epoch,
            'arch' : self.config.arch,
            'model' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict()
        }

    def load_checkpoint(self, checkpoint):
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def train_new_round(self, s_train, seen_classes, start_epoch=0):
        self._train_mode()
        with SetPrintMode(hidden=not self.config.verbose):
            target_mapping_func = get_target_mapping_func(self.train_instance.classes,
                                                          seen_classes)
            
            dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                 list(s_train),
                                                 target_mapping_func,
                                                 batch_size=self.config.batch,
                                                 workers=self.config.workers)
            
            self._update_fc_layer(len(seen_classes))
            self.optimizer = self._get_network_optimizer()
            self.scheduler = self._get_network_scheduler()

            criterion = nn.CrossEntropyLoss()
            avg_loss = 0.
            avg_acc = 0.

            for epoch in range(start_epoch, self.max_epochs):
                print('Epoch {}/{}'.format(epoch, self.max_epochs - 1))
                print('-' * 10)

                for phase in dataloaders.keys():
                    if phase == "train":
                        self.scheduler.step()
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0.
                    count = 0

                    if self.config.verbose:
                        pbar = tqdm(dataloaders[phase], ncols=80)
                    else:
                        pbar = dataloaders[phase]

                    for batch, data in enumerate(pbar):
                        inputs, labels = data
                        count += inputs.size(0)
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        if self.config.verbose:
                            pbar.set_postfix(loss=running_loss/count, 
                                             acc=float(running_corrects)/count)

                    avg_loss = running_loss/count
                    avg_acc = float(running_corrects)/count
                    print(f"Epoch {epoch} => "
                          f"Loss {avg_loss}, Accuracy {avg_acc}")
                print()
        print(f"Train => {self.max_epochs} epochs => "
              f"Loss {avg_loss}, Accuracy {avg_acc}")

    def eval(self, test_dataset, seen_classes):
        self._eval_mode()
        with SetPrintMode(hidden=not self.config.verbose):
            target_mapping_func = get_target_mapping_func(self.train_instance.classes,
                                                          seen_classes)
            dataloader = get_test_loader(test_dataset,
                                         target_mapping_func,
                                         batch_size=self.config.batch,
                                         workers=self.config.workers)
            # running_loss = 0.0

            open_set_criterion = self._get_open_set_crit_func()
            open_set_prediction = self._get_open_set_pred_func()

            if self.config.verbose:
                pbar = tqdm(dataloader, ncols=80)
            else:
                pbar = dataloader

            performance_dict = {'acc' : {'corrects' : 0., 'count' : 0.},
                                'open_acc' : {'corrects' : 0., 'count' : 0.},
                                'mult_acc' : {'corrects' : 0., 'count' : 0.}}

            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, labels = data
                    performance_dict['acc']['count'] += inputs.size(0)
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    preds = open_set_prediction(outputs)
                    # loss = open_set_criterion(outputs, labels)

                    # statistics
                    # running_loss += loss.item() * inputs.size(0)
                    performance_dict['acc']['corrects'] += torch.sum(preds == labels.data)
                    open_indices = labels == -1
                    mult_indices = labels > -1
                    performance_dict['open_acc']['count'] += float(torch.sum(open_indices))
                    performance_dict['mult_acc']['count'] += float(torch.sum(mult_indices))
                    performance_dict['open_acc']['corrects'] += torch.sum(
                                                                    torch.masked_select(
                                                                        (preds==labels.data),
                                                                        open_indices
                                                                    )
                                                                ).float()
                    performance_dict['mult_acc']['corrects'] += torch.sum(
                                                                    torch.masked_select(
                                                                        (preds==labels.data),
                                                                        mult_indices
                                                                    )
                                                                ).float()

                    batch_result = get_acc_from_performance_dict(performance_dict)
                    if self.config.verbose:
                        pbar.set_postfix(batch_result)

                epoch_result = get_acc_from_performance_dict(performance_dict)
                acc = epoch_result['acc']
                multi_class_acc = epoch_result['mult_acc']
                open_set_acc = epoch_result['open_acc']
        print(f"Test => "
              f"Overall Acc {acc}, "
              f"Open-S Acc {open_set_acc}, Mul-C Acc {multi_class_acc}")
        return multi_class_acc, open_set_acc

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode == 'threshold'
        if self.config.network_eval_mode == 'threshold':
            threshold = self.config.network_eval_threshold
            def open_set_prediction(outputs):
                softmax_outputs = F.softmax(outputs, dim=1)
                softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
                preds = torch.where(softmax_max < threshold, 
                                    torch.LongTensor([-1]).to(outputs.device), 
                                    softmax_preds)
                return preds
            return open_set_prediction
        else:
            raise NotImplementedError()

    def _get_open_set_crit_func(self):
        print('Ignore open set samples. TODO: Define a better criterion')
        def open_set_criterion(outputs, labels):
            # outputs = torch.where(
            #               labels > -1, 
            #               outputs,
            #               torch.LongTensor([0]).expand_as(outputs).to(outputs.device)
            #           )
            # outputs = outputs.nonzero()
            # labels = torch.where(
            #               labels > -1, 
            #               labels,
            #               torch.LongTensor([0]).expand_as(labels).to(outputs.device)
            #           )
            # labels = labels.nonzero()
            return nn.CrossEntropyLoss()(outputs, labels)
        return None

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)()
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)

    def _get_pretrained_model_state_dict(self):
        if self.config.pretrained == 'CIFAR10':
            if self.config.arch == 'ResNet50':
                print("Use pretrained ResNet50 model trained on cifar10 that achieved .86 acc")
                state_dict = torch.load(
                    PRETRAINED_MODEL_PATH[self.config.pretrained][self.config.arch]
                )['net']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module.' == k[:7]:
                        name = k[7:] # remove `module.`
                    else:
                        name = k
                    if "linear." == name[:7]:
                        name = "fc." + name[7:]
                    new_state_dict[name] = v
                del state_dict
                return new_state_dict
            else:
                raise ValueError("Pretrained Model not prepared")
        else:
            raise ValueError("Pretrained Model not prepared")
        
    def _get_network_optimizer(self):
        """ Get softmax network optimizer
        """
        assert hasattr(self, 'model')
        if self.config.optim == 'sgd':
            optim_module = optim.SGD
            optim_param = {"lr" : self.config.lr, 
                           "momentum" : self.config.momentum,
                           "weight_decay" : 0 if self.config.wd == None else 10**self.config.wd}
        elif self.config.optim == "adam":
            optim_module = optim.Adam
            optim_param = {"lr": self.config.lr, 
                           "weight_decay": 10**self.config.wd, 
                           "amsgrad": self.config.amsgrad}
        else:
            raise ValueError("optim type not supported")
            
        optimizer = optim_module(
                        filter(lambda x : x.requires_grad, self.model.parameters()), 
                        **optim_param
                    )
        return optimizer

    def _get_network_scheduler(self):
        """ Get softmax network optimizer
        """
        assert hasattr(self, 'optimizer')
        if self.config.lr_decay_step == None:
            decay_step = self.max_epochs
        else:
            decay_step = self.config.lr_decay_step
        scheduler = lr_scheduler.StepLR(
                        self.optimizer, 
                        step_size=decay_step, 
                        gamma=self.config.lr_decay_ratio
                    )
        return scheduler

    def _update_fc_layer(self, output_size):
        if "resnet" in self.config.arch.lower():
            fd = int(self.model.fc.weight.size()[1])
            self.model.fc = nn.Linear(fd, output_size)
            self.model.fc.weight.data.normal_(0, 0.01)
            self.model.fc.bias.data.zero_()
            self.model.fc.to(self.device)
        else:
            raise NotImplementedError()

def get_acc_from_performance_dict(dict):
    result_dict = {}
    for k in dict.keys():
        if dict[k]['count'] == 0:
            result_dict[k] = "N/A"
        else:
            result_dict[k] = float(dict[k]['corrects']/dict[k]['count'])
    return result_dict