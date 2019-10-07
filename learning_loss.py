import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm
import copy

from trainer_machine import Network
from instance_info import LearningLossInfoCollector

import models
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func, get_target_unmapping_dict

from global_setting import OPEN_CLASS_INDEX, UNSEEN_CLASS_INDEX, PRETRAINED_MODEL_PATH

import libmr
import math


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResNetLearningLoss(nn.Module):
    def __init__(self, resnet_model, resnet_type='ResNet18', mid_feature_size=128):
        super(self.__class__, self).__init__()
        self.resnet_model = resnet_model
        self.resnet_type = resnet_type
        if self.resnet_type == "ResNet18":
            layer_sizes = [64, 128, 256, 512]
        elif self.resnet_type == "ResNet50":
            layer_sizes = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError()

        mid_loss_layers = []
        for layer_size in layer_sizes:
            mid_loss_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(layer_size, mid_feature_size),
                nn.ReLU(),
            )
            mid_loss_layers.append(mid_loss_layer)

        self.mid_loss_layer_1 = mid_loss_layers[0]
        self.mid_loss_layer_2 = mid_loss_layers[1]
        self.mid_loss_layer_3 = mid_loss_layers[2]
        self.mid_loss_layer_4 = mid_loss_layers[3] 

        self.loss_layer = nn.Linear(len(layer_sizes)*mid_feature_size, 1)

    def forward(self, x):
        out = F.relu(self.resnet_model.bn1(self.resnet_model.conv1(x)))
        out = self.resnet_model.layer1(out)
        f_1 = self.mid_loss_layer_1(out)
        out = self.resnet_model.layer2(out)
        f_2 = self.mid_loss_layer_2(out)
        out = self.resnet_model.layer3(out)
        f_3 = self.mid_loss_layer_3(out)
        out = self.resnet_model.layer4(out)
        f_4 = self.mid_loss_layer_4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        pred_loss = self.loss_layer(torch.cat([f_1, f_2, f_3, f_4], dim=1))
        return out, pred_loss


def train_epochs_learning_loss(model, dataloaders, optimizer, scheduler, lmb=1.0, margin=1.0, stop_prop_epoch=120, device='cuda', start_epoch=0, max_epochs=-1, verbose=True):
    """Regular PyTorch training procedure: Train model using data in dataloaders['train'] from start_epoch to max_epochs-1
    """
    assert start_epoch < max_epochs
    avg_loss = 0.
    avg_acc = 0.
    avg_loss_loss = 0.
    avg_loss_acc = 0.

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    for epoch in range(start_epoch, max_epochs):
        # print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        # print('-' * 10)
        for phase in dataloaders.keys():
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            # For cross entropy
            running_loss = 0.0
            running_corrects = 0.
            count = 0

            # For pairwise loss prediction
            running_loss_loss = 0.0
            running_loss_corrects = 0.
            pair_count = 0

            if verbose:
                pbar = tqdm(dataloaders[phase], ncols=80)
            else:
                pbar = dataloaders[phase]

            for batch, data in enumerate(pbar):
                inputs, labels = data
                count += math.floor(inputs.size(0)/2)*2 # Ignore last odd item
                pair_count += math.floor(inputs.size(0)/2) # Ignore last odd item
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, pred_losses = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    pred_losses = pred_losses.view(inputs.shape[0])

                    if inputs.shape[0] == 1:
                        loss_loss = torch.zeros([0.0]).to(device)
                        loss = torch.zeros([0.0]).to(device)
                    else:
                        if inputs.shape[0] % 2 != 0:
                            loss = loss[-1:]
                            pred_losses = pred_losses[-1:]
                        split_index = int(loss.shape[0]/2)
                        loss_1, loss_2 = loss[:split_index], loss[split_index:]
                        pred_losses_1, pred_losses_2 = pred_losses[:split_index], pred_losses[split_index:]
                        loss_gt = torch.where(
                                      loss_1 > loss_2,
                                      torch.Tensor([1.]).to(device),
                                      torch.Tensor([-1.]).to(device)
                                  )
                        loss_corrects = (loss_1 > loss_2) == (pred_losses_1 > pred_losses_2)
                        loss_loss = torch.max(torch.FloatTensor([0]).to(device),
                                              -1*loss_gt*(pred_losses_1-pred_losses_2)+margin)

                    if epoch < stop_prop_epoch:
                        combined_loss = loss.mean() + 2*lmb*loss_loss.mean()
                    else:
                        combined_loss = loss.mean()

                    if phase == 'train':
                        combined_loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.mean().item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    running_loss_loss += loss_loss.mean().item() * math.floor(inputs.size(0)/2)
                    running_loss_corrects += torch.sum(loss_corrects)
                    if verbose:
                        pbar.set_postfix(loss=running_loss/count, 
                                         acc=float(running_corrects)/count,
                                         l_loss=float(running_loss_loss)/pair_count,
                                         l_acc=float(running_loss_corrects)/pair_count,
                                         epoch=epoch)

                avg_loss = running_loss/count
                avg_acc = float(running_corrects)/count
                avg_loss_loss = running_loss_loss/pair_count
                avg_loss_acc = float(running_loss_corrects)/pair_count
                # print(f"Epoch {epoch} => "
                #       f"Loss {avg_loss}, Accuracy {avg_acc}")
            # print()

    return avg_loss, avg_acc, avg_loss_loss, avg_loss_acc


class NetworkLearningLoss(Network):
    def __init__(self, *args, **kwargs):
        super(NetworkLearningLoss, self).__init__(*args, **kwargs)
        assert self.config.class_weight in ['uniform'] # No 'class_imbalanced'
        self.learning_loss_train_mode = self.config.learning_loss_train_mode
        self.learning_loss_lambda = self.config.learning_loss_lambda
        self.learning_loss_margin = self.config.learning_loss_margin
        self.learning_loss_stop_epoch = self.config.learning_loss_stop_epoch

        self.info_collector_class = LearningLossInfoCollector


    def _train(self, model, s_train, seen_classes, start_epoch=0):
        self._train_mode()
        target_mapping_func = self._get_target_mapp_func(seen_classes)
        self.dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                  list(s_train),
                                                  [], # TODO: Make a validation set
                                                  target_mapping_func,
                                                  batch_size=self.config.batch,
                                                  workers=self.config.workers)
        
        self._update_last_layer(model, len(seen_classes), device=self.device)
        optimizer = self._get_network_optimizer(model)
        scheduler = self._get_network_scheduler(optimizer)

        # self.criterion = self._get_criterion(self.dataloaders['train'],
        #                                      seen_classes=seen_classes,
        #                                      criterion_class=self.criterion_class)

        with SetPrintMode(hidden=not self.config.verbose):
            train_loss, train_acc, loss_loss, loss_acc = train_epochs_learning_loss(
                                                             model,
                                                             self.dataloaders,
                                                             optimizer,
                                                             scheduler,
                                                             # self.criterion,
                                                             lmb=self.learning_loss_lambda,
                                                             margin=self.learning_loss_margin,
                                                             stop_prop_epoch=self.learning_loss_stop_epoch, #TODO: Figure out what this should be
                                                             device=self.device,
                                                             start_epoch=start_epoch,
                                                             max_epochs=self.max_epochs,
                                                             verbose=self.config.verbose,
                                                         )
        print(f"Train => {self.round} round => "
              f"Label_Loss {train_loss}, Accuracy {train_acc}"
              f"Pred_Loss {loss_loss}, Accuracy {loss_acc}")
        return train_loss, train_acc

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
        if self.config.network_eval_mode == 'threshold':
            assert self.config.threshold_metric == "softmax"
            threshold = self.config.network_eval_threshold
        elif self.config.network_eval_mode == 'dynamic_threshold':
            assert type(self.log) == list
            if len(self.log) == 0:
                # First round, use default threshold
                threshold = self.config.network_eval_threshold
                print(f"First round. Use default threshold {threshold}")
            else:
                try:
                    threshold = trainer_machine.get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
                except trainer_machine.NoSeenClassException:
                    # Error when no new instances from seen class
                    threshold = self.config.network_eval_threshold
                    print(f"No seen class instances. Threshold set to {threshold}")
                except trainer_machine.NoUnseenClassException:
                    threshold = self.config.network_eval_threshold
                    print(f"No unseen class instances. Threshold set to {threshold}")
                else:
                    print(f"Threshold set to {threshold} based on all existing instances.")
        elif self.config.network_eval_mode in ['pseuopen_threshold']:
            assert hasattr(self, 'pseuopen_threshold')
            print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
            threshold = self.pseuopen_threshold

        def open_set_prediction(outputs_tuple, inputs=None):
            # TODO: output is a tuple
            outputs, losses = outputs_tuple
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            if self.config.threshold_metric == 'softmax':
                scores = softmax_max
            elif self.config.threshold_metric == 'entropy':
                scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!

            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
            self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
            self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
            self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
            self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
            self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
            
            preds = torch.where(scores < threshold,
                                torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device), 
                                softmax_preds)
            return preds
        return open_set_prediction


    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)()
        model = ResNetLearningLoss(model, self.config.arch)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.resnet_model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)


    def _update_last_layer(self, model, output_size, device='cuda'):
        if "resnet" in self.config.arch.lower():
            fd = int(model.resnet_model.fc.weight.size()[1])
            model.fc = nn.Linear(fd, output_size)
            model.fc.weight.data.normal_(0, 0.01)
            model.fc.bias.data.zero_()
            model.fc.to(device)
        elif self.config.arch in ['classifier32', 'classifier32_instancenorm']:
            raise NotImplementedError()
            # fd = int(model.fc1.weight.size()[1])
            # model.fc1 = nn.Linear(fd, output_size)
            # model.fc1.weight.data.normal_(0, 0.01)
            # model.fc1.bias.data.zero_()
            # model.fc1.to(device)
        else:
            raise NotImplementedError()
