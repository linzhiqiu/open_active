import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm
import copy

import os, random

import models
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list
from distance import eu_distance, cos_distance, eu_distance_batch, cos_distance_batch
from deep_metric import *

from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX, PRETRAINED_MODEL_PATH

import libmr
import math

def get_trainer_machine(training_method, dataset_info, trainer_config):
    """Return a TrainerMachine object
        Args:
            training_method (str) : The training method
            dataset_info (dataset_factory.DatasetInfo) : Dataset information
            trainer_config (train_config.TrainerConfig) : The details about hyperparameter and etc.
    """
    if training_method == "softmax_network":
        trainer_machine_class = SoftmaxNetwork
    elif training_method == "cosine_network":
        trainer_machine_class = CosineNetwork
    
    return trainer_machine_class(dataset_info, trainer_config)

class TrainerMachine(object):
    """A template class for all training machine classes

    Args:
        dataset_info (dataset_factory.DatasetInfo): Dataset information
        trainer_config (train_config.TrainerConfig): Training configuration
    """    
    def __init__(self, dataset_info, trainer_config):
        super(TrainerMachine, self).__init__()
        self.dataset_info = dataset_info
        
        self.optim_config = trainer_config.optim_config
        
        self.batch = trainer_config.batch
        self.workers = trainer_config.workers
        self.device = trainer_config.device
        
        self.trainer_config  = trainer_config
        self.feature_dim = trainer_config.feature_dim

        self.backbone = None # Feature extractor
        self.classifier = None # Initialize per call to train().
        
        self.ckpt_dict = None # A dictionary that holds all checkpoint information

    def train(self, discovered_samples, discovered_classes, ckpt_path=None, verbose=False):
        """Perform the train step (starting from a random initialization)
        """
        if os.path.exists(ckpt_path):
            print("Load from pre-existing ckpt for an already trained network.")
            self.ckpt_dict = torch.load(ckpt_path)
            self._load_ckpt_dict(self.ckpt_dict)
        else:
            print(f"Training the model from scratch. Ckpt will be saved at {ckpt_path}")
            self.backbone = self._get_backbone_network(self.trainer_config.backbone).to(self.device)
            if not self.optim_config.random_restart:
                print("Load random network to ensure same network random initialization")
                random_model_weight_path = os.path.join(".", "weights", self.trainer_config.backbone+".pt")
                if not os.path.exists(random_model_weight_path):
                    if not os.path.exists('./weights'): os.makedirs("./weights"); print("Make a new folder at ./weights/ to store random weights")
                    print(f"Model {random_model_weight_path} doesn't exist. Generating a random model for the first time.")
                    torch.save(self.backbone.state_dict(), random_model_weight_path)
                self.backbone.load_state_dict(torch.load(random_model_weight_path))
            else:
                print("Using random initialization (Not loading from any checkpoint)")
                pass # Not doing anything, just use a random initialization
            self.ckpt_dict = self._train_helper(self.optim_config,
                                                discovered_samples,
                                                discovered_classes,
                                                verbose=verbose)
            torch.save(self.ckpt_dict, ckpt_path)
    
    def eval_closed_set(self, discovered_classes, result_path=None, verbose=True):
        """ Performing closed set evaluation
        """
        if os.path.exists(result_path):
            print("Closed set result already saved.")
            self.closed_set_result = torch.load(result_path)
        else:
            self.closed_set_result = self._eval_closed_set_helper(discovered_classes,
                                                                  verbose=verbose)
            torch.save(self.closed_set_result, result_path)
        return self.closed_set_result['acc']
    
    def get_prob_scores(self, inputs):
        """Returns the prob scores for each inputs
            Returns:
                prob_scores (B x NUM_OF_DISCOVERED_CLASSES)
            Args:
                inputs (B x 3 x ? x ?)
        """
        self.classifier.eval()
        self.backbone.eval()
        return torch.nn.functional.softmax(self.classifier(self.backbone(inputs)),dim=1)

    def get_class_scores(self, inputs):
        """Returns the class scores for each inputs (before softmax)
            Returns:
                prob_scores (B x NUM_OF_DISCOVERED_CLASSES)
            Args:
                inputs (B x 3 x ? x ?)
        """
        self.classifier.eval()
        self.backbone.eval()
        return self.classifier(self.backbone(inputs))

    def get_features(self, inputs):
        """Returns the features for each inputs
            Returns:
                features (B x feature_dim)
            Args:
                inputs (B x 3 x ? x ?)
        """
        self.backbone.eval()
        return self.backbone(inputs)
    
    def _train_helper(self, cfg, discovered_samples, discovered_classes, verbose=True):
        """ The subclasses only need to overwrite this function
        """
        raise NotImplementedError()
    
    def _eval_closed_set_helper(discovered_classes, verbose=True):
        """ To be overwrite by subclass
        """
        raise NotImplementedError()

    # Below are some helper functions shared by all subclasses
    def _load_ckpt_dict(self, ckpt_dict):
        self.backbone = self._get_backbone_network(self.trainer_config.backbone).to(self.device)
        self.classifier = self._get_classifier(ckpt_dict['discovered_classes']).to(self.device)
        self.classifier.load_state_dict(ckpt_dict['classifier'])
        self.backbone.load_state_dict(ckpt_dict['backbone'])

    def _get_backbone_network(self, backbone_name):
        if backbone_name == 'ResNet18':
            backbone = models.ResNet18(last_relu=False)
        elif backbone_name == 'ResNet18HighRes':
            backbone = models.ResNet18(last_relu=False, high_res=True)
        else:
            raise NotImplementedError()
        return backbone

    def get_trainloaders(self, discovered_samples, shuffle=True):
        train_samples = list(set(discovered_samples).difference(self.dataset_info.trainset_info.val_samples))
        print(f"We have {len(train_samples)} train samples and {len(self.dataset_info.trainset_info.val_samples)} val samples.")
        return get_subset_dataloaders(self.dataset_info.train_dataset,
                                      train_samples,
                                      self.dataset_info.trainset_info.val_samples,
                                      None, # No target transform
                                      batch_size=self.batch,
                                      shuffle=shuffle,
                                      workers=self.workers), train_samples, self.dataset_info.trainset_info.val_samples
        
    def _get_target_mapp_func(self, discovered_classes):
        return get_target_mapping_func_for_tensor(self.dataset_info.class_info.classes,
                                                  discovered_classes,
                                                  self.dataset_info.class_info.open_classes,
                                                  device=self.device)
    
    def _get_target_unmapping_func_for_list(self, discovered_classes):
        return get_target_unmapping_func_for_list(self.dataset_info.class_info.classes, discovered_classes)


class Network(TrainerMachine):
    def __init__(self, *args, **kwargs):
        """A base Network (with last relu layer disabled) class
        """
        super(Network, self).__init__(*args, **kwargs)
    
    def _get_classifier(self, discovered_classes):
        raise NotImplementedError()

    def _train_helper(self, cfg, discovered_samples, discovered_classes, verbose=True):
        """Train a new model/classifier and return the ckpt as ckpt_dict
        """
        self.classifier = self._get_classifier(discovered_classes).to(self.device)
        self.backbone.train()
        self.classifier.train()
        
        optimizer = self._get_optimizer(cfg, self.backbone, self.classifier)
        scheduler = self._get_scheduler(cfg, optimizer)
            
        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        
        trainloaders, train_samples, val_samples = self.get_trainloaders(discovered_samples)
        criterion = torch.nn.NLLLoss(reduction='mean')

        avg_loss_per_epoch = []
        avg_acc_per_epoch = []
        avg_val_loss_per_epoch = []
        avg_val_acc_per_epoch = []

        best_val_acc = 0
        best_val_epoch = None
            
        with SetPrintMode(hidden=not verbose):
            for epoch in range(0, cfg.epochs):
                # if epoch == 1:
                #     import pdb; pdb.set_trace()
                for phase in trainloaders.keys():
                    if phase == 'train':
                        # Important
                        self.backbone.train()
                        self.classifier.train()
                    elif phase == 'val':
                        self.backbone.eval()
                        self.classifier.eval()
                    running_loss = 0.0
                    running_corrects = 0.
                    count = 0

                    if verbose:
                        pbar = tqdm(trainloaders[phase], ncols=80)
                    else:
                        pbar = trainloaders[phase]

                    for batch, data in enumerate(pbar):
                        inputs, real_labels = data
                        labels = target_mapping_func(real_labels)
                        count += inputs.size(0)
                        
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        if phase == 'train': optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            features = self.backbone(inputs)
                            outputs = self.classifier(features)
                            _, preds = torch.max(outputs, 1)

                            log_probability = F.log_softmax(outputs, dim=1)

                            loss = criterion(log_probability, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        if verbose:
                            pbar.set_postfix(loss=float(running_loss)/count, 
                                             acc=float(running_corrects)/count,
                                             epoch=epoch,
                                             phase=phase)
                    
                    avg_loss = float(running_loss)/count
                    avg_acc = float(running_corrects)/count
                    if phase == 'train': 
                        avg_loss_per_epoch.append(avg_loss)
                        avg_acc_per_epoch.append(avg_acc)
                        scheduler.step()
                        if 'val' not in trainloaders.keys():
                            best_val_acc_backbone_state_dict = self.backbone.state_dict()
                            best_val_acc_classifier_state_dict = self.classifier.state_dict()
                    elif phase == 'val':
                        avg_val_loss_per_epoch.append(avg_loss)
                        avg_val_acc_per_epoch.append(avg_acc)
                        if avg_acc > best_val_acc:
                            print(f"Best val accuracy at epoch {epoch} being {avg_acc}")
                            best_val_epoch = epoch
                            best_val_acc = avg_acc
                            best_val_acc_backbone_state_dict = self.backbone.state_dict()
                            best_val_acc_classifier_state_dict = self.classifier.state_dict()
                    print(f"Average {phase} Loss {avg_loss}, Accuracy {avg_acc}")
                print()
        
        if 'val' in trainloaders.keys():
            print(f"Load state dict of best val accuracy at epoch {best_val_epoch} being {best_val_acc}")
        else:
            print("Use the model checkpoint at last epoch")
        self.backbone.load_state_dict(best_val_acc_backbone_state_dict)
        self.classifier.load_state_dict(best_val_acc_classifier_state_dict)
        ckpt_dict = {
            'backbone' : self.backbone.state_dict(),
            'classifier' : self.classifier.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'discovered_samples' : discovered_samples,
            'discovered_classes' : discovered_classes,
            'train_samples' : train_samples,
            'val_samples' : val_samples,
            'loss_curve' : avg_loss_per_epoch,
            'acc_curve' : avg_acc_per_epoch,
        }
        if 'val' in trainloaders.keys():
            ckpt_dict['val_loss_curve'] = avg_val_loss_per_epoch
            ckpt_dict['val_acc_curve'] = avg_val_acc_per_epoch
            ckpt_dict['best_val_epoch'] = best_val_epoch
            ckpt_dict['best_val_acc'] = best_val_acc
        return ckpt_dict
    
    
    
    def _eval_closed_set_helper(self, discovered_classes, verbose=True):
        """Test the model/classifier and return the acc in ckpt_dict
        """
        self.backbone.eval()
        self.classifier.eval()

        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(discovered_classes) # Only for transforming predicted label (in network indices) to real indices
        dataloader = get_loader(self.dataset_info.test_dataset,
                                None,
                                shuffle=False,
                                batch_size=self.batch,
                                workers=self.workers)
        
        with SetPrintMode(hidden=not verbose):
            if verbose:
                pbar = tqdm(dataloader, ncols=80)
            else:
                pbar = dataloader

            performance_dict = {'corrects' : 0., 'not_seen' : 0., 'count' : 0.} # Accuracy of all non-hold out open class examples. If some classes not seen yet, accuracy = 0
            closed_set_result = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
                                 'real_labels' : [], # The real labels for CIFAR100 or other datasets.
                                 'closed_predicted' : [], # The predicted closed set label (indices in network output)
                                 'closed_predicted_real' : [], # The predicted closed set label (real labels)
                                 'closed_argmax_prob' : [], # The probability for predicted closed set class
                                 } # A list of dictionary

            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, real_labels = data

                    inputs = inputs.to(self.device)
                    labels = target_mapping_func(real_labels.to(self.device))

                    outputs = self.classifier(self.backbone(inputs))
                    softmax_max, preds = torch.max(outputs, 1)

                    undiscovered_open_indices = labels == UNDISCOVERED_CLASS_INDEX
                    discovered_closed_indices = labels >= 0
                    hold_out_open_indices = labels == OPEN_CLASS_INDEX
                    unlabeled_pool_class_indices = undiscovered_open_indices | discovered_closed_indices
                    assert torch.sum(undiscovered_open_indices & discovered_closed_indices & hold_out_open_indices) == 0

                    performance_dict['count'] += float(torch.sum(unlabeled_pool_class_indices))

                    performance_dict['not_seen'] += torch.sum(undiscovered_open_indices).float()
                    performance_dict['corrects'] += torch.sum(
                                                                 torch.masked_select(
                                                                     (preds==labels.data),
                                                                     discovered_closed_indices
                                                                 )
                                                             ).float()
                    
                    closed_set_result['ground_truth'] += labels.tolist()
                    closed_set_result['real_labels'] += real_labels.tolist()
                    closed_set_result['closed_predicted'] += preds.tolist()
                    closed_set_result['closed_predicted_real'] += target_unmapping_func_for_list(preds.tolist())
                    closed_set_result['closed_argmax_prob'] += softmax_max.tolist()

            test_acc = performance_dict['corrects'] / performance_dict['count']
            seen_rate = 1. - performance_dict['not_seen'] / performance_dict['count']
            
            
            print(f"Test => "
                  f"Closed set test Acc {test_acc}, "
                  f"Discovered precentage {seen_rate}")

            print(f"Test Accuracy {test_acc}.")
        test_dict = {
            'acc' : test_acc,
            'seen' : seen_rate,
            'closed_set_result' : closed_set_result
        }
        return test_dict
    
    def _get_optimizer(self, cfg, backbone, classifier):
        """ Get optimizer of both backbone and classifier
        """
        if cfg.optim == 'sgd':
            optim_module = torch.optim.SGD
            optim_param = {"lr" : cfg.lr, 
                           "momentum" : cfg.momentum,
                           "weight_decay" : float(cfg.weight_decay)}
        else: raise NotImplementedError()
        optimizer = optim_module(
                        [
                            {'params': filter(lambda x : x.requires_grad, backbone.parameters())},
                            {'params': filter(lambda x : x.requires_grad, classifier.parameters())}
                        ],
                        **optim_param
                    )
        return optimizer

    def _get_scheduler(self, cfg, optimizer):
        """ Get learning rate scheduler
        """
        if cfg.decay_epochs == None:
            decay_step = cfg.epochs
        else:
            decay_step = cfg.decay_epochs
        scheduler = lr_scheduler.StepLR(
                        optimizer, 
                        step_size=decay_step, 
                        gamma=cfg.decay_by
                    )
        return scheduler


class SoftmaxNetwork(Network):
    def __init__(self, *args, **kwargs):
        """A softmax network (with last relu layer disabled) class
        """
        super(SoftmaxNetwork, self).__init__(*args, **kwargs)
    
    def _get_classifier(self, discovered_classes):
        return torch.nn.Linear(self.feature_dim, len(discovered_classes))


class CosineNetwork(Network):
    def __init__(self, *args, **kwargs):
        """A cosine similarity network (with last relu layer disabled) class
        """
        super(CosineNetwork, self).__init__(*args, **kwargs)
    
    def _get_classifier(self, discovered_classes):
        return models.cosine_clf(self.feature_dim, len(discovered_classes))


def get_acc_from_performance_dict(dict):
    result_dict = {}
    for k in dict.keys():
        if dict[k]['count'] == 0:
            result_dict[k] = "N/A"
        else:
            result_dict[k] = float(dict[k]['corrects']/dict[k]['count'])
    return result_dict
