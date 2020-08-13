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
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list, get_index_mapping_func
from utils import IndexDataset
from distance import eu_distance, cos_distance, eu_distance_batch, cos_distance_batch
from deep_metric import *

from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX, PRETRAINED_MODEL_PATH

import libmr
import math

def get_trainer_machine(training_method, train_mode, trainset_info, trainer_config, test_dataset, val_samples=None, active_test_val_diff=False):
    """Return a TrainerMachine object
        Args:
            training_method (str) : The training method
            train_mode (str) : The training mode (with/without finetune)
            trainset_info (TrainsetInfo) : The details about training set
            trainer_config (dict) : The details about hyperparameter and etc.
            test_dataset
            val_samples
            active_test_val_diff
    """
    if training_method == "softmax_network":
        trainer_machine_class = SoftmaxNetwork
    elif training_method == "cosine_network":
        trainer_machine_class = CosineNetwork
    elif training_method == 'deep_metric':
        trainer_machine_class = DeepMetricNetwork
    # elif training_method == 'sigmoid_network':
    #     trainer_machine_class = SigmoidNetwork
    else:
        raise NotImplementedError()
    
    return trainer_machine_class(train_mode, trainset_info, trainer_config, test_dataset, val_samples=val_samples, active_test_val_diff=active_test_val_diff)

class TrainerMachine(object):
    """Abstract class"""
    def __init__(self, train_mode, trainset_info, trainer_config, test_dataset, val_samples=None, active_test_val_diff=False):
        super(TrainerMachine, self).__init__()
        self.train_mode = train_mode
        self.trainset_info = trainset_info
        
        self.train_config = trainer_config['train']
        self.finetune_config = trainer_config['finetune']
        
        self.batch = trainer_config['batch']
        self.workers = trainer_config['workers']
        self.device = trainer_config['device']
        
        self.trainer_config  = trainer_config
        self.backbone = None
        self.feature_dim = trainer_config['feature_dim']
        self.classifier = None # Initialize per train()/finetune() call.
        
        self.ckpt_dict = None # A dictionary that holds all checkpoint information
        self.val_samples = val_samples
        self.test_dataset = test_dataset
        self.active_test_val_diff = active_test_val_diff # whether to compare val acc against test acc for every epochs

    def train(self, discovered_samples, discovered_classes, ckpt_path=None, verbose=False):
        """Perform the train step (starting from a random initialization)
        """
        if os.path.exists(ckpt_path):
            print("Load from pre-existing ckpt. No training will be performed.")
            self.ckpt_dict = torch.load(ckpt_path)
            self._load_ckpt_dict(self.ckpt_dict)
        else:
            print(f"Training the model from scratch. Ckpt will be saved at {ckpt_path}")
            self.backbone = self._get_backbone_network(self.trainer_config['backbone']).to(self.device)
            if not self.train_config.random_restart:
                # Load random weight from a checkpoint to ensure different round uses same initialization
                random_model_weight_path = os.path.join(".", "weights", self.trainer_config['backbone']+".pt")
                if not os.path.exists(random_model_weight_path):
                    if not os.path.exists('./weights'): os.makedirs("./weights"); print("Make a new folder at ./weights/ to store random weights")
                    print(f"Model {random_model_weight_path} doesn't exist. Generating a random model for the first time.")
                    torch.save(self.backbone.state_dict(), random_model_weight_path)
                self.backbone.load_state_dict(torch.load(random_model_weight_path))
            else:
                print("Using random initialization (Not loading from any checkpoint)")
                pass # Not doing anything, just use a random initialization
            self.ckpt_dict = self._train_helper(self.train_config,
                                                discovered_samples,
                                                discovered_classes,
                                                verbose=verbose)
            torch.save(self.ckpt_dict, ckpt_path)

    def finetune(self, discovered_samples, discovered_classes, ckpt_path=None, verbose=False):
        """Perform the finetuning step
        """
        if os.path.exists(ckpt_path):
            print("Load from pre-existing ckpt. No finetuning will be performed.")
            self.ckpt_dict = torch.load(ckpt_path)
            self._load_ckpt_dict(self.ckpt_dict)
        else:
            print(f"First time finetuning the model. Ckpt will be saved at {ckpt_path}")
            if self.train_mode in ['retrain']:
                print("Use a new backbone network without finetuning.")
                self.backbone = self._get_backbone_network(self.trainer_config['backbone']).to(self.device)
            elif self.train_mode == 'fix_feature_extractor':
                print("Fix feature extractor..")
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self.ckpt_dict = self._train_helper(self.finetune_config,
                                                discovered_samples,
                                                discovered_classes,
                                                verbose=verbose)
            torch.save(self.ckpt_dict, ckpt_path)
    
    # def eval_closed_set(self, discovered_classes, test_dataset, result_path=None, verbose=True):
    def eval_closed_set(self, discovered_classes, result_path=None, verbose=True):
        """ Performing closed set evaluation
        """
        if os.path.exists(result_path):
            print("Closed set result already saved.")
            self.closed_set_result = torch.load(result_path)
            if not 'closed_set_result' in self.closed_set_result.keys():
                print("Closed set result incomplete. We will do it once again")
                self.closed_set_result = self._eval_closed_set_helper(discovered_classes,
                                                                      self.test_dataset,
                                                                      verbose=verbose)
                torch.save(self.closed_set_result, result_path)
        else:
            self.closed_set_result = self._eval_closed_set_helper(discovered_classes,
                                                                  self.test_dataset,
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
    
    def _eval_closed_set_helper(discovered_classes, test_dataset, verbose=True):
        """ To be overwrite by subclass
        """
        raise NotImplementedError()

    # Below are some helper functions shared by all subclasses
    def _load_ckpt_dict(self, ckpt_dict):
        self.backbone = self._get_backbone_network(self.trainer_config['backbone']).to(self.device)
        self.classifier = self._get_classifier(ckpt_dict['discovered_classes']).to(self.device)
        self.classifier.load_state_dict(ckpt_dict['classifier'])
        self.backbone.load_state_dict(ckpt_dict['backbone'])

    def _get_backbone_network(self, backbone_name):
        if backbone_name == 'ResNet18':
            backbone = models.ResNet18(last_relu=False) # Always false.
        elif backbone_name == 'ResNet18HighRes':
            backbone = models.ResNet18(last_relu=False, high_res=True)
        elif backbone_name == 'ResNet18ImgNet':
            import torchvision
            pretrained_model = torchvision.models.resnet18(pretrained=True)
            class PretrainedResNetBackbone(torch.nn.Module):
                def __init__(self, pretrained_model):
                    super().__init__()
                    self.model = pretrained_model
                
                def forward(self, x):
                    x = self.model.relu(self.model.bn1(self.model.conv1(x)))
                    x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    x = self.model.avgpool(x)
                    return x.view(x.shape[0], -1)
            backbone = PretrainedResNetBackbone(pretrained_model)
        else:
            raise NotImplementedError()
        return backbone

    def get_trainloaders(self, discovered_samples, shuffle=True):
        # if self.val_mode == None:
        #     train_samples = discovered_samples
        #     val_samples = []
        # else:
        # val_ratio = 0.05
        # print(f"Using validation set ratio {val_ratio}")
        # val_size = int(val_ratio * len(discovered_samples))
        # if self.val_mode == 'randomized':
        #     print("Select the validation set randomly..")
        #     discovered_samples_copy = discovered_samples.copy()
        #     random.shuffle(discovered_samples_copy)
            
        #     train_samples = discovered_samples_copy[val_size:]
        #     val_samples = discovered_samples_copy[:val_size]
        # elif self.val_mode == 'balanced':
        #     print("Select the validation set to have a balanced distribution..")
        #     class_to_indices = {}
        #     for sample_i in discovered_samples:
        #         class_i = self.trainset_info.train_labels[sample_i]
        #         if not class_i in class_to_indices:
        #             class_to_indices[class_i] = []
        #         class_to_indices[class_i].append(sample_i)
        #     num_classes = len(class_to_indices.keys())
            
        #     val_samples = []
        #     if val_size < num_classes:
        #         print(f"Cannot afford to have more than one val sample per class, so we do it for {val_size} classes.")
        #         for class_i in class_to_indices.keys():
        #             if len(val_samples) >= val_size:
        #                 break
        #             val_samples.append(class_to_indices[class_i][0])
        #     else:
        #         per_class = float(val_size)/num_classes
        #         print(f"We have on avg {int(per_class)} sample per class")
        #         for class_i in class_to_indices.keys():
        #             val_samples += class_to_indices[class_i][:int(per_class)]
        #         remaining_val_size = val_size - len(val_samples)
        #         print(f"We have remaining {remaining_val_size} samples, and we pick one sample from random {remaining_val_size} classes")
        #         for class_i in class_to_indices.keys():
        #             if len(val_samples) >= val_size:
        #                 break
        #             if len(class_to_indices[class_i]) > int(per_class):
        #                 val_samples.append(class_to_indices[class_i][int(per_class)])
        #     train_samples = list(set(discovered_samples).difference(set(val_samples)))
        # elif self.val_mode == 'stratified':
        #     import math
        #     print(f"Select the validation set to be {val_ratio:.2%} of each discovered class. Ensure at least 1 training sample...")
        #     class_to_indices = {}
        #     for sample_i in discovered_samples:
        #         class_i = self.trainset_info.train_labels[sample_i]
        #         if not class_i in class_to_indices:
        #             class_to_indices[class_i] = []
        #         class_to_indices[class_i].append(sample_i)
        #     num_classes = len(class_to_indices.keys())
            
        #     val_samples = []
        #     for class_i in class_to_indices.keys():
        #         if len(class_to_indices[class_i]) <= 1:
        #             continue
        #         else:
        #             samples_in_class_i = class_to_indices[class_i].copy()
        #             random.shuffle(samples_in_class_i)
        #             class_i_size = math.ceil(len(samples_in_class_i) * val_ratio)
        #             val_samples += samples_in_class_i[:class_i_size]
        #     train_samples = list(set(discovered_samples).difference(set(val_samples)))
        # elif self.val_mode == 'fixed_stratified':
        #     import math
        #     print(f"Select the validation set to be {val_ratio:.2%} of each discovered class. Ensure at least 1 training sample...")
        #     class_to_indices = {}
        #     for sample_i in discovered_samples:
        #         class_i = self.trainset_info.train_labels[sample_i]
        #         if not class_i in class_to_indices:
        #             class_to_indices[class_i] = []
        #         class_to_indices[class_i].append(sample_i)
        #     num_classes = len(class_to_indices.keys())
            
        #     val_samples = []
        #     for class_i in class_to_indices.keys():
        #         if len(class_to_indices[class_i]) <= 1:
        #             continue
        #         else:
        #             samples_in_class_i = class_to_indices[class_i].copy()
        #             random.shuffle(samples_in_class_i)
        #             class_i_size = math.ceil(len(samples_in_class_i) * val_ratio)
        #             val_samples += samples_in_class_i[:class_i_size]
        #     train_samples = list(set(discovered_samples).difference(set(val_samples)))
        # else:
        #     raise NotImplementedError()
        train_samples = list(set(discovered_samples).difference(self.val_samples))
        print(f"We have {len(train_samples)} train samples and {len(self.val_samples)} val samples.")
        return get_subset_dataloaders(self.trainset_info.train_dataset,
                                      train_samples,
                                      self.val_samples,
                                      None, # No target transform
                                      batch_size=self.batch,
                                      shuffle=shuffle,
                                      workers=self.workers), train_samples, self.val_samples
        
    def _get_target_mapp_func(self, discovered_classes):
        return get_target_mapping_func_for_tensor(self.trainset_info.classes,
                                                  discovered_classes,
                                                  self.trainset_info.open_classes,
                                                  device=self.device)
    
    def _get_target_unmapping_func_for_list(self, discovered_classes):
        return get_target_unmapping_func_for_list(self.trainset_info.classes, discovered_classes)

    def _get_index_mapp_func(self, discovered_samples):
        return get_index_mapping_func(discovered_samples)
                                                  
    # def _get_target_unmapping_func_for_list(self, discovered_classes):
    #     return get_target_unmapping_func_for_list(self.trainset_info.classes,

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

        if self.active_test_val_diff: avg_test_acc_per_epoch = []

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
                        if self.active_test_val_diff:
                            # eval on test set as well
                            test_acc = self._eval_closed_set_helper(discovered_classes,
                                                                    self.test_dataset,
                                                                    verbose=verbose)
                            avg_test_acc_per_epoch.append(test_acc)
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
        if self.active_test_val_diff:
            ckpt_dict['test_acc_curve'] = avg_test_acc_per_epoch
        return ckpt_dict
    
    
    
    def _eval_closed_set_helper(self, discovered_classes, test_dataset, verbose=True):
        """Test the model/classifier and return the acc in ckpt_dict
        """
        self.backbone.eval()
        self.classifier.eval()

        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(discovered_classes) # Only for transforming predicted label (in network indices) to real indices
        dataloader = get_loader(test_dataset,
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

class DeepMetricNetwork(Network): # Xiuyu : You may also inherit the Network class
    def __init__(self, *args, **kwargs):
        """A deep metric network (with last relu layer disabled) class
        """
        super(DeepMetricNetwork, self).__init__(*args, **kwargs)
        self.num_neighbours = self.trainer_config['num_neighbours']
        self.sigma = self.trainer_config['sigma']
        self.interval = self.trainer_config['interval']

    def get_prob_scores(self, inputs):
        return self.get_class_scores(inputs)

    def _get_optimizer(self, cfg, backbone, classifier):
        """ Get optimizer of both backbone and classifier
        """

        if cfg.optim == 'sgd':
            optim_module = torch.optim.SGD
            optim_param = {"lr" : cfg.lr, 
                           "momentum" : cfg.momentum,
                           "weight_decay" : float(cfg.weight_decay)}
        elif cfg.optim == 'adam':
            optim_module = torch.optim.Adam
            optim_param = {"lr" : cfg.lr, 
                           "weight_decay" : float(cfg.weight_decay)}
        else: raise NotImplementedError()

        optimizer = optim_module(
                    [
                        {'params': filter(lambda x : x.requires_grad, backbone.parameters())},
                        {'params': filter(lambda x : x.requires_grad, classifier.parameters()), 'lr': cfg.lr}
                    ],
                    **optim_param
                )
        
        return optimizer
    
    def load_backbone(self, path):
        print(f"loading pretrained softmax network from path {path}")
        self.backbone.load_state_dict(torch.load(path))
    
    def _train_softmax_helper(self, cfg, discovered_samples, discovered_classes, verbose=True):
        self.backbone.train()
        self.classifier = torch.nn.Linear(self.feature_dim, len(discovered_classes)).to(self.device)
        self.classifier.train()

        optim_param = {"lr" : cfg.softmax_lr, 
                        "momentum" : cfg.momentum,
                        "weight_decay" : float(cfg.softmax_weight_decay)}
        optimizer = torch.optim.SGD(
                        [
                            {'params': filter(lambda x : x.requires_grad, self.backbone.parameters())},
                            {'params': filter(lambda x : x.requires_grad, self.classifier.parameters())}
                        ],
                        **optim_param
                    )
        if cfg.softmax_decay_epochs == None:
            decay_step = cfg.softmax_epochs
        else:
            decay_step = cfg.softmax_decay_epochs
        scheduler = lr_scheduler.StepLR(
                        optimizer, 
                        step_size=decay_step, 
                        gamma=cfg.softmax_decay_by
                    )
        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        trainloader = self.get_trainloader(discovered_samples)

        criterion = torch.nn.NLLLoss(reduction='mean')

        avg_loss_per_epoch = []
        avg_acc_per_epoch = []
        avg_loss = None
        avg_acc = None
        with SetPrintMode(hidden=not verbose):
            for epoch in range(0, cfg.softmax_epochs):
                running_loss = 0.0
                running_corrects = 0.
                count = 0

                if verbose:
                    pbar = tqdm(trainloader, ncols=80)
                else:
                    pbar = trainloader

                for batch, data in enumerate(pbar):
                    inputs, real_labels = data
                    labels = target_mapping_func(real_labels)
                    count += inputs.size(0)

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    features = self.backbone(inputs)
                    outputs = self.classifier(features)
                    _, preds = torch.max(outputs, 1)

                    log_probability = F.log_softmax(outputs, dim=1)

                    loss = criterion(log_probability, labels)

                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if verbose:
                        pbar.set_postfix(loss=float(running_loss)/count, 
                                         acc=float(running_corrects)/count,
                                         epoch=epoch)

                avg_loss = float(running_loss)/count
                avg_acc = float(running_corrects)/count
                avg_loss_per_epoch.append(avg_loss)
                avg_acc_per_epoch.append(avg_acc)
                scheduler.step()
            print(f"Average Loss {avg_loss}, Accuracy {avg_acc}")

    def compute_novel_distance(self, discovered_samples):
        return
        undiscovered_samples = list(self.trainset_info.query_samples.difference(discovered_samples))
        open_loader = get_subset_loader(self.trainset_info.train_dataset,
                                          list(self.trainset_info.open_samples),
                                          None, # No target transform
                                          batch_size=self.batch,
                                          shuffle=False,
                                          workers=self.workers)
        undiscovered_loader = get_subset_loader(self.trainset_info.train_dataset,
                                          undiscovered_samples,
                                          None, # No target transform
                                          batch_size=self.batch,
                                          shuffle=False,
                                          workers=self.workers)
        discovered_loader = get_subset_loader(self.trainset_info.train_dataset,
                                          discovered_samples,
                                          None, # No target transform
                                          batch_size=self.batch,
                                          shuffle=False,
                                          workers=self.workers)                                          
        open_features = self.collect_features(open_loader, verbose=False).cpu()
        undiscovered_features = self.collect_features(undiscovered_loader, verbose=False).cpu()
        discovered_features = self.collect_features(discovered_loader, verbose=False).cpu()

        from query_machine import distance_matrix
        open_distance = distance_matrix(open_features, discovered_features)
        undiscovered_distance = distance_matrix(undiscovered_features, discovered_features)

        open_avg = float(open_distance.mean())
        undiscovered_avg = float(undiscovered_distance.mean())
        open_min_avg = float(open_distance.min(dim=1)[0].mean())
        undiscovered_min_avg = float(undiscovered_distance.min(dim=1)[0].mean())
        print(f"Open to discovered: Avg dist {open_avg}, Avg min dist {open_min_avg}")
        print(f"Undiscovered to discovered: Avg dist {undiscovered_avg}, Avg min dist {undiscovered_min_avg}")
    
    def collect_features(self, dataloader, verbose=True):
        features = torch.Tensor([]).to(self.device)
        for batch, data in enumerate(dataloader):
            inputs, _ = data
            
            with torch.no_grad():
                cur_features = self.get_features(inputs.to(self.device))

            features = torch.cat((features, cur_features),dim=0)

        return features


    def _train_helper(self, cfg, discovered_samples, discovered_classes, verbose=True):
        # No matter what, first trained a softmax network
        self._train_softmax_helper(cfg, discovered_samples, discovered_classes, verbose=verbose)
        train_dataset_with_index = IndexDataset(self.trainset_info.train_dataset)
        update_loader = get_subset_loader(train_dataset_with_index,
                                          discovered_samples,
                                          None, # No target transform
                                          batch_size=self.batch,
                                          shuffle=False,
                                          workers=self.workers)
        train_loader_with_index = get_subset_loader(train_dataset_with_index,
                                                    discovered_samples,
                                                    None, # No target transform
                                                    batch_size=self.batch,
                                                    shuffle=True,
                                                    workers=self.workers)
        self.num_train = len(discovered_samples)        

        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        index_mapping_func = self._get_index_mapp_func(discovered_samples)

        # Create tensor to store kernel centres
        self.centres = torch.zeros(self.num_train, self.feature_dim).type(torch.FloatTensor).to(self.device)
        print("Size of centres is {0}".format(self.centres.size()))

        # Create tensor to store labels of centres
        targets = [target_mapping_func(self.trainset_info.train_dataset.targets[i]) for i in discovered_samples]
        # print(f'targets {targets}')
        self.centre_labels = torch.LongTensor(targets).to(self.device)

        self.classifier = self._get_classifier(discovered_classes).to(self.device)
        
        optimizer = self._get_optimizer(cfg, self.backbone, self.classifier)
        scheduler = self._get_scheduler(cfg, optimizer)

        criterion = nn.NLLLoss()

        avg_loss_per_epoch = []
        avg_acc_per_epoch = []
        with SetPrintMode(hidden=not verbose):
            for epoch in range(0, cfg.epochs):
                print('Epoch' + str(epoch))
                self.compute_novel_distance(discovered_samples)
                running_loss = 0.0
                running_corrects = 0.
                count = 0

                # Update stored kernel centres
                if (epoch % self.interval) == 0:

                    print("Updating kernel centres...")
                    self.centres = update_centres(self.backbone, self.centres, update_loader, self.batch, self.device)
                    print("Finding training set neighbours...")
                    self.centres = self.centres.cpu()
                    neighbours_tr = find_neighbours(self.num_neighbours, self.centres)
                    self.centres = self.centres.to(self.device)
                    self.classifier.centres = self.centres
                    print("Finished update!")

                self.backbone.train()
                self.classifier.train()

                if verbose:
                    pbar = tqdm(train_loader_with_index, ncols=80)
                else:
                    pbar = train_loader_with_index

                for batch, data in enumerate(pbar):
                    inputs, real_labels, indices = data
                    labels = target_mapping_func(real_labels)
                    indices = torch.tensor([index_mapping_func(indice.item()) for indice in indices])
                    # print(f'indices {indices}')
                    count += inputs.size(0)
                    
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    indices = indices.to(self.device)

                    optimizer.zero_grad()

                    features = self.backbone(inputs)
                    outputs = self.classifier(features, neighbours_tr[indices, :])
                    _, preds = torch.max(outputs, 1)

                    log_probability = outputs

                    loss = criterion(log_probability, labels)

                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if verbose:
                        pbar.set_postfix(loss=float(running_loss)/count, 
                                         acc=float(running_corrects)/count,
                                         epoch=epoch)
                
                avg_loss = float(running_loss)/count
                avg_acc = float(running_corrects)/count
                avg_loss_per_epoch.append(avg_loss)
                avg_acc_per_epoch.append(avg_acc)
                scheduler.step()
            print("Updating kernel centres (final time)...")
            self.centres = update_centres(self.backbone, self.centres, update_loader, self.batch, self.device)
            self.classifier.centres = self.centres
            print(f"Average Loss {avg_loss}, Accuracy {avg_acc}")
        ckpt_dict = {
            'backbone' : self.backbone.state_dict(),
            'classifier' : self.classifier.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'discovered_samples' : discovered_samples,
            'discovered_classes' : discovered_classes,
            'loss_curve' : avg_loss_per_epoch,
            'acc_curve' : avg_acc_per_epoch,
            'num_train' : self.num_train,
            'centres' : self.centres,
            'centre_labels': self.centre_labels
        }
        return ckpt_dict

    def _get_classifier(self, discovered_classes):
        # Create Gaussian kernel classifier
        kernel_classifier = GaussianKernels(
            len(discovered_classes), self.num_neighbours, self.num_train, self.sigma, self.centres, self.centre_labels)
        kernel_classifier = kernel_classifier.to(self.device)
        return kernel_classifier

    def _load_ckpt_dict(self, ckpt_dict):
        self.num_train = ckpt_dict['num_train']
        self.centres = ckpt_dict['centres']
        self.centre_labels = ckpt_dict['centre_labels']

        self.classifier = self._get_classifier(ckpt_dict['discovered_classes']).to(self.device)
        self.classifier.load_state_dict(ckpt_dict['classifier'])
        self.backbone.load_state_dict(ckpt_dict['backbone'])

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
