import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm
import copy

import os

import models
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list, get_index_mapping_func
from utils import IndexDataset
from distance import eu_distance, cos_distance, eu_distance_batch, cos_distance_batch
from deep_metric import *

from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX, PRETRAINED_MODEL_PATH

import libmr
import math

def get_trainer_machine(training_method, train_mode, trainset_info, trainer_config):
    """Return a TrainerMachine object
        Args:
            training_method (str) : The training method
            train_mode (str) : The training mode (with/without finetune)
            trainset_info (TrainsetInfo) : The details about training set
            trainer_config (dict) : The details about hyperparameter and etc.
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
    
    return trainer_machine_class(train_mode, trainset_info, trainer_config)

class TrainerMachine(object):
    """Abstract class"""
    def __init__(self, train_mode, trainset_info, trainer_config):
        super(TrainerMachine, self).__init__()
        self.train_mode = train_mode
        self.trainset_info = trainset_info
        
        self.train_config = trainer_config['train']
        self.finetune_config = trainer_config['finetune']
        
        self.batch = trainer_config['batch']
        self.workers = trainer_config['workers']
        self.device = trainer_config['device']
        
        self.trainer_config  = trainer_config
        self.backbone = self._get_backbone_network(trainer_config['backbone']).to(self.device)
        self.feature_dim = trainer_config['feature_dim']
        self.classifier = None # Initialize per train()/finetune() call.
        
        self.ckpt_dict = None # A dictionary that holds all checkpoint information

    def train(self, discovered_samples, discovered_classes, ckpt_path=None, verbose=False):
        """Perform the train step
        """
        if os.path.exists(ckpt_path):
            print("Load from pre-existing ckpt. No training will be performed.")
            self.ckpt_dict = torch.load(ckpt_path)
            self.num_train = self.ckpt_dict['num_train']
            self.centres = self.ckpt_dict['centres']
            self.centre_labels = self.ckpt_dict['centre_labels']
            self._load_ckpt_dict(self.ckpt_dict)
        else:
            print(f"First time training the model. Ckpt will be saved at {ckpt_path}")
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
            if self.train_mode == "no_finetune":
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
    
    def eval_closed_set(self, discovered_classes, test_dataset, result_path=None, verbose=True):
        """ Performing closed set evaluation
        """
        if os.path.exists(result_path):
            print("Closed set result already saved.")
            self.closed_set_result = torch.load(result_path)
        else:
            self.closed_set_result = self._eval_closed_set_helper(discovered_classes,
                                                                  test_dataset,
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

    def get_trainloader(self, discovered_samples, shuffle=True):
        return get_subset_loader(self.trainset_info.train_dataset,
                                 discovered_samples,
                                 None, # No target transform
                                 batch_size=self.batch,
                                 shuffle=shuffle,
                                 workers=self.workers)
        
    def _get_target_mapp_func(self, discovered_classes):
        return get_target_mapping_func_for_tensor(self.trainset_info.classes,
                                                  discovered_classes,
                                                  self.trainset_info.open_classes,
                                                  device=self.device)

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
        trainloader = self.get_trainloader(discovered_samples)

        criterion = torch.nn.NLLLoss(reduction='mean')

        avg_loss_per_epoch = []
        avg_acc_per_epoch = []
        with SetPrintMode(hidden=not verbose):
            for epoch in range(0, cfg.epochs):
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
        ckpt_dict = {
            'backbone' : self.backbone.state_dict(),
            'classifier' : self.classifier.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'discovered_samples' : discovered_samples,
            'discovered_classes' : discovered_classes,
            'loss_curve' : avg_loss_per_epoch,
            'acc_curve' : avg_acc_per_epoch
        }
        return ckpt_dict
    
    def _eval_closed_set_helper(self, discovered_classes, test_dataset, verbose=True):
        """Test the model/classifier and return the acc in ckpt_dict
        """
        self.backbone.eval()
        self.classifier.eval()
        
        target_mapping_func = self._get_target_mapp_func(discovered_classes)
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
                
            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, real_labels = data

                    inputs = inputs.to(self.device)
                    labels = target_mapping_func(real_labels.to(self.device))

                    outputs = self.classifier(self.backbone(inputs))
                    _, preds = torch.max(outputs, 1)

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
            test_acc = performance_dict['corrects'] / performance_dict['count']
            seen_rate = 1. - performance_dict['not_seen'] / performance_dict['count']
            
            
            print(f"Test => "
                  f"Closed set test Acc {test_acc}, "
                  f"Discovered precentage {seen_rate}")

            print(f"Test Accuracy {test_acc}.")
        test_dict = {
            'acc' : test_acc,
            'seen' : seen_rate
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
    
    def _train_helper(self, cfg, discovered_samples, discovered_classes, verbose=True):
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


#     def _eval(self, model, test_dataset, discovered_classes, verbose=True):
#         self._eval_mode()

#         # Update self.thresholds_checkpoints
#         # assert self.round not in self.thresholds_checkpoints.keys()
#         # Caveat: Currently, the open_set_score is updated in the open_set_prediction function. Yet the grouth_truth is updated in self._eval()
#         self.thresholds_checkpoints[self.round] = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
#                                                    'real_labels' : [], # The real labels for CIFAR100 or other datasets.
#                                                    'open_set_score' : [], # Higher the score, more likely to be open set
#                                                    'closed_predicted' : [], # If fail the open set detection, then what's the predicted closed set label (network output)?
#                                                    'closed_predicted_real' : [], # If fail the open set detection, then what's the predicted closed set label (real labels)?
#                                                    'closed_argmax_prob' : [], # If fail the open set detection, then what's the probability for predicted closed set class (real labels)?
#                                                    'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
#                                                    'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
#                                                    'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
#                                                    'learningloss_pred_loss' : [], # Loss predicted by learning loss
#                                                    'actual_loss' : [], # actual losses
#                                                   } # A list of dictionary

#         target_mapping_func = self._get_target_mapp_func(discovered_classes)
#         target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(discovered_classes) # Only for transforming predicted label (in network indices) to real indices
#         dataloader = get_loader(test_dataset,
#                                 # self._get_target_mapp_func(discovered_classes),
#                                 None,
#                                 shuffle=False,
#                                 batch_size=self.config.batch,
#                                 workers=self.config.workers)
        
#         open_set_prediction = self._get_open_set_pred_func()

#         if self.icalr_strategy == 'proto':
#             cur_features = []
#             def forward_hook_func(module, inputs, outputs):
#                 cur_features.append(inputs[0])
#             handle = model.fc.register_forward_hook(forward_hook_func)

#         with SetPrintMode(hidden=not verbose):
#             if verbose:
#                 pbar = tqdm(dataloader, ncols=80)
#             else:
#                 pbar = dataloader

#             performance_dict = {'train_class_acc' : {'corrects' : 0., 'not_seen' : 0., 'count' : 0.}, # Accuracy of all non-hold out open class examples. If some classes not seen yet, accuracy = 0
#                                 'unseen_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all unseen open class examples.
#                                 'overall_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all examples. Counting accuracy of unseen open examples.
#                                 'holdout_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of hold out open class examples
#                                 'seen_closed_acc' : {'open' : 0., 'corrects' : 0., 'count' : 0.}, # Accuracy of seen class examples
#                                 'all_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all open class examples (unseen open + hold-out open)
#                                 }

#             if self.config.debug:
#                 import pdb; pdb.set_trace()  # breakpoint a8dffc68 //
                
#             with torch.no_grad():
#                 for batch, data in enumerate(pbar):
#                     inputs, real_labels = data
                    
#                     inputs = inputs.to(self.device)
#                     labels = target_mapping_func(real_labels.to(self.device))
#                     labels_for_openset_pred = torch.where(
#                                                   labels == OPEN_CLASS_INDEX,
#                                                   torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(labels.device),
#                                                   labels
#                                               ) # This change hold out open set examples' indices to unseen open set examples indices
#                     label_for_learnloss = labels.clone()
#                     label_for_learnloss[label_for_learnloss<0] = 0
                    
#                     outputs = model(inputs)
#                     if self.icalr_strategy == 'proto':
#                         preds = open_set_prediction(outputs, inputs=inputs, features=cur_features[0], label_for_learnloss=label_for_learnloss) # Open set index == UNDISCOVERED_CLASS_INDEX
#                         cur_features = []
#                     elif self.icalr_strategy in ['naive', 'smooth']:
#                         preds = open_set_prediction(outputs, inputs=inputs, features=None, label_for_learnloss=label_for_learnloss) # Open set index == UNDISCOVERED_CLASS_INDEX
#                     # loss = open_set_criterion(outputs, labels)

#                     # labels_for_ground_truth = torch.where(
#                     #                               (labels != OPEN_CLASS_INDEX) & (labels != UNDISCOVERED_CLASS_INDEX),
#                     #                               torch.LongTensor([0]).to(labels.device),
#                     #                               labels
#                     #                           ) # This change hold out open set examples' indices to unseen open set examples indices

#                     self.thresholds_checkpoints[self.round]['ground_truth'] += labels.tolist()
#                     self.thresholds_checkpoints[self.round]['real_labels'] += real_labels.tolist()
#                     # statistics
#                     # running_loss += loss.item() * inputs.size(0)
#                     performance_dict['overall_acc']['count'] += inputs.size(0)
#                     performance_dict['overall_acc']['corrects'] += float(torch.sum(preds == labels_for_openset_pred.data))
                    
#                     undiscovered_open_indices = labels == UNDISCOVERED_CLASS_INDEX
#                     discovered_closed_indices = labels >= 0
#                     hold_out_open_indices = labels == OPEN_CLASS_INDEX
#                     unlabeled_pool_class_indices = undiscovered_open_indices | discovered_closed_indices
#                     all_open_indices = undiscovered_open_indices | hold_out_open_indices
#                     assert torch.sum(undiscovered_open_indices & discovered_closed_indices & hold_out_open_indices) == 0
                    
#                     performance_dict['train_class_acc']['count'] += float(torch.sum(unlabeled_pool_class_indices))
#                     performance_dict['unseen_open_acc']['count'] += float(torch.sum(undiscovered_open_indices))
#                     performance_dict['holdout_open_acc']['count'] += float(torch.sum(hold_out_open_indices))
#                     performance_dict['seen_closed_acc']['count'] += float(torch.sum(discovered_closed_indices))
#                     performance_dict['all_open_acc']['count'] += float(torch.sum(all_open_indices))

#                     performance_dict['train_class_acc']['not_seen'] += torch.sum(
#                                                                            undiscovered_open_indices
#                                                                        ).float()
#                     performance_dict['train_class_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                discovered_closed_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['unseen_open_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                undiscovered_open_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['holdout_open_acc']['corrects'] += torch.sum(
#                                                                             torch.masked_select(
#                                                                                 (preds==labels_for_openset_pred.data),
#                                                                                 hold_out_open_indices
#                                                                             )
#                                                                         ).float()
#                     performance_dict['seen_closed_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                discovered_closed_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['seen_closed_acc']['open'] += torch.sum(
#                                                                        torch.masked_select(
#                                                                            (preds==UNDISCOVERED_CLASS_INDEX),
#                                                                            discovered_closed_indices
#                                                                        )
#                                                                    ).float()
#                     performance_dict['all_open_acc']['corrects'] += torch.sum(
#                                                                         torch.masked_select(
#                                                                             (preds==labels_for_openset_pred.data),
#                                                                             all_open_indices
#                                                                         )
#                                                                     ).float()

#                     batch_result = get_acc_from_performance_dict(performance_dict)
#                     if verbose:
#                         pbar.set_postfix(batch_result)

#                 epoch_result = get_acc_from_performance_dict(performance_dict)
#                 train_class_acc = epoch_result['train_class_acc']
#                 unseen_open_acc = epoch_result['unseen_open_acc']
#                 overall_acc = epoch_result['overall_acc']
#                 holdout_open_acc = epoch_result['holdout_open_acc']
#                 seen_closed_acc = epoch_result['seen_closed_acc']
#                 all_open_acc = epoch_result['all_open_acc']

#                 overall_count = performance_dict['overall_acc']['count']
#                 train_class_count = performance_dict['train_class_acc']['count']
#                 unseen_open_count = performance_dict['unseen_open_acc']['count']
#                 holdout_open_count = performance_dict['holdout_open_acc']['count']
#                 seen_closed_count = performance_dict['seen_closed_acc']['count']
#                 all_open_count = performance_dict['all_open_acc']['count']

#                 train_class_corrects = performance_dict['train_class_acc']['corrects']
#                 unseen_open_corrects = performance_dict['unseen_open_acc']['corrects']
#                 seen_closed_corrects = performance_dict['seen_closed_acc']['corrects']

#                 train_class_notseen = performance_dict['train_class_acc']['not_seen']
#                 seen_closed_open = performance_dict['seen_closed_acc']['open']

#             self.thresholds_checkpoints[self.round]['closed_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['closed_predicted'])
#             self.thresholds_checkpoints[self.round]['open_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['open_predicted'])
            
#             print(f"Test => "
#                   f"Training Class Acc {train_class_acc}, "
#                   f"Hold-out Open-Set Acc {holdout_open_acc}")
#             print(f"Details => "
#                   f"Overall Acc {overall_acc}, "
#                   f"Overall Open-Set Acc {all_open_acc}, Overall Seen-Class Acc {seen_closed_acc}")
#             print(f"Training Classes Accuracy Details => "
#                   f"[{train_class_notseen}/{train_class_count}] not in seen classes, "
#                   f"and for seen class samples [{seen_closed_corrects}/{seen_closed_count} corrects | [{seen_closed_open}/{seen_closed_count}] wrongly as open set]")
        
#         if self.icalr_strategy == 'proto': handle.remove()
#         return epoch_result


#     def _get_open_set_pred_func(self):
#         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
#         if self.config.network_eval_mode == 'threshold':
#             # assert self.config.threshold_metric == "softmax"
#             threshold = self.config.network_eval_threshold
#         elif self.config.network_eval_mode == 'dynamic_threshold':
#             # raise NotImplementedError()
#             assert type(self.log) == list
#             if len(self.log) == 0:
#                 # First round, use default threshold
#                 threshold = self.config.network_eval_threshold
#                 print(f"First round. Use default threshold {threshold}")
#             else:
#                 try:
#                     threshold = trainer_machine.get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
#                 except trainer_machine.NoSeenClassException:
#                     # Error when no new instances from seen class
#                     threshold = self.config.network_eval_threshold
#                     print(f"No seen class instances. Threshold set to {threshold}")
#                 except trainer_machine.NoUnseenClassException:
#                     threshold = self.config.network_eval_threshold
#                     print(f"No unseen class instances. Threshold set to {threshold}")
#                 else:
#                     print(f"Threshold set to {threshold} based on all existing instances.")
#         elif self.config.network_eval_mode in ['pseuopen_threshold']:
#             # assert hasattr(self, 'pseuopen_threshold')
#             # print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
#             # threshold = self.pseuopen_threshold
#             raise NotImplementedError()

#         def open_set_prediction(outputs, inputs=None, features=None, label_for_learnloss=None):
#             if self.icalr_strategy == 'proto':
#                 # First step is to normalize the features
#                 features = features / ((features*features).sum(1) ** 0.5).unsqueeze(1)

#                 difference = (features.unsqueeze(1) - self.proto.unsqueeze(0)).abs()
#                 distances = (difference*difference).sum(2)**0.5
#                 outputs = -distances

#             softmax_outputs = F.softmax(outputs, dim=1)
#             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#             if self.config.threshold_metric == 'softmax':
#                 scores = softmax_max
#             elif self.config.threshold_metric == 'entropy':
#                 neg_entropy = softmax_outputs*softmax_outputs.log()
#                 neg_entropy[softmax_outputs < 1e-5] = 0
#                 scores = neg_entropy.sum(dim=1) # negative entropy!

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
#             self.thresholds_checkpoints[self.round]['actual_loss'] += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()


#             preds = torch.where(scores < threshold,
#                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
#                                 softmax_preds)
#             return preds
#         return open_set_prediction

#     def get_open_score_func(self):
#         if self.icalr_strategy == 'proto':
#             self.cur_features = []
#             def forward_hook_func(module, inputs, outputs):
#                 self.cur_features.append(inputs[0])
#             self.handle = self.model.fc.register_forward_hook(forward_hook_func)
#             def open_score_func(inputs):
#                 _ = self.model(inputs)
#                 features = self.cur_features[0]
#                 features = features / ((features*features).sum(1) ** 0.5).unsqueeze(1)

#                 difference = (features.unsqueeze(1) - self.proto.unsqueeze(0)).abs()
#                 distances = (difference*difference).sum(2)**0.5

#                 softmax_outputs = F.softmax(-distances, dim=1)
#                 softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#                 if self.config.threshold_metric == 'softmax':
#                     scores = softmax_max
#                 elif self.config.threshold_metric == 'entropy':
#                     neg_entropy = softmax_outputs*softmax_outputs.log()
#                     neg_entropy[softmax_outputs < 1e-5] = 0
#                     scores = neg_entropy.sum(dim=1) # negative entropy!
#                 self.cur_features = []
#                 return -scores
#             return open_score_func
#         else:
#             def open_score_func(inputs):
#                 outputs = self.model(inputs)
#                 softmax_outputs = F.softmax(outputs, dim=1)
#                 softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#                 if self.config.threshold_metric == 'softmax':
#                     scores = softmax_max
#                 elif self.config.threshold_metric == 'entropy':
#                     neg_entropy = softmax_outputs*softmax_outputs.log()
#                     neg_entropy[softmax_outputs < 1e-5] = 0
#                     scores = neg_entropy.sum(dim=1) # negative entropy!
#                 self.cur_features = []
#                 return -scores
#             return open_score_func


# class Network(TrainerMachine):
#     def __init__(self, *args, **kwargs):
#         super(Network, self).__init__(*args, **kwargs)
#         self.epoch = 0
#         self.round = 0
#         self.max_epochs = self.config.epochs
#         self.device = self.config.device
#         self.model = self._get_network_model()
#         self.info_collector_class = BasicInfoCollector
#         self.criterion_class = nn.CrossEntropyLoss

#         # Current training state. Update in train_and_eval(). Used in other functions.
#         self.discovered_classes = set()

#     def get_logging_str(self, verbose=True):
#         if verbose:
#             setting_str = "Not Implemented"
#         else:
#             setting_str = config.threshold_metric

#     def _get_criterion(self, dataloader, target_mapping_func, discovered_classes=set(), criterion_class=nn.CrossEntropyLoss):
#         assert discovered_classes.__len__() > 0
#         assert self.config.class_weight in ['uniform', 'imbal']
#         if self.config.class_weight == 'uniform':
#             weight = None
#             print('Using uniform class weight.')
#         elif self.config.class_weight == 'imbal':
#             weight = torch.zeros(len(discovered_classes)).to(self.device)
#             total = 0.0
#             for _, data in enumerate(tqdm(dataloader, ncols=80)):
#                 _, real_labels = data
#                 labels = target_mapping_func(real_labels)
#                 for label_i in labels:
#                     weight[label_i] += 1.
#                     total += 1.
#             weight = total / weight
#             weight = weight / weight.min() # TODO: Figure out whether or not need this min()
#             weight[weight > 10.] = 10.
#             class_weight_info = {}
#             unmap_dict = get_target_unmapping_dict(self.train_instance.classes, discovered_classes)
#             for i, w_i in enumerate(weight):
#                 class_weight_info[unmap_dict[i]] = float(w_i)
#             print(f'Using class weight: {class_weight_info}')
#         return criterion_class(weight=weight)

#     def _train(self, model, discovered_samples, discovered_classes, start_epoch=0):
#         self._train_mode()
#         target_mapping_func = self._get_target_mapp_func(discovered_classes)
#         self.dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
#                                                   list(discovered_samples),
#                                                   [], # TODO: Make a validation set
#                                                   None, # No target transform is made
#                                                   batch_size=self.config.batch,
#                                                   workers=self.config.workers)
        
#         self._update_last_layer(model, len(discovered_classes), device=self.device)
#         optimizer = self._get_network_optimizer(model)
#         scheduler = self._get_network_scheduler(optimizer)

#         self.criterion = self._get_criterion(self.dataloaders['train'],
#                                              target_mapping_func,
#                                              discovered_classes=discovered_classes,
#                                              criterion_class=self.criterion_class)

#         with SetPrintMode(hidden=not self.config.verbose):
#             train_loss, train_acc = train_epochs(
#                                         model,
#                                         self.dataloaders,
#                                         optimizer,
#                                         scheduler,
#                                         self.criterion,
#                                         target_mapping_func,
#                                         device=self.device,
#                                         start_epoch=start_epoch,
#                                         max_epochs=self.max_epochs,
#                                         verbose=self.config.verbose,
#                                     )
#         print(f"Train => {self.round} round => "
#               f"Loss {train_loss}, Accuracy {train_acc}")
#         return train_loss, train_acc

#     def train_then_eval(self, discovered_samples, discovered_classes, test_dataset, eval_verbose=True, start_epoch=0):
#         self.round += 1
#         train_loss, train_acc = self._train(self.model, discovered_samples, discovered_classes, start_epoch=0)  
#         eval_results = self._eval(self.model, test_dataset, discovered_classes, verbose=eval_verbose)
#         return train_loss, train_acc, eval_results

#     def _eval(self, model, test_dataset, discovered_classes, verbose=True):
#         self._eval_mode()

#         # Update self.thresholds_checkpoints
#         # assert self.round not in self.thresholds_checkpoints.keys()
#         # Caveat: Currently, the open_set_score is updated in the open_set_prediction function. Yet the grouth_truth is updated in self._eval()
#         self.thresholds_checkpoints[self.round] = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
#                                                    'real_labels' : [], # The real labels for CIFAR100 or other datasets.
#                                                    'open_set_score' : [], # Higher the score, more likely to be open set
#                                                    'closed_predicted' : [], # If fail the open set detection, then what's the predicted closed set label (network output)?
#                                                    'closed_predicted_real' : [], # If fail the open set detection, then what's the predicted closed set label (real labels)?
#                                                    'closed_argmax_prob' : [], # If fail the open set detection, then what's the probability for predicted closed set class (real labels)?
#                                                    'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
#                                                    'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
#                                                    'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
#                                                   } # A list of dictionary

#         target_mapping_func = self._get_target_mapp_func(discovered_classes)
#         target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(discovered_classes) # Only for transforming predicted label (in network indices) to real indices
#         dataloader = get_loader(test_dataset,
#                                 # self._get_target_mapp_func(discovered_classes),
#                                 None,
#                                 shuffle=False,
#                                 batch_size=self.config.batch,
#                                 workers=self.config.workers)
        
#         open_set_prediction = self._get_open_set_pred_func()

#         with SetPrintMode(hidden=not verbose):
#             if verbose:
#                 pbar = tqdm(dataloader, ncols=80)
#             else:
#                 pbar = dataloader

#             performance_dict = {'train_class_acc' : {'corrects' : 0., 'not_seen' : 0., 'count' : 0.}, # Accuracy of all non-hold out open class examples. If some classes not seen yet, accuracy = 0
#                                 'unseen_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all unseen open class examples.
#                                 'overall_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all examples. Counting accuracy of unseen open examples.
#                                 'holdout_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of hold out open class examples
#                                 'seen_closed_acc' : {'open' : 0., 'corrects' : 0., 'count' : 0.}, # Accuracy of seen class examples
#                                 'all_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all open class examples (unseen open + hold-out open)
#                                 }

#             with torch.no_grad():
#                 for batch, data in enumerate(pbar):
#                     inputs, real_labels = data
                    
#                     inputs = inputs.to(self.device)
#                     labels = target_mapping_func(real_labels.to(self.device))
#                     labels_for_openset_pred = torch.where(
#                                                   labels == OPEN_CLASS_INDEX,
#                                                   torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(labels.device),
#                                                   labels
#                                               ) # This change hold out open set examples' indices to unseen open set examples indices

#                     outputs = model(inputs)
#                     preds = open_set_prediction(outputs, inputs=inputs) # Open set index == UNDISCOVERED_CLASS_INDEX

#                     self.thresholds_checkpoints[self.round]['ground_truth'] += labels.tolist()
#                     self.thresholds_checkpoints[self.round]['real_labels'] += real_labels.tolist()
#                     # statistics
#                     # running_loss += loss.item() * inputs.size(0)
#                     performance_dict['overall_acc']['count'] += inputs.size(0)
#                     performance_dict['overall_acc']['corrects'] += float(torch.sum(preds == labels_for_openset_pred.data))
                    
#                     undiscovered_open_indices = labels == UNDISCOVERED_CLASS_INDEX
#                     discovered_closed_indices = labels >= 0
#                     hold_out_open_indices = labels == OPEN_CLASS_INDEX
#                     unlabeled_pool_class_indices = undiscovered_open_indices | discovered_closed_indices
#                     all_open_indices = undiscovered_open_indices | hold_out_open_indices
#                     assert torch.sum(undiscovered_open_indices & discovered_closed_indices & hold_out_open_indices) == 0
                    
#                     performance_dict['train_class_acc']['count'] += float(torch.sum(unlabeled_pool_class_indices))
#                     performance_dict['unseen_open_acc']['count'] += float(torch.sum(undiscovered_open_indices))
#                     performance_dict['holdout_open_acc']['count'] += float(torch.sum(hold_out_open_indices))
#                     performance_dict['seen_closed_acc']['count'] += float(torch.sum(discovered_closed_indices))
#                     performance_dict['all_open_acc']['count'] += float(torch.sum(all_open_indices))

#                     performance_dict['train_class_acc']['not_seen'] += torch.sum(
#                                                                            undiscovered_open_indices
#                                                                        ).float()
#                     performance_dict['train_class_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                discovered_closed_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['unseen_open_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                undiscovered_open_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['holdout_open_acc']['corrects'] += torch.sum(
#                                                                             torch.masked_select(
#                                                                                 (preds==labels_for_openset_pred.data),
#                                                                                 hold_out_open_indices
#                                                                             )
#                                                                         ).float()
#                     performance_dict['seen_closed_acc']['corrects'] += torch.sum(
#                                                                            torch.masked_select(
#                                                                                (preds==labels.data),
#                                                                                discovered_closed_indices
#                                                                            )
#                                                                        ).float()
#                     performance_dict['seen_closed_acc']['open'] += torch.sum(
#                                                                        torch.masked_select(
#                                                                            (preds==UNDISCOVERED_CLASS_INDEX),
#                                                                            discovered_closed_indices
#                                                                        )
#                                                                    ).float()
#                     performance_dict['all_open_acc']['corrects'] += torch.sum(
#                                                                         torch.masked_select(
#                                                                             (preds==labels_for_openset_pred.data),
#                                                                             all_open_indices
#                                                                         )
#                                                                     ).float()

#                     batch_result = get_acc_from_performance_dict(performance_dict)
#                     if verbose:
#                         pbar.set_postfix(batch_result)

#                 epoch_result = get_acc_from_performance_dict(performance_dict)
#                 train_class_acc = epoch_result['train_class_acc']
#                 unseen_open_acc = epoch_result['unseen_open_acc']
#                 overall_acc = epoch_result['overall_acc']
#                 holdout_open_acc = epoch_result['holdout_open_acc']
#                 seen_closed_acc = epoch_result['seen_closed_acc']
#                 all_open_acc = epoch_result['all_open_acc']

#                 overall_count = performance_dict['overall_acc']['count']
#                 train_class_count = performance_dict['train_class_acc']['count']
#                 unseen_open_count = performance_dict['unseen_open_acc']['count']
#                 holdout_open_count = performance_dict['holdout_open_acc']['count']
#                 seen_closed_count = performance_dict['seen_closed_acc']['count']
#                 all_open_count = performance_dict['all_open_acc']['count']

#                 train_class_corrects = performance_dict['train_class_acc']['corrects']
#                 unseen_open_corrects = performance_dict['unseen_open_acc']['corrects']
#                 seen_closed_corrects = performance_dict['seen_closed_acc']['corrects']

#                 train_class_notseen = performance_dict['train_class_acc']['not_seen']
#                 seen_closed_open = performance_dict['seen_closed_acc']['open']

#             self.thresholds_checkpoints[self.round]['closed_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['closed_predicted'])
#             self.thresholds_checkpoints[self.round]['open_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['open_predicted'])
#             print(f"Test => "
#                   f"Training Class Acc {train_class_acc}, "
#                   f"Hold-out Open-Set Acc {holdout_open_acc}")
#             print(f"Details => "
#                   f"Overall Acc {overall_acc}, "
#                   f"Overall Open-Set Acc {all_open_acc}, Overall Seen-Class Acc {seen_closed_acc}")
#             print(f"Training Classes Accuracy Details => "
#                   f"[{train_class_notseen}/{train_class_count}] not in seen classes, "
#                   f"and for seen class samples [{seen_closed_corrects}/{seen_closed_count} corrects | [{seen_closed_open}/{seen_closed_count}] wrongly as open set]")
#         return epoch_result


#     def _get_open_set_pred_func(self):
#         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
#         if self.config.network_eval_mode == 'threshold':
#             assert self.config.threshold_metric == "softmax"
#             threshold = self.config.network_eval_threshold
#         elif self.config.network_eval_mode == 'dynamic_threshold':
#             assert type(self.log) == list
#             if len(self.log) == 0:
#                 # First round, use default threshold
#                 threshold = self.config.network_eval_threshold
#                 print(f"First round. Use default threshold {threshold}")
#             else:
#                 try:
#                     threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
#                 except NoSeenClassException:
#                     # Error when no new instances from seen class
#                     threshold = self.config.network_eval_threshold
#                     print(f"No seen class instances. Threshold set to {threshold}")
#                 except NoUnseenClassException:
#                     threshold = self.config.network_eval_threshold
#                     print(f"No unseen class instances. Threshold set to {threshold}")
#                 else:
#                     print(f"Threshold set to {threshold} based on all existing instances.")
#         elif self.config.network_eval_mode in ['pseuopen_threshold']:
#             assert hasattr(self, 'pseuopen_threshold')
#             print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
#             threshold = self.pseuopen_threshold

#         def open_set_prediction(outputs, inputs=None):
#             softmax_outputs = F.softmax(outputs, dim=1)
#             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#             if self.config.threshold_metric == 'softmax':
#                 scores = softmax_max
#             elif self.config.threshold_metric == 'entropy':
#                 neg_entropy = softmax_outputs*softmax_outputs.log()
#                 neg_entropy[softmax_outputs < 1e-5] = 0
#                 scores = neg_entropy.sum(dim=1) # negative entropy!

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
            
#             preds = torch.where(scores < threshold,
#                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
#                                 softmax_preds)
#             return preds
#         return open_set_prediction

#     def get_open_score_func(self):
#         def open_score_func(inputs):
#             outputs = self.model(inputs)
#             softmax_outputs = F.softmax(outputs, dim=1)
#             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#             if self.config.threshold_metric == 'softmax':
#                 scores = softmax_max
#             elif self.config.threshold_metric == 'entropy':
#                 neg_entropy = softmax_outputs*softmax_outputs.log()
#                 neg_entropy[softmax_outputs < 1e-5] = 0
#                 scores = neg_entropy.sum(dim=1) # negative entropy!
#             return -scores
#         return open_score_func

#     def _train_mode(self):
#         self.model.train()

#     def _eval_mode(self):
#         self.model.eval()

#     def _get_network_model(self):
#         """ Get the regular softmax network model
#         """
#         model = getattr(models, self.config.arch)(last_relu=False)
#         if self.config.pretrained != None:
#             state_dict = self._get_pretrained_model_state_dict()
#             model.load_state_dict(state_dict)
#             del state_dict
#         else:
#             print("Using random initialized model")
#         return model.to(self.device)

#     def _get_pretrained_model_state_dict(self):
#         if self.config.pretrained == 'CIFAR10':
#             if self.config.arch == 'ResNet50':
#                 print("Use pretrained ResNet50 model trained on cifar10 that achieved .86 acc")
#                 state_dict = torch.load(
#                     PRETRAINED_MODEL_PATH[self.config.pretrained][self.config.arch]
#                 )['net']
#                 new_state_dict = OrderedDict()
#                 for k, v in state_dict.items():
#                     if 'module.' == k[:7]:
#                         name = k[7:] # remove `module.`
#                     else:
#                         name = k
#                     if "linear." == name[:7]:
#                         name = "fc." + name[7:]
#                     new_state_dict[name] = v
#                 del state_dict
#                 return new_state_dict
#             else:
#                 raise ValueError("Pretrained Model not prepared")
#         else:
#             raise ValueError("Pretrained Model not prepared")
        
#     def _get_network_optimizer(self, model):
#         """ Get softmax network optimizer
#         """
#         if self.config.optim == 'sgd':
#             optim_module = optim.SGD
#             optim_param = {"lr" : self.config.lr, 
#                            "momentum" : self.config.momentum,
#                            "weight_decay" : 0 if self.config.wd == None else float(self.config.wd)}
#         elif self.config.optim == "adam":
#             optim_module = optim.Adam
#             optim_param = {"lr": self.config.lr, 
#                            "weight_decay": float(self.config.wd), 
#                            "amsgrad": self.config.amsgrad}
#         else:
#             raise ValueError("optim type not supported")
            
#         optimizer = optim_module(
#                         filter(lambda x : x.requires_grad, model.parameters()), 
#                         **optim_param
#                     )
#         return optimizer

#     def _get_network_scheduler(self, optimizer):
#         """ Get softmax network optimizer
#         """
#         if self.config.lr_decay_step == None:
#             decay_step = self.max_epochs
#         else:
#             decay_step = self.config.lr_decay_step
#         scheduler = lr_scheduler.StepLR(
#                         optimizer, 
#                         step_size=decay_step, 
#                         gamma=self.config.lr_decay_ratio
#                     )
#         return scheduler

#     def _update_last_layer(self, model, output_size, device='cuda'):
#         if "resnet" in self.config.arch.lower():
#             fd = int(model.fc.weight.size()[1])
#             model.fc = nn.Linear(fd, output_size)
#             model.fc.weight.data.normal_(0, 0.01)
#             model.fc.bias.data.zero_()
#             model.fc.to(device)
#         elif self.config.arch in ['classifier32', 'classifier32_instancenorm']:
#             fd = int(model.fc1.weight.size()[1])
#             model.fc1 = nn.Linear(fd, output_size)
#             model.fc1.weight.data.normal_(0, 0.01)
#             model.fc1.bias.data.zero_()
#             model.fc1.to(device)
#         else:
#             raise NotImplementedError()

# class ClusterNetwork(Network):
#     def __init__(self, *args, **kwargs):
#         # The output of self.model is raw 1-D feature vector x. cluster_predict(self.model, x) gives the prob score after normalization
#         super(ClusterNetwork, self).__init__(*args, **kwargs)
#         # Before everything else, remove the relu from layer4

#         self.network_output_size = self._get_network_output_size(self.model) # Before changing the architecture
                
#         self.clustering = self.config.clustering
#         if self.clustering == 'rbf_train':
#             self.gamma = self.config.rbf_gamma

#         if self.pseudo_open_set == None:
#             self.cluster_eval_threshold = self.config.cluster_eval_threshold
#         else:
#             self.cluster_eval_threshold = None

#         self.cluster_level = self.config.cluster_level

#         self.distance_metric = self.config.distance_metric
#         self.div_eu = self.config.div_eu
#         if self.distance_metric == 'eu':
#             self.distance_func = lambda a, b: eu_distance_batch(a,b,div_eu=self.div_eu)
#         elif self.distance_metric == 'eucos':
#             self.distance_func = lambda a, b: eu_distance_batch(a,b,div_eu=self.div_eu) + cos_distance_batch(a,b)
#         elif self.distance_metric == 'cos':
#             self.distance_func = cos_distance_batch
#         else:
#             raise NotImplementedError()

#         self.info_collector_class = lambda *args, **kwargs: ClusterInfoCollector(self.gamma, *args, **kwargs)
#         # self.criterion_class = lambda **dict: lambda x, y: nn.NLLLoss(weight=dict['weight'])(torch.log(x / x.sum(1, keepdim=True)), y)
#         # self.criterion_class = lambda **dict: lambda x, y: nn.CrossEntropyLoss(weight=dict['weight'])(torch.nn.LogSoftmax(dim=1)(x), y)
#         self.criterion_class = lambda **dict: lambda x, y: nn.CrossEntropyLoss(weight=dict['weight'])(x, y) # ? This version - yes

#     def _get_network_model(self):
#         """ Get a network model without last ReLU layer
#         """
#         model = getattr(models, self.config.arch)(last_relu=False)
#         if self.config.pretrained != None:
#             state_dict = self._get_pretrained_model_state_dict()
#             model.load_state_dict(state_dict)
#             del state_dict
#         else:
#             print("Using random initialized model")
#         return model.to(self.device)

#     def _get_network_output_size(self, model):
#         """Caveat: Can only be called before _update_last_layer()"""
#         if "resnet" in self.config.arch.lower():
#             return int(model.fc.weight.size()[1])
#         else:
#             raise NotImplementedError()

#     def _update_last_layer(self, model, output_size, device='cuda'):
#         assert hasattr(self, 'network_output_size')
#         if self.cluster_level == 'after_fc':
#             if "resnet" in self.config.arch.lower():
#                 fd = int(self.network_output_size)
#                 fc = nn.Linear(fd, output_size)
#                 fc.weight.data.normal_(0, 0.01)
#                 fc.bias.data.zero_()
#                 fc.to(device)
#             else:
#                 raise NotImplementedError()
#             cluster_layer = models.ClusterLayer(output_size, output_size, self.distance_func, gamma=self.gamma).to(device)
#             model.fc = torch.nn.Sequential(fc, cluster_layer)
#         else:
#             cluster_layer = models.ClusterLayer(output_size, self.network_output_size, self.distance_func, gamma=self.gamma).to(device)
#             model.fc = cluster_layer

#     def _get_open_set_pred_func(self):
#         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
#         if self.config.network_eval_mode == 'threshold':
#             assert self.config.threshold_metric == "softmax"
#             threshold = self.cluster_eval_threshold
#         elif self.config.network_eval_mode == 'dynamic_threshold':
#             assert type(self.log) == list
#             if len(self.log) == 0:
#                 # First round, use default threshold
#                 threshold = self.cluster_eval_threshold
#                 print(f"First round. Use default threshold {threshold}")
#             else:
#                 try:
#                     threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
#                 except NoSeenClassException:
#                     # Error when no new instances from seen class
#                     threshold = self.config.network_eval_threshold
#                     print(f"No seen class instances. Threshold set to {threshold}")
#                 except NoUnseenClassException:
#                     threshold = self.config.network_eval_threshold
#                     print(f"No unseen class instances. Threshold set to {threshold}")
#                 else:
#                     print(f"Threshold set to {threshold} based on all existing instances.")
#         elif self.config.network_eval_mode in ['pseuopen_threshold']:
#             assert hasattr(self, 'pseuopen_threshold')
#             print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
#             threshold = self.pseuopen_threshold

#         def open_set_prediction(outputs, inputs=None):
#             if self.config.threshold_metric == 'gaussian':
#                 maxs, preds = torch.max(outputs, 1)
#                 scores = (outputs.exp() / (math.pi * (1./self.gamma))**.5).mean(1)
#             elif self.config.threshold_metric == 'softmax':
#                 # outputs = outputs / outputs.sum(1, keepdim=True)
#                 outputs = torch.nn.functional.Softmax()(outputs)
#                 maxs, preds = torch.max(outputs, 1)
#                 scores = maxs
#             elif self.config.threshold_metric == 'entropy':
#                 # outputs = outputs / outputs.sum(1, keepdim=True)
#                 outputs = torch.nn.functional.Softmax()(outputs)
#                 maxs, preds = torch.max(outputs, 1)
#                 neg_entropy = outputs*outputs.log()
#                 neg_entropy[outputs < 1e-5] = 0
#                 scores = neg_entropy.sum(dim=1) # negative entropy!

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += maxs.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += preds.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += maxs.tolist()

#             preds = torch.where(scores < threshold,
#                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
#                                 preds)
#             return preds
#         return open_set_prediction

#     def get_open_score_func(self):
#         raise NotImplementedError()
#         # def open_score_func(inputs):
#         #     outputs = model(inputs)
#         #     softmax_outputs = F.softmax(outputs, dim=1)
#         #     softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#         #     if self.config.threshold_metric == 'softmax':
#         #         scores = softmax_max
#         #     elif self.config.threshold_metric == 'entropy':
#         #         scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
#         #     return scores
#         # return open_score_func

# def sigmoid_loss(weight=None, mode='mean'):
#     def loss_func(x, y):
#         sigmoids = 1 - nn.Sigmoid()(x)
#         sigmoids[torch.arange(x.shape[0]), y] = nn.Sigmoid()(x)[torch.arange(x.shape[0]), y]
#         # log_sigmoids_expand = log_sigmoids.unsqueeze(1).expand(-1,log_sigmoids.shape[1],-1)
#         sigmoids = torch.max(torch.Tensor([1e-10]).to(sigmoids.device), sigmoids)
#         log_sigmoids = sigmoids.log()
#         if weight != None:
#             log_sigmoids = weights * log_sigmoids
#         if mode == 'mean':
#             res = log_sigmoids.mean(1).mean()
#         elif mode == 'sum':
#             res = log_sigmoids.sum(1).mean()
#         else:
#             raise NotImplementedError()
#         return - res
#     return loss_func

# class SigmoidNetwork(Network):
#     def __init__(self, *args, **kwargs):
#         super(SigmoidNetwork, self).__init__(*args, **kwargs)
#         # Before everything else, remove the relu from layer4
        
#         if self.pseudo_open_set == None:
#             self.network_eval_threshold = self.config.network_eval_threshold
#         else:
#             self.network_eval_threshold = None

#         self.info_collector_class = lambda *args, **kwargs: SigmoidInfoCollector(*args, **kwargs)

#         self.sigmoid_train_mode = self.config.sigmoid_train_mode
#         self.criterion_class = lambda **dict: lambda x, y: sigmoid_loss(weight=dict['weight'], mode=self.sigmoid_train_mode)(x, y)

#     def _get_network_model(self):
#         """ Get the regular softmax network model
#         """
#         model = getattr(models, self.config.arch)(last_relu=False)
#         if self.config.pretrained != None:
#             state_dict = self._get_pretrained_model_state_dict()
#             model.load_state_dict(state_dict)
#             del state_dict
#         else:
#             print("Using random initialized model")
#         return model.to(self.device)

#     def _get_open_set_pred_func(self):
#         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
#         if self.config.network_eval_mode == 'threshold':
#             threshold = self.config.network_eval_threshold
#         elif self.config.network_eval_mode == 'dynamic_threshold':
#             assert type(self.log) == list
#             if len(self.log) == 0:
#                 # First round, use default threshold
#                 threshold = self.config.network_eval_threshold
#                 print(f"First round. Use default threshold {threshold}")
#             else:
#                 try:
#                     threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
#                 except NoSeenClassException:
#                     # Error when no new instances from seen class
#                     threshold = self.config.network_eval_threshold
#                     print(f"No seen class instances. Threshold set to {threshold}")
#                 except NoUnseenClassException:
#                     threshold = self.config.network_eval_threshold
#                     print(f"No unseen class instances. Threshold set to {threshold}")
#                 else:
#                     print(f"Threshold set to {threshold} based on all existing instances.")
#         elif self.config.network_eval_mode in ['pseuopen_threshold']:
#             assert hasattr(self, 'pseuopen_threshold')
#             print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
#             threshold = self.pseuopen_threshold

#         def open_set_prediction(outputs, inputs=None):
#             sigmoid_outputs = nn.Sigmoid()(outputs)
#             sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
#             softmax_outputs = F.softmax(outputs, dim=1)
#             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#             # if self.config.threshold_metric == 'softmax':
#             #     scores = softmax_max
#             # elif self.config.threshold_metric == 'entropy':
#             #     scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
#             scores = sigmoid_max

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
            
#             preds = torch.where(scores < threshold,
#                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
#                                 softmax_preds)
#             return preds
#         return open_set_prediction

#     def get_open_score_func(self):
#         def open_score_func(inputs):
#             outputs = self.model(inputs)
#             sigmoid_outputs = nn.Sigmoid()(outputs)
#             sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
#             scores = sigmoid_max
#             return -scores
#         return open_score_func

# class BinarySoftmaxNetwork(Network):
#     def __init__(self, *args, **kwargs):
#         super(BinarySoftmaxNetwork, self).__init__(*args, **kwargs)
#         # Before everything else, remove the relu from layer4
        
#         if self.pseudo_open_set == None:
#             self.network_eval_threshold = self.config.network_eval_threshold
#         else:
#             self.network_eval_threshold = None

#         self.info_collector_class = lambda *args, **kwargs: SigmoidInfoCollector(*args, **kwargs)

#     def _get_network_model(self):
#         """ Get the regular softmax network model
#         """
#         model = getattr(models, self.config.arch)(last_relu=False)
#         if self.config.pretrained != None:
#             state_dict = self._get_pretrained_model_state_dict()
#             model.load_state_dict(state_dict)
#             del state_dict
#         else:
#             print("Using random initialized model")
#         return model.to(self.device)

#     def _get_open_set_pred_func(self):
#         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
#         if self.config.network_eval_mode == 'threshold':
#             threshold = self.config.network_eval_threshold
#         elif self.config.network_eval_mode == 'dynamic_threshold':
#             assert type(self.log) == list
#             if len(self.log) == 0:
#                 # First round, use default threshold
#                 threshold = self.config.network_eval_threshold
#                 print(f"First round. Use default threshold {threshold}")
#             else:
#                 try:
#                     threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
#                 except NoSeenClassException:
#                     # Error when no new instances from seen class
#                     threshold = self.config.network_eval_threshold
#                     print(f"No seen class instances. Threshold set to {threshold}")
#                 except NoUnseenClassException:
#                     threshold = self.config.network_eval_threshold
#                     print(f"No unseen class instances. Threshold set to {threshold}")
#                 else:
#                     print(f"Threshold set to {threshold} based on all existing instances.")
#         elif self.config.network_eval_mode in ['pseuopen_threshold']:
#             assert hasattr(self, 'pseuopen_threshold')
#             print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
#             threshold = self.pseuopen_threshold

#         def open_set_prediction(outputs, inputs=None):
#             sigmoid_outputs = nn.Sigmoid()(outputs)
#             sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
#             softmax_outputs = F.softmax(outputs, dim=1)
#             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
#             # if self.config.threshold_metric == 'softmax':
#             #     scores = softmax_max
#             # elif self.config.threshold_metric == 'entropy':
#             #     scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
#             scores = sigmoid_max

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
            
#             preds = torch.where(scores < threshold,
#                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
#                                 softmax_preds)
#             return preds
#         return open_set_prediction

#     def get_open_score_func(self):
#         def open_score_func(inputs):
#             outputs = self.model(inputs)
#             sigmoid_outputs = nn.Sigmoid()(outputs)
#             sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
#             scores = sigmoid_max
#             return -scores
#         return open_score_func


# class OSDNNetwork(Network):
#     def __init__(self, *args, **kwargs):
#         # The Open Set Deep Network is essentially trained in the same way as a 
#         # regular softmax network. The difference occurs during the eval stage.
#         # So this is subclass from Network class, but has the eval function overwritten.
#         super(OSDNNetwork, self).__init__(*args, **kwargs)
#         # self.use_feature_before_fc = self.config.trainer in ['osdn_fc', 'osdn_modified_fc']

#         assert self.config.threshold_metric == 'softmax'

#         self.div_eu = self.config.div_eu
#         self.distance_metric = self.config.distance_metric
#         if self.distance_metric == 'eu':
#             self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu)
#         elif self.distance_metric == 'eucos':
#             self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu) + cos_distance(a,b)
#         elif self.distance_metric == 'cos':
#             self.distance_func = cos_distance
#         else:
#             raise NotImplementedError()


#         self.openmax_meta_learn = self.config.openmax_meta_learn
#         if self.openmax_meta_learn == None:
#             print("Using fixed OpenMax hyper")
#             if 'fixed' in self.config.weibull_tail_size:
#                 self.weibull_tail_size = int(self.config.weibull_tail_size.split("_")[-1])
#             else:
#                 raise NotImplementedError()

#             if 'fixed' in self.config.alpha_rank: 
#                 self.alpha_rank = int(self.config.alpha_rank.split('_')[-1])
#             else:
#                 raise NotImplementedError()

#             self.osdn_eval_threshold = self.config.osdn_eval_threshold
#         else:
#             print("Using meta learning on pseudo-open class examples")
#             self.weibull_tail_size = None
#             self.alpha_rank = None
#             self.osdn_eval_threshold = None
#             self.pseudo_open_set_metric = self.config.pseudo_open_set_metric

#         self.mav_features_selection = self.config.mav_features_selection
#         self.weibull_distributions = None # The dictionary that contains all weibull related information
#         # self.training_features = None # A dictionary holding the features of all examples

#     def _meta_learning(self, pseudo_open_model, full_train_set, pseudo_discovered_classes):
#         ''' Meta learning on self.curr_full_train_set and self.pseudo_open_set_classes
#             Pick and update the best openmax hyper including:
#                 self.weibull_tail_size
#                 self.alpha_rank
#                 self.osdn_eval_threshold
#         '''
#         print("Perform cross validation using pseudo open class..")
#         assert hasattr(self, 'dataloaders') # Should be updated after calling super._train()
#         training_features = self._gather_correct_features(pseudo_open_model,
#                                                           self.dataloaders['train'], # from super._train()
#                                                           discovered_classes=pseudo_discovered_classes,
#                                                           mav_features_selection=self.mav_features_selection) # A dict of (key: seen_class_indice, value: A list of feature vector that has correct prediction in this class)
        
#         from global_setting import OPENMAX_META_LEARN
#         meta_setting = OPENMAX_META_LEARN[self.openmax_meta_learn]
#         list_w_tail_size = meta_setting['weibull_tail_size']
#         list_alpha_rank = meta_setting['alpha_rank']
#         list_threshold = meta_setting['osdn_eval_threshold']

#         curr_train_set = torch.utils.data.Subset(
#                              self.train_instance.train_dataset,
#                              full_train_set
#                          )
#         meta_learn_result = [] # A list of tuple: (acc : float, hyper_setting : dict)
#         for tail_size in list_w_tail_size:
#             for alpha_rank in list_alpha_rank:
#                 for threshold in list_threshold:
#                     self.weibull_tail_size = tail_size
#                     self.alpha_rank = alpha_rank
#                     self.osdn_eval_threshold = threshold

#                     self.weibull_distributions = self._gather_weibull_distribution(
#                                                      training_features,
#                                                      distance_metric=self.distance_metric,
#                                                      weibull_tail_size=self.weibull_tail_size
#                                                  ) # A dict of (key: seen_class_indice, value: A per channel, MAV + weibull model)

#                     eval_result = self._eval(
#                                       pseudo_open_model,
#                                       curr_train_set,
#                                       pseudo_discovered_classes,
#                                       # verbose=False,
#                                       verbose=True,
#                                       training_features=training_features # If not None, can save time by not recomputing it
#                                   )
#                     if self.pseudo_open_set_metric == 'weighted':
#                         acc = eval_result['overall_acc']
#                     elif self.pseudo_open_set_metric == 'average':
#                         acc = (eval_result['seen_closed_acc'] + eval_result['all_open_acc']) / 2.
#                     elif self.pseudo_open_set_metric == '7_3':
#                         acc = 0.7 * eval_result['seen_closed_acc'] + 0.3 * eval_result['all_open_acc']
#                     else:
#                         raise NotImplementedError()
#                     meta_learn_result.append(
#                         { 'acc':acc,
#                           'tail':tail_size,
#                           'alpha':alpha_rank,
#                           'threshold':threshold
#                         }
#                     )
#         meta_learn_result.sort(reverse=True, key=lambda x:x['acc'])
#         print("Meta Learning result (sorted by acc):")
#         print(meta_learn_result)
#         best_res = meta_learn_result[0]
#         self.weibull_tail_size = best_res['tail']
#         self.alpha_rank = best_res['alpha']
#         self.osdn_eval_threshold = best_res['threshold']
#         print(f"Updated to : W_TAIL={self.weibull_tail_size}, ALPHA={best_res['alpha']}, THRESHOLD={best_res['threshold']}")

#     def _gather_correct_features(self, model, train_loader, discovered_classes=set(), mav_features_selection='correct'):
#         assert len(discovered_classes) > 0
#         assert mav_features_selection in ['correct', 'none_correct_then_all', 'all']
#         mapping_func = get_target_mapping_func(self.train_instance.classes,
#                                                discovered_classes,
#                                                self.train_instance.open_classes)
#         target_mapping_func = self._get_target_mapp_func(discovered_classes)
#         seen_class_softmax_indices = [mapping_func(i) for i in discovered_classes]

#         if mav_features_selection == 'correct':
#             print("Gather feature vectors for each class that are predicted correctly")
#         elif mav_features_selection == 'all':
#             print("Gather all feature vectors for each class")
#         elif mav_features_selection == 'none_correct_then_all':
#             print("Gather correctly predicted feature vectors for each class. If none, then use all examples")
#         model.eval()
#         if mav_features_selection in ['correct', 'all']:
#             features_dict = {seen_class_softmax_index: [] for seen_class_softmax_index in seen_class_softmax_indices}
#         elif mav_features_selection == 'none_correct_then_all':
#             features_dict = {seen_class_softmax_index: {'correct':[],'all':[]} for seen_class_softmax_index in seen_class_softmax_indices}

#         if self.config.verbose:
#             pbar = tqdm(train_loader, ncols=80)
#         else:
#             pbar = train_loader

#         # cur_features = []
#         # if self.use_feature_before_fc:
#         #     def forward_hook_func(self, inputs, outputs):
#         #         cur_features.append(inputs[0])
#         #     handle = self.model.fc.register_forward_hook(forward_hook_func)

#         for batch, data in enumerate(pbar):
#             inputs, real_labels = data
            
#             inputs = inputs.to(self.device)
#             labels = target_mapping_func(real_labels.to(self.device))

#             with torch.set_grad_enabled(False):
#                 outputs = model(inputs)

#                 cur_features = []
#                 _, preds = torch.max(outputs, 1)
#                 correct_indices = preds == labels.data

#                 for i in range(inputs.size(0)):
#                     if mav_features_selection == 'correct':
#                         if not correct_indices[i]:
#                             continue
#                         else:
#                             features_dict[int(labels[i])].append(outputs[i].unsqueeze(0))
#                     elif mav_features_selection == 'all':
#                         features_dict[int(labels[i])].append(outputs[i].unsqueeze(0))
#                     elif mav_features_selection == 'none_correct_then_all':
#                         if correct_indices[i]:
#                             features_dict[int(labels[i])]['correct'].append(outputs[i].unsqueeze(0))

#                         features_dict[int(labels[i])]['all'].append(outputs[i].unsqueeze(0))

#         if mav_features_selection == 'none_correct_then_all':
#             none_correct_classes = [] # All None correct classes
#             new_features_dict = {seen_class_softmax_index: None for seen_class_softmax_index in seen_class_softmax_indices}
#             for class_index in seen_class_softmax_indices:
#                 if len(features_dict[class_index]['correct']) == 0:
#                     none_correct_classes.append(class_index)
#                     if len(features_dict[class_index]['all']) == 0:
#                         import pdb; pdb.set_trace()  # breakpoint 98f3a0ff //
#                         print(f"No training example for {class_index}?")
#                     new_features_dict[class_index] = features_dict[class_index]['all']
#                 else:
#                     new_features_dict[class_index] = features_dict[class_index]['correct']
#             print("These classes has no correct feature. So we use all inputs.")
#             print(none_correct_classes)
#             features_dict = new_features_dict

#         return features_dict

#     def _gather_weibull_distribution(self, training_features, distance_metric='eucos', weibull_tail_size=20):
#         assert distance_metric in ['eucos', 'eu', 'cos']
#         weibull = {seen_class_index : {'mav': None, 'eu_distances': None, 'cos_distances': None, 'eucos_distances': None, 'weibull_model': None} 
#                    for seen_class_index in training_features.keys()}
#         for index in training_features.keys():
#             if not len(training_features[index]) > 0:
#                 print(f"Error: No training examples for category {index}")
#                 import pdb; pdb.set_trace()  # breakpoint 18e1e416 //
#             else:
#                 features_tensor = torch.cat(training_features[index], dim=0)
#                 mav = torch.mean(features_tensor, 0)
#                 mav_matrix = mav.unsqueeze(0).expand(features_tensor.size(0), -1)
#                 eu_distances = torch.sqrt(torch.sum((mav_matrix - features_tensor) ** 2, dim=1)) / self.div_eu # EU distance divided by 200.
#                 cos_distances = 1 - torch.nn.CosineSimilarity(dim=1)(mav_matrix, features_tensor)
#                 eucos_distances = eu_distances + cos_distances

#                 weibull[index]['mav'] = mav
#                 weibull[index]['eu_distances'] = eu_distances
#                 weibull[index]['cos_distances'] = cos_distances
#                 weibull[index]['eucos_distances'] = eucos_distances

#                 if distance_metric == 'eu':
#                     distance_scores = list(eu_distances)
#                 elif distance_metric == 'eucos':
#                     distance_scores = list(eucos_distances)
#                 elif distance_metric == 'cos':
#                     distance_scores = list(cos_distances)
#                 mr = libmr.MR()
#                 tailtofit = sorted(distance_scores)[-weibull_tail_size:]
#                 mr.fit_high(tailtofit, len(tailtofit))
#                 weibull[index]['weibull_model'] = mr
#         return weibull

#     def compute_open_max(self, outputs, log_thresholds=True):
#         """ Return (openset_score, openset_preds)
#         """
#         alpha_weights = [((self.alpha_rank+1) - i)/float(self.alpha_rank) for i in range(1, self.alpha_rank+1)]
#         softmax_outputs = F.softmax(outputs, dim=1)
#         _, softmax_rank = torch.sort(softmax_outputs, descending=True)
        
#         open_scores = torch.zeros(outputs.size(0)).to(outputs.device)
#         for batch_i in range(outputs.size(0)):
#             # if self.use_feature_before_fc:
#             #     batch_i_output = features[batch_i]
#             # else:
#             batch_i_output = outputs[batch_i]
#             batch_i_rank = softmax_rank[batch_i]
            
#             for i in range(len(alpha_weights)):
#                 class_index = int(batch_i_rank[i])
#                 alpha_i = alpha_weights[i]
#                 distance = self.distance_func(self.weibull_distributions[class_index]['mav'], batch_i_output)
#                 wscore = self.weibull_distributions[class_index]['weibull_model'].w_score(distance)
#                 modified_score = batch_i_output[class_index] * (1 - alpha_i*wscore)
#                 open_scores[batch_i] += batch_i_output[class_index] * alpha_i*wscore
#                 batch_i_output[class_index] = modified_score # should change the corresponding score in outputs

#         # total_denominators = torch.sum(torch.exp(outputs), dim=1) + torch.exp(open_scores)
#         openmax_outputs = F.softmax(torch.cat((outputs, open_scores.unsqueeze(1)), dim=1), dim=1)
#         openmax_max, openmax_preds = torch.max(openmax_outputs, 1)
#         openmax_top2, openmax_preds2 = torch.topk(openmax_outputs, 2, dim=1)
#         closed_preds = torch.where(openmax_preds == self.num_discovered_classes, 
#                                    openmax_preds2[:, 1],
#                                    openmax_preds)
#         closed_maxs = torch.where(openmax_preds == self.num_discovered_classes, 
#                                   openmax_top2[:, 1],
#                                   openmax_max)
#         openmax_predicted_label = torch.where(openmax_preds == self.num_discovered_classes, 
#                                               torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device),
#                                               openmax_preds)
#         if log_thresholds:
#             # First update the open set stats
#             self._update_open_set_stats(openmax_max, openmax_preds)

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (openmax_outputs[:, self.num_discovered_classes]).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += closed_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += closed_maxs.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += openmax_predicted_label.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += openmax_max.tolist()

#         # Return the prediction
#         preds = torch.where((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_discovered_classes), 
#                             torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device),
#                             openmax_preds)
#         return openmax_outputs, preds

#     def get_open_score_func(self):
#         def open_score_func(inputs):
#             outputs = self.model(inputs)
#             scores, _ = self.compute_open_max(outputs, log_thresholds=False)
#             return scores[:, -1]
#         return open_score_func

#     def _get_open_set_pred_func(self):
#         """ Caveat: Open set class is represented as -1.
#         """
#         return lambda outputs, inputs : self.compute_open_max(outputs)[1]

#     def _get_open_set_score_func(self):
#         """ Return the Openmax score. Say seen class has length 100, then return a tensor of length 101, where index 101 is the open set score
#         """
#         return lambda outputs, inputs : self.compute_open_max(outputs)[0]

#     def _eval(self, model, test_dataset, discovered_classes, verbose=False, training_features=None):
#         assert hasattr(self, 'dataloaders') # Should be updated after calling super._train()
#         if training_features == None:
#             training_features = self._gather_correct_features(model,
#                                                               self.dataloaders['train'], # from super._train()
#                                                               discovered_classes=discovered_classes,
#                                                               mav_features_selection=self.mav_features_selection) # A dict of (key: seen_class_indice, value: A list of feature vector that has correct prediction in this class)

#         self.weibull_distributions = self._gather_weibull_distribution(training_features,
#                                                                        distance_metric=self.distance_metric,
#                                                                        weibull_tail_size=self.weibull_tail_size) # A dict of (key: seen_class_indice, value: A per channel, MAV + weibull model)

#         assert len(self.weibull_distributions.keys()) == len(discovered_classes)
#         self.num_discovered_classes = len(discovered_classes)
#         self._reset_open_set_stats() # Open set status is the summary of 1/ Number of threshold reject 2/ Number of Open Class reject
#         eval_result = super(OSDNNetwork, self)._eval(model, test_dataset, discovered_classes, verbose=verbose)
#         if verbose:
#             print(f"Rejection details: Total rejects {self.open_set_stats['total_reject']}. "
#                   f"By threshold ({self.osdn_eval_threshold}) {self.open_set_stats['threshold_reject']}. "
#                   f"By being open class {self.open_set_stats['open_class_reject']}. "
#                   f"By both {self.open_set_stats['both_reject']}. ")
#         return eval_result

#     def _update_open_set_stats(self, openmax_max, openmax_preds):
#         # For each batch
#         self.open_set_stats['threshold_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & ~(openmax_preds == self.num_discovered_classes) ))
#         self.open_set_stats['open_class_reject'] += float(torch.sum(~(openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_discovered_classes) ))
#         self.open_set_stats['both_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_discovered_classes) ))
#         self.open_set_stats['total_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_discovered_classes) ))
#         assert self.open_set_stats['threshold_reject'] + self.open_set_stats['open_class_reject'] + self.open_set_stats['both_reject'] == self.open_set_stats['total_reject']

#     def _reset_open_set_stats(self):
#         # threshold_reject and open_class_reject are mutually exclusive
#         self.open_set_stats = {'threshold_reject': 0., 
#                                'open_class_reject': 0.,
#                                'both_reject': 0.,
#                                'total_reject': 0.}


# class OSDNNetworkModified(OSDNNetwork):
#     def __init__(self, *args, **kwargs):
#         # This is essentially using the open max algorithm. But this modified version 
#         # doesn't change the activation score of the seen classes.
#         super(OSDNNetworkModified, self).__init__(*args, **kwargs)

#     def compute_open_max(self, outputs, log_thresholds=True):
#         """ Return (openset_score, openset_preds)
#         """
#         alpha_weights = [((self.alpha_rank+1) - i)/float(self.alpha_rank) for i in range(1, self.alpha_rank+1)]
#         softmax_outputs = F.softmax(outputs, dim=1)
#         _, softmax_rank = torch.sort(softmax_outputs, descending=True)
        
#         open_scores = torch.zeros(outputs.size(0)).to(outputs.device)
#         for batch_i in range(outputs.size(0)):
#             batch_i_output = outputs[batch_i]
#             batch_i_rank = softmax_rank[batch_i]
            
#             for i in range(len(alpha_weights)):
#                 class_index = int(batch_i_rank[i])
#                 alpha_i = alpha_weights[i]
#                 distance = self.distance_func(self.weibull_distributions[class_index]['mav'], batch_i_output)
#                 wscore = self.weibull_distributions[class_index]['weibull_model'].w_score(distance)
#                 open_scores[batch_i] += batch_i_output[class_index] * alpha_i*wscore

#         # total_denominators = torch.sum(torch.exp(outputs), dim=1) + torch.exp(open_scores)
#         openmax_outputs = F.softmax(torch.cat((outputs, open_scores.unsqueeze(1)), dim=1), dim=1)
#         openmax_max, openmax_preds = torch.max(openmax_outputs, 1)
#         openmax_top2, openmax_preds2 = torch.topk(openmax_outputs, 2, dim=1)

#         closed_preds = torch.where(openmax_preds == self.num_discovered_classes, 
#                                    openmax_preds2[:, 1],
#                                    openmax_preds)
#         closed_maxs = torch.where(openmax_preds == self.num_discovered_classes, 
#                                   openmax_top2[:, 1],
#                                   openmax_max)
#         openmax_predicted_label = torch.where(openmax_preds == self.num_discovered_classes, 
#                                               torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device),
#                                               openmax_preds)
        
#         if log_thresholds:
#             # First update the open set stats
#             self._update_open_set_stats(openmax_max, openmax_preds)

#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
#             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
#             self.thresholds_checkpoints[self.round]['open_set_score'] += (openmax_outputs[:, self.num_discovered_classes]).tolist()
#             self.thresholds_checkpoints[self.round]['closed_predicted'] += closed_preds.tolist()
#             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += closed_maxs.tolist()
#             self.thresholds_checkpoints[self.round]['open_predicted'] += openmax_predicted_label.tolist()
#             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += openmax_max.tolist()

        
#         preds = torch.where((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_discovered_classes), 
#                             torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device),
#                             openmax_preds)
#         return openmax_outputs, preds

def get_acc_from_performance_dict(dict):
    result_dict = {}
    for k in dict.keys():
        if dict[k]['count'] == 0:
            result_dict[k] = "N/A"
        else:
            result_dict[k] = float(dict[k]['corrects']/dict[k]['count'])
    return result_dict
