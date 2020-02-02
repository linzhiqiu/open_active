import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from collections import OrderedDict
from tqdm import tqdm
import copy

import models
from instance_info import BasicInfoCollector, ClusterInfoCollector, SigmoidInfoCollector
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list
from distance import eu_distance, cos_distance, eu_distance_batch, cos_distance_batch

from global_setting import OPEN_CLASS_INDEX, UNSEEN_CLASS_INDEX, PRETRAINED_MODEL_PATH

import libmr
import math

class NoSeenClassException(Exception):
    def __init__(self, message):
        self.message = message

class NoUnseenClassException(Exception):
    def __init__(self, message):
        self.message = message

def get_dynamic_threshold(log : list, metric='softmax', mode='weighted'):
    assert len(log) > 0
    assert metric in ['softmax', 'entropy', 'gaussian']
    assert mode in ['weighted', 'average', '7_3']
    assert hasattr(log[0], metric)
    log_copy = log.copy() # To avoid sorting the original list
    log_copy.sort(key= lambda x: getattr(x, metric))

    num_seen = 0
    num_unseen = 0
    for instance in log_copy:
        if instance.seen == 1:
            num_seen += 1
        else:
            num_unseen += 1

    if num_seen == 0:
        raise NoSeenClassException("No new instances from seen class")
    if num_unseen == 0:
        raise NoUnseenClassException("No new instances from unseen class")

    if mode == 'weighted': 
        candidates = [(getattr(log_copy[0], metric), num_unseen)] # A list of tuple of potential candidates. Each tuple is (threshold, number of errors)
        for index, instance in enumerate(log_copy[1:]):
            # index is the index of the last item
            if log_copy[index].seen == 1:
                new_error = candidates[-1][1] + 1
            else:
                new_error = candidates[-1][1] - 1
            candidates.append((getattr(instance, metric), new_error))

        candidates.sort(key=lambda x: x[1])
        print(f"Find threshold {candidates[0][0]} with total error {candidates[0][1]}/{num_seen + num_unseen}")
    elif mode in ['average', '7_3']:
        num_unseen_as_seen = float(num_unseen)
        num_seen_as_unseen = 0.0
        candidates = [(getattr(log_copy[0], metric), {'closed' : 0.0, 'open' : 1.0})] # A list of tuple of potential candidates. Each tuple is (threshold, dict_of_error_rate)
        for index, instance in enumerate(log_copy[1:]):
            # index is the index of the last item
            if log_copy[index].seen == 1:
                num_seen_as_unseen += 1
            else:
                num_unseen_as_seen -= 1
            closed_err = num_seen_as_unseen / num_seen
            open_err = num_unseen_as_seen / num_unseen
            candidates.append((getattr(instance, metric), {'closed' : closed_err, 'open' : open_err}))
        if mode == 'average':
            candidates.sort(key=lambda x: x[1]['closed'] + x[1]['open'])
        elif mode == '7_3':
            candidates.sort(key=lambda x: 0.7 * x[1]['closed'] + 0.3 * x[1]['open'])
        print(f"Find threshold {candidates[0][0]} with closed set acc {1.-candidates[0][1]['closed']} and open set acc {1.-candidates[0][1]['open']}")
    return candidates[0][0]


def train_epochs(model, dataloaders, optimizer, scheduler, criterion, target_mapping_func, device='cuda', start_epoch=0, max_epochs=-1, verbose=True):
    """Regular PyTorch training procedure: Train model using data in dataloaders['train'] from start_epoch to max_epochs-1
    """
    assert start_epoch <= max_epochs
    avg_loss = 0.
    avg_acc = 0.

    for epoch in range(start_epoch, max_epochs):
        # print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        # print('-' * 10)
        for phase in dataloaders.keys():
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.
            count = 0

            if verbose:
                pbar = tqdm(dataloaders[phase], ncols=80)
            else:
                pbar = dataloaders[phase]

            for batch, data in enumerate(pbar):
                inputs, real_labels = data
                labels = target_mapping_func(real_labels)
                count += inputs.size(0)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    if verbose:
                        pbar.set_postfix(loss=running_loss/count, 
                                         acc=float(running_corrects)/count,
                                         epoch=epoch)

                avg_loss = running_loss/count
                avg_acc = float(running_corrects)/count
                # print(f"Epoch {epoch} => "
                #       f"Loss {avg_loss}, Accuracy {avg_acc}")
            # print()

    return avg_loss, avg_acc

class TrainerMachine(object):
    """Abstract class"""
    def __init__(self, config, train_instance):
        super(TrainerMachine, self).__init__()
        self.config = config
        self.train_instance = train_instance
        self.log = None # This should be maintained by the corresponding label picker. It will be updated after each call to label_picker.select_new_data() 
        self.exemplar_set = None # Only works for ICALR based modules
        # For ROC/other open set evaluation metric. self.thresholds_checkpoints[epoch] is a dictionary of all the information for a given epoch
        # This metric should be updated in the open_set_prediction function returned by self._get_open_set_pred()
        self.thresholds_checkpoints = {}
        self.handle = None # For any handle object

    def get_thresholds_checkpoints(self):
        return self.thresholds_checkpoints

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def train_then_eval(self, s_train, seen_classes, test_dataset, eval_verbose=True):
        raise NotImplementedError()

    def remove_handle(self):
        # Currently only used in openCollector
        if type(self.handle) != type(None):
            self.handle.remove()

    # Below are some helper functions shared by all subclasses
    def _get_target_mapp_func(self, seen_classes):
        return get_target_mapping_func_for_tensor(self.train_instance.classes,
                                                  seen_classes,
                                                  self.train_instance.open_classes)

    def _get_target_unmapping_func_for_list(self, seen_classes):
        return get_target_unmapping_func_for_list(self.train_instance.classes,
                                                  seen_classes)

class Network(TrainerMachine):
    def __init__(self, *args, **kwargs):
        super(Network, self).__init__(*args, **kwargs)
        self.epoch = 0
        self.round = 0
        self.max_epochs = self.config.epochs
        self.device = self.config.device
        self.model = self._get_network_model()
        self.info_collector_class = BasicInfoCollector
        self.criterion_class = nn.CrossEntropyLoss
        # self.optimizer = self._get_network_optimizer(self.model)
        # self.scheduler = self._get_network_scheduler(self.optimizer)

        # Current training state. Update in train_new_round(). Used in other functions.
        self.seen_classes = set()

        if self.config.pseudo_open_set != None:
            if self.config.pseudo_same_network:
                print("Warning using same network for pseudo open training.")
                self.pseudo_open_model = self.model
            else:
                self.pseudo_open_model = self._get_network_model()
            # self.pseudo_open_optimizer = self._get_network_optimizer(self.pseudo_open_model)
            # self.pseudo_open_scheduler = self._get_network_scheduler(self.pseudo_open_optimizer)
            self.pseudo_open_set_metric = self.config.pseudo_open_set_metric # weighted or average
            self.pseudo_open_set = self.config.pseudo_open_set
            self.pseudo_open_set_rounds = self.config.pseudo_open_set_rounds
            print(f"Using the first {self.pseudo_open_set} as pseudo classes for {self.pseudo_open_set_rounds} rounds.")
            self.pseudo_open_set_classes = set([i for i in range(self.pseudo_open_set)])
        else:
            self.pseudo_open_set = self.config.pseudo_open_set
            self.pseudo_open_set_rounds = 0
            self.pseudo_open_set_classes = set() # No pseudo open set classes

    def get_checkpoint(self):
        return {
            'round' : self.round,
            'epoch' : self.epoch,
            'arch'  : self.config.arch,
            'model' : self.model.state_dict(),
            # 'optimizer' : self.optimizer.state_dict(),
            # 'scheduler' : self.scheduler.state_dict()
        }

    def load_checkpoint(self, checkpoint):
        self.epoch = checkpoint['epoch']
        new_model_checkpoint = copy.deepcopy(checkpoint['model'])
        for layer_name in checkpoint['model']:
            if "fc.weight" in layer_name or "fc.bias" in layer_name:
                del new_model_checkpoint[layer_name]
        self.model.load_state_dict(new_model_checkpoint, strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])

    def _get_criterion(self, dataloader, target_mapping_func, seen_classes=set(), criterion_class=nn.CrossEntropyLoss):
        assert seen_classes.__len__() > 0
        assert self.config.class_weight in ['uniform', 'imbal']
        if self.config.class_weight == 'uniform':
            weight = None
            print('Using uniform class weight.')
        elif self.config.class_weight == 'imbal':
            weight = torch.zeros(len(seen_classes)).to(self.device)
            total = 0.0
            for _, data in enumerate(tqdm(dataloader, ncols=80)):
                _, real_labels = data
                labels = target_mapping_func(real_labels)
                for label_i in labels:
                    weight[label_i] += 1.
                    total += 1.
            weight = total / weight
            weight = weight / weight.min() # TODO: Figure out whether or not need this min()
            weight[weight > 10.] = 10.
            class_weight_info = {}
            unmap_dict = get_target_unmapping_dict(self.train_instance.classes, seen_classes)
            for i, w_i in enumerate(weight):
                class_weight_info[unmap_dict[i]] = float(w_i)
            print(f'Using class weight: {class_weight_info}')
        return criterion_class(weight=weight)

    def _filter_pseudo_open_set(self, samples : list, all_seen_class : set):
        assert len(self.pseudo_open_set_classes) > 0
        for pseudo_open_class in self.pseudo_open_set_classes:
            assert pseudo_open_class in all_seen_class

        # Remove pseudo open class examples from training
        remaining_seen_classes = all_seen_class.difference(self.pseudo_open_set_classes)
        # pseudo_open_samples = list(filter(lambda x : self.train_instance.train_labels[x] in self.pseudo_open_set_classes, samples))
        remaining_samples = list(filter(lambda x : self.train_instance.train_labels[x] in remaining_seen_classes, samples))
        return remaining_samples, remaining_seen_classes

    def _train(self, model, s_train, seen_classes, start_epoch=0):
        self._train_mode()
        target_mapping_func = self._get_target_mapp_func(seen_classes)
        self.dataloaders = get_subset_dataloaders(self.train_instance.train_dataset,
                                                  list(s_train),
                                                  [], # TODO: Make a validation set
                                                  None, # No target transform is made
                                                  batch_size=self.config.batch,
                                                  workers=self.config.workers)
        
        self._update_last_layer(model, len(seen_classes), device=self.device)
        optimizer = self._get_network_optimizer(model)
        scheduler = self._get_network_scheduler(optimizer)

        self.criterion = self._get_criterion(self.dataloaders['train'],
                                             target_mapping_func,
                                             seen_classes=seen_classes,
                                             criterion_class=self.criterion_class)

        with SetPrintMode(hidden=not self.config.verbose):
            train_loss, train_acc = train_epochs(
                                        model,
                                        self.dataloaders,
                                        optimizer,
                                        scheduler,
                                        self.criterion,
                                        target_mapping_func,
                                        device=self.device,
                                        start_epoch=start_epoch,
                                        max_epochs=self.max_epochs,
                                        verbose=self.config.verbose,
                                    )
        print(f"Train => {self.round} round => "
              f"Loss {train_loss}, Accuracy {train_acc}")
        return train_loss, train_acc

    def _meta_learning(self, pseudo_open_model, full_train_set, pseudo_seen_classes):
        assert self.round <= self.pseudo_open_set_rounds
        unmap_dict = get_target_unmapping_dict(self.train_instance.classes, pseudo_seen_classes)
        info_collector = self.info_collector_class(self.round, unmap_dict, pseudo_seen_classes)
        dataloader = get_subset_loader(self.train_instance.train_dataset,
                                       list(full_train_set),
                                       None, # No target transform is needed! 
                                       shuffle=False,
                                       batch_size=self.config.batch,
                                       workers=self.config.workers)
        _, info = info_collector.gather_instance_info(dataloader, pseudo_open_model, device=self.device)
    
        assert len(info) > 0
        self.pseuopen_threshold = get_dynamic_threshold(info, metric=self.config.threshold_metric, mode=self.pseudo_open_set_metric)
        print(f"Update pseudo open set threshold to {self.pseuopen_threshold}")

    def train_then_eval(self, s_train, seen_classes, test_dataset, eval_verbose=True, start_epoch=0):
        self.round += 1
        if self.round <= self.pseudo_open_set_rounds:
            # self.curr_full_train_set = s_train.copy() # Save for self._get_open_pred_func()
            # full_train_set = s_train.copy()
            # full_seen_classes = seen_classes.copy()
            pseudo_s_train, pseudo_seen_classes = self._filter_pseudo_open_set(s_train, seen_classes)
            self._train(self.pseudo_open_model, pseudo_s_train, pseudo_seen_classes, start_epoch=0)
            self._meta_learning(self.pseudo_open_model, s_train, pseudo_seen_classes)
        
        train_loss, train_acc = self._train(self.model, s_train, seen_classes, start_epoch=0)  
        eval_results = self._eval(self.model, test_dataset, seen_classes, verbose=eval_verbose)
        return train_loss, train_acc, eval_results

    def _eval(self, model, test_dataset, seen_classes, verbose=True):
        self._eval_mode()

        # Update self.thresholds_checkpoints
        # assert self.round not in self.thresholds_checkpoints.keys()
        # Caveat: Currently, the open_set_score is updated in the open_set_prediction function. Yet the grouth_truth is updated in self._eval()
        self.thresholds_checkpoints[self.round] = {'ground_truth' : [], # 0 if closed set, UNSEEN_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
                                                   'real_labels' : [], # The real labels for CIFAR100 or other datasets.
                                                   'open_set_score' : [], # Higher the score, more likely to be open set
                                                   'closed_predicted' : [], # If fail the open set detection, then what's the predicted closed set label (network output)?
                                                   'closed_predicted_real' : [], # If fail the open set detection, then what's the predicted closed set label (real labels)?
                                                   'closed_argmax_prob' : [], # If fail the open set detection, then what's the probability for predicted closed set class (real labels)?
                                                   'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
                                                   'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
                                                   'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
                                                  } # A list of dictionary

        target_mapping_func = self._get_target_mapp_func(seen_classes)
        target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(seen_classes) # Only for transforming predicted label (in network indices) to real indices
        dataloader = get_loader(test_dataset,
                                # self._get_target_mapp_func(seen_classes),
                                None,
                                shuffle=False,
                                batch_size=self.config.batch,
                                workers=self.config.workers)
        
        open_set_prediction = self._get_open_set_pred_func()

        with SetPrintMode(hidden=not verbose):
            if verbose:
                pbar = tqdm(dataloader, ncols=80)
            else:
                pbar = dataloader

            performance_dict = {'train_class_acc' : {'corrects' : 0., 'not_seen' : 0., 'count' : 0.}, # Accuracy of all non-hold out open class examples. If some classes not seen yet, accuracy = 0
                                'unseen_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all unseen open class examples.
                                'overall_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all examples. Counting accuracy of unseen open examples.
                                'holdout_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of hold out open class examples
                                'seen_closed_acc' : {'open' : 0., 'corrects' : 0., 'count' : 0.}, # Accuracy of seen class examples
                                'all_open_acc' : {'corrects' : 0., 'count' : 0.}, # Accuracy of all open class examples (unseen open + hold-out open)
                                }

            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, real_labels = data
                    
                    inputs = inputs.to(self.device)
                    labels = target_mapping_func(real_labels.to(self.device))
                    labels_for_openset_pred = torch.where(
                                                  labels == OPEN_CLASS_INDEX,
                                                  torch.LongTensor([UNSEEN_CLASS_INDEX]).to(labels.device),
                                                  labels
                                              ) # This change hold out open set examples' indices to unseen open set examples indices

                    outputs = model(inputs)
                    preds = open_set_prediction(outputs, inputs=inputs) # Open set index == UNSEEN_CLASS_INDEX

                    self.thresholds_checkpoints[self.round]['ground_truth'] += labels.tolist()
                    self.thresholds_checkpoints[self.round]['real_labels'] += real_labels.tolist()
                    # statistics
                    # running_loss += loss.item() * inputs.size(0)
                    performance_dict['overall_acc']['count'] += inputs.size(0)
                    performance_dict['overall_acc']['corrects'] += float(torch.sum(preds == labels_for_openset_pred.data))
                    
                    unseen_open_indices = labels == UNSEEN_CLASS_INDEX
                    seen_closed_indices = labels >= 0
                    hold_out_open_indices = labels == OPEN_CLASS_INDEX
                    train_class_indices = unseen_open_indices | seen_closed_indices
                    all_open_indices = unseen_open_indices | hold_out_open_indices
                    assert torch.sum(unseen_open_indices & seen_closed_indices & hold_out_open_indices) == 0
                    
                    performance_dict['train_class_acc']['count'] += float(torch.sum(train_class_indices))
                    performance_dict['unseen_open_acc']['count'] += float(torch.sum(unseen_open_indices))
                    performance_dict['holdout_open_acc']['count'] += float(torch.sum(hold_out_open_indices))
                    performance_dict['seen_closed_acc']['count'] += float(torch.sum(seen_closed_indices))
                    performance_dict['all_open_acc']['count'] += float(torch.sum(all_open_indices))

                    performance_dict['train_class_acc']['not_seen'] += torch.sum(
                                                                           unseen_open_indices
                                                                       ).float()
                    performance_dict['train_class_acc']['corrects'] += torch.sum(
                                                                           torch.masked_select(
                                                                               (preds==labels.data),
                                                                               seen_closed_indices
                                                                           )
                                                                       ).float()
                    performance_dict['unseen_open_acc']['corrects'] += torch.sum(
                                                                           torch.masked_select(
                                                                               (preds==labels.data),
                                                                               unseen_open_indices
                                                                           )
                                                                       ).float()
                    performance_dict['holdout_open_acc']['corrects'] += torch.sum(
                                                                            torch.masked_select(
                                                                                (preds==labels_for_openset_pred.data),
                                                                                hold_out_open_indices
                                                                            )
                                                                        ).float()
                    performance_dict['seen_closed_acc']['corrects'] += torch.sum(
                                                                           torch.masked_select(
                                                                               (preds==labels.data),
                                                                               seen_closed_indices
                                                                           )
                                                                       ).float()
                    performance_dict['seen_closed_acc']['open'] += torch.sum(
                                                                       torch.masked_select(
                                                                           (preds==UNSEEN_CLASS_INDEX),
                                                                           seen_closed_indices
                                                                       )
                                                                   ).float()
                    performance_dict['all_open_acc']['corrects'] += torch.sum(
                                                                        torch.masked_select(
                                                                            (preds==labels_for_openset_pred.data),
                                                                            all_open_indices
                                                                        )
                                                                    ).float()

                    batch_result = get_acc_from_performance_dict(performance_dict)
                    if verbose:
                        pbar.set_postfix(batch_result)

                epoch_result = get_acc_from_performance_dict(performance_dict)
                train_class_acc = epoch_result['train_class_acc']
                unseen_open_acc = epoch_result['unseen_open_acc']
                overall_acc = epoch_result['overall_acc']
                holdout_open_acc = epoch_result['holdout_open_acc']
                seen_closed_acc = epoch_result['seen_closed_acc']
                all_open_acc = epoch_result['all_open_acc']

                overall_count = performance_dict['overall_acc']['count']
                train_class_count = performance_dict['train_class_acc']['count']
                unseen_open_count = performance_dict['unseen_open_acc']['count']
                holdout_open_count = performance_dict['holdout_open_acc']['count']
                seen_closed_count = performance_dict['seen_closed_acc']['count']
                all_open_count = performance_dict['all_open_acc']['count']

                train_class_corrects = performance_dict['train_class_acc']['corrects']
                unseen_open_corrects = performance_dict['unseen_open_acc']['corrects']
                seen_closed_corrects = performance_dict['seen_closed_acc']['corrects']

                train_class_notseen = performance_dict['train_class_acc']['not_seen']
                seen_closed_open = performance_dict['seen_closed_acc']['open']

            self.thresholds_checkpoints[self.round]['closed_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['closed_predicted'])
            self.thresholds_checkpoints[self.round]['open_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['open_predicted'])
            print(f"Test => "
                  f"Training Class Acc {train_class_acc}, "
                  f"Hold-out Open-Set Acc {holdout_open_acc}")
            print(f"Details => "
                  f"Overall Acc {overall_acc}, "
                  f"Overall Open-Set Acc {all_open_acc}, Overall Seen-Class Acc {seen_closed_acc}")
            print(f"Training Classes Accuracy Details => "
                  f"[{train_class_notseen}/{train_class_count}] not in seen classes, "
                  f"and for seen class samples [{seen_closed_corrects}/{seen_closed_count} corrects | [{seen_closed_open}/{seen_closed_count}] wrongly as open set]")
        return epoch_result


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
                    threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
                except NoSeenClassException:
                    # Error when no new instances from seen class
                    threshold = self.config.network_eval_threshold
                    print(f"No seen class instances. Threshold set to {threshold}")
                except NoUnseenClassException:
                    threshold = self.config.network_eval_threshold
                    print(f"No unseen class instances. Threshold set to {threshold}")
                else:
                    print(f"Threshold set to {threshold} based on all existing instances.")
        elif self.config.network_eval_mode in ['pseuopen_threshold']:
            assert hasattr(self, 'pseuopen_threshold')
            print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
            threshold = self.pseuopen_threshold

        def open_set_prediction(outputs, inputs=None):
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            if self.config.threshold_metric == 'softmax':
                scores = softmax_max
            elif self.config.threshold_metric == 'entropy':
                neg_entropy = softmax_outputs*softmax_outputs.log()
                neg_entropy[softmax_outputs < 1e-5] = 0
                scores = neg_entropy.sum(dim=1) # negative entropy!

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

    def get_open_score_func(self):
        def open_score_func(inputs):
            outputs = self.model(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            if self.config.threshold_metric == 'softmax':
                scores = softmax_max
            elif self.config.threshold_metric == 'entropy':
                neg_entropy = softmax_outputs*softmax_outputs.log()
                neg_entropy[softmax_outputs < 1e-5] = 0
                scores = neg_entropy.sum(dim=1) # negative entropy!
            return -scores
        return open_score_func

    def _train_mode(self):
        self.model.train()

    def _eval_mode(self):
        self.model.eval()

    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)(last_relu=False)
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
        
    def _get_network_optimizer(self, model):
        """ Get softmax network optimizer
        """
        if self.config.optim == 'sgd':
            optim_module = optim.SGD
            optim_param = {"lr" : self.config.lr, 
                           "momentum" : self.config.momentum,
                           "weight_decay" : 0 if self.config.wd == None else float(self.config.wd)}
        elif self.config.optim == "adam":
            optim_module = optim.Adam
            optim_param = {"lr": self.config.lr, 
                           "weight_decay": float(self.config.wd), 
                           "amsgrad": self.config.amsgrad}
        else:
            raise ValueError("optim type not supported")
            
        optimizer = optim_module(
                        filter(lambda x : x.requires_grad, model.parameters()), 
                        **optim_param
                    )
        return optimizer

    def _get_network_scheduler(self, optimizer):
        """ Get softmax network optimizer
        """
        if self.config.lr_decay_step == None:
            decay_step = self.max_epochs
        else:
            decay_step = self.config.lr_decay_step
        scheduler = lr_scheduler.StepLR(
                        optimizer, 
                        step_size=decay_step, 
                        gamma=self.config.lr_decay_ratio
                    )
        return scheduler

    def _update_last_layer(self, model, output_size, device='cuda'):
        if "resnet" in self.config.arch.lower():
            fd = int(model.fc.weight.size()[1])
            model.fc = nn.Linear(fd, output_size)
            model.fc.weight.data.normal_(0, 0.01)
            model.fc.bias.data.zero_()
            model.fc.to(device)
        elif self.config.arch in ['classifier32', 'classifier32_instancenorm']:
            fd = int(model.fc1.weight.size()[1])
            model.fc1 = nn.Linear(fd, output_size)
            model.fc1.weight.data.normal_(0, 0.01)
            model.fc1.bias.data.zero_()
            model.fc1.to(device)
        else:
            raise NotImplementedError()

class ClusterNetwork(Network):
    def __init__(self, *args, **kwargs):
        # The output of self.model is raw 1-D feature vector x. cluster_predict(self.model, x) gives the prob score after normalization
        super(ClusterNetwork, self).__init__(*args, **kwargs)
        # Before everything else, remove the relu from layer4

        self.network_output_size = self._get_network_output_size(self.model) # Before changing the architecture
                
        self.clustering = self.config.clustering
        if self.clustering == 'rbf_train':
            self.gamma = self.config.rbf_gamma

        if self.pseudo_open_set == None:
            self.cluster_eval_threshold = self.config.cluster_eval_threshold
        else:
            self.cluster_eval_threshold = None

        self.cluster_level = self.config.cluster_level

        self.distance_metric = self.config.distance_metric
        self.div_eu = self.config.div_eu
        if self.distance_metric == 'eu':
            self.distance_func = lambda a, b: eu_distance_batch(a,b,div_eu=self.div_eu)
        elif self.distance_metric == 'eucos':
            self.distance_func = lambda a, b: eu_distance_batch(a,b,div_eu=self.div_eu) + cos_distance_batch(a,b)
        elif self.distance_metric == 'cos':
            self.distance_func = cos_distance_batch
        else:
            raise NotImplementedError()

        self.info_collector_class = lambda *args, **kwargs: ClusterInfoCollector(self.gamma, *args, **kwargs)
        # self.criterion_class = lambda **dict: lambda x, y: nn.NLLLoss(weight=dict['weight'])(torch.log(x / x.sum(1, keepdim=True)), y)
        # self.criterion_class = lambda **dict: lambda x, y: nn.CrossEntropyLoss(weight=dict['weight'])(torch.nn.LogSoftmax(dim=1)(x), y)
        self.criterion_class = lambda **dict: lambda x, y: nn.CrossEntropyLoss(weight=dict['weight'])(x, y) # ? This version - yes

    def _get_network_model(self):
        """ Get a network model without last ReLU layer
        """
        model = getattr(models, self.config.arch)(last_relu=False)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)

    def _get_network_output_size(self, model):
        """Caveat: Can only be called before _update_last_layer()"""
        if "resnet" in self.config.arch.lower():
            return int(model.fc.weight.size()[1])
        else:
            raise NotImplementedError()

    def _update_last_layer(self, model, output_size, device='cuda'):
        assert hasattr(self, 'network_output_size')
        if self.cluster_level == 'after_fc':
            if "resnet" in self.config.arch.lower():
                fd = int(self.network_output_size)
                fc = nn.Linear(fd, output_size)
                fc.weight.data.normal_(0, 0.01)
                fc.bias.data.zero_()
                fc.to(device)
            else:
                raise NotImplementedError()
            cluster_layer = models.ClusterLayer(output_size, output_size, self.distance_func, gamma=self.gamma).to(device)
            model.fc = torch.nn.Sequential(fc, cluster_layer)
        else:
            cluster_layer = models.ClusterLayer(output_size, self.network_output_size, self.distance_func, gamma=self.gamma).to(device)
            model.fc = cluster_layer

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
        if self.config.network_eval_mode == 'threshold':
            assert self.config.threshold_metric == "softmax"
            threshold = self.cluster_eval_threshold
        elif self.config.network_eval_mode == 'dynamic_threshold':
            assert type(self.log) == list
            if len(self.log) == 0:
                # First round, use default threshold
                threshold = self.cluster_eval_threshold
                print(f"First round. Use default threshold {threshold}")
            else:
                try:
                    threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
                except NoSeenClassException:
                    # Error when no new instances from seen class
                    threshold = self.config.network_eval_threshold
                    print(f"No seen class instances. Threshold set to {threshold}")
                except NoUnseenClassException:
                    threshold = self.config.network_eval_threshold
                    print(f"No unseen class instances. Threshold set to {threshold}")
                else:
                    print(f"Threshold set to {threshold} based on all existing instances.")
        elif self.config.network_eval_mode in ['pseuopen_threshold']:
            assert hasattr(self, 'pseuopen_threshold')
            print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
            threshold = self.pseuopen_threshold

        def open_set_prediction(outputs, inputs=None):
            if self.config.threshold_metric == 'gaussian':
                maxs, preds = torch.max(outputs, 1)
                scores = (outputs.exp() / (math.pi * (1./self.gamma))**.5).mean(1)
            elif self.config.threshold_metric == 'softmax':
                # outputs = outputs / outputs.sum(1, keepdim=True)
                outputs = torch.nn.functional.Softmax()(outputs)
                maxs, preds = torch.max(outputs, 1)
                scores = maxs
            elif self.config.threshold_metric == 'entropy':
                # outputs = outputs / outputs.sum(1, keepdim=True)
                outputs = torch.nn.functional.Softmax()(outputs)
                maxs, preds = torch.max(outputs, 1)
                neg_entropy = outputs*outputs.log()
                neg_entropy[outputs < 1e-5] = 0
                scores = neg_entropy.sum(dim=1) # negative entropy!

            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
            self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
            self.thresholds_checkpoints[self.round]['closed_predicted'] += preds.tolist()
            self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += maxs.tolist()
            self.thresholds_checkpoints[self.round]['open_predicted'] += preds.tolist()
            self.thresholds_checkpoints[self.round]['open_argmax_prob'] += maxs.tolist()

            preds = torch.where(scores < threshold,
                                torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device), 
                                preds)
            return preds
        return open_set_prediction

    def get_open_score_func(self):
        raise NotImplementedError()
        # def open_score_func(inputs):
        #     outputs = model(inputs)
        #     softmax_outputs = F.softmax(outputs, dim=1)
        #     softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
        #     if self.config.threshold_metric == 'softmax':
        #         scores = softmax_max
        #     elif self.config.threshold_metric == 'entropy':
        #         scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
        #     return scores
        # return open_score_func

def sigmoid_loss(weight=None, mode='mean'):
    def loss_func(x, y):
        sigmoids = 1 - nn.Sigmoid()(x)
        sigmoids[torch.arange(x.shape[0]), y] = nn.Sigmoid()(x)[torch.arange(x.shape[0]), y]
        # log_sigmoids_expand = log_sigmoids.unsqueeze(1).expand(-1,log_sigmoids.shape[1],-1)
        sigmoids = torch.max(torch.Tensor([1e-10]).to(sigmoids.device), sigmoids)
        log_sigmoids = sigmoids.log()
        if weight != None:
            log_sigmoids = weights * log_sigmoids
        if mode == 'mean':
            res = log_sigmoids.mean(1).mean()
        elif mode == 'sum':
            res = log_sigmoids.sum(1).mean()
        else:
            raise NotImplementedError()
        return - res
    return loss_func

class SigmoidNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(SigmoidNetwork, self).__init__(*args, **kwargs)
        # Before everything else, remove the relu from layer4
        
        if self.pseudo_open_set == None:
            self.network_eval_threshold = self.config.network_eval_threshold
        else:
            self.network_eval_threshold = None

        self.info_collector_class = lambda *args, **kwargs: SigmoidInfoCollector(*args, **kwargs)

        self.sigmoid_train_mode = self.config.sigmoid_train_mode
        self.criterion_class = lambda **dict: lambda x, y: sigmoid_loss(weight=dict['weight'], mode=self.sigmoid_train_mode)(x, y)

    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)(last_relu=False)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
        if self.config.network_eval_mode == 'threshold':
            threshold = self.config.network_eval_threshold
        elif self.config.network_eval_mode == 'dynamic_threshold':
            assert type(self.log) == list
            if len(self.log) == 0:
                # First round, use default threshold
                threshold = self.config.network_eval_threshold
                print(f"First round. Use default threshold {threshold}")
            else:
                try:
                    threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
                except NoSeenClassException:
                    # Error when no new instances from seen class
                    threshold = self.config.network_eval_threshold
                    print(f"No seen class instances. Threshold set to {threshold}")
                except NoUnseenClassException:
                    threshold = self.config.network_eval_threshold
                    print(f"No unseen class instances. Threshold set to {threshold}")
                else:
                    print(f"Threshold set to {threshold} based on all existing instances.")
        elif self.config.network_eval_mode in ['pseuopen_threshold']:
            assert hasattr(self, 'pseuopen_threshold')
            print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
            threshold = self.pseuopen_threshold

        def open_set_prediction(outputs, inputs=None):
            sigmoid_outputs = nn.Sigmoid()(outputs)
            sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            # if self.config.threshold_metric == 'softmax':
            #     scores = softmax_max
            # elif self.config.threshold_metric == 'entropy':
            #     scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
            scores = sigmoid_max

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

    def get_open_score_func(self):
        def open_score_func(inputs):
            outputs = self.model(inputs)
            sigmoid_outputs = nn.Sigmoid()(outputs)
            sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
            scores = sigmoid_max
            return -scores
        return open_score_func

class BinarySoftmaxNetwork(Network):
    def __init__(self, *args, **kwargs):
        super(BinarySoftmaxNetwork, self).__init__(*args, **kwargs)
        # Before everything else, remove the relu from layer4
        
        if self.pseudo_open_set == None:
            self.network_eval_threshold = self.config.network_eval_threshold
        else:
            self.network_eval_threshold = None

        self.info_collector_class = lambda *args, **kwargs: SigmoidInfoCollector(*args, **kwargs)

    def _get_network_model(self):
        """ Get the regular softmax network model
        """
        model = getattr(models, self.config.arch)(last_relu=False)
        if self.config.pretrained != None:
            state_dict = self._get_pretrained_model_state_dict()
            model.load_state_dict(state_dict)
            del state_dict
        else:
            print("Using random initialized model")
        return model.to(self.device)

    def _get_open_set_pred_func(self):
        assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
        if self.config.network_eval_mode == 'threshold':
            threshold = self.config.network_eval_threshold
        elif self.config.network_eval_mode == 'dynamic_threshold':
            assert type(self.log) == list
            if len(self.log) == 0:
                # First round, use default threshold
                threshold = self.config.network_eval_threshold
                print(f"First round. Use default threshold {threshold}")
            else:
                try:
                    threshold = get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
                except NoSeenClassException:
                    # Error when no new instances from seen class
                    threshold = self.config.network_eval_threshold
                    print(f"No seen class instances. Threshold set to {threshold}")
                except NoUnseenClassException:
                    threshold = self.config.network_eval_threshold
                    print(f"No unseen class instances. Threshold set to {threshold}")
                else:
                    print(f"Threshold set to {threshold} based on all existing instances.")
        elif self.config.network_eval_mode in ['pseuopen_threshold']:
            assert hasattr(self, 'pseuopen_threshold')
            print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
            threshold = self.pseuopen_threshold

        def open_set_prediction(outputs, inputs=None):
            sigmoid_outputs = nn.Sigmoid()(outputs)
            sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            # if self.config.threshold_metric == 'softmax':
            #     scores = softmax_max
            # elif self.config.threshold_metric == 'entropy':
            #     scores = (softmax_outputs*softmax_outputs.log()).sum(dim=1) # negative entropy!
            scores = sigmoid_max

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

    def get_open_score_func(self):
        def open_score_func(inputs):
            outputs = self.model(inputs)
            sigmoid_outputs = nn.Sigmoid()(outputs)
            sigmoid_max, sigmoid_preds = torch.max(sigmoid_outputs, 1)
            scores = sigmoid_max
            return -scores
        return open_score_func


class OSDNNetwork(Network):
    def __init__(self, *args, **kwargs):
        # The Open Set Deep Network is essentially trained in the same way as a 
        # regular softmax network. The difference occurs during the eval stage.
        # So this is subclass from Network class, but has the eval function overwritten.
        super(OSDNNetwork, self).__init__(*args, **kwargs)
        # self.use_feature_before_fc = self.config.trainer in ['osdn_fc', 'osdn_modified_fc']

        assert self.config.threshold_metric == 'softmax'

        self.div_eu = self.config.div_eu
        self.distance_metric = self.config.distance_metric
        if self.distance_metric == 'eu':
            self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu)
        elif self.distance_metric == 'eucos':
            self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu) + cos_distance(a,b)
        elif self.distance_metric == 'cos':
            self.distance_func = cos_distance
        else:
            raise NotImplementedError()


        self.openmax_meta_learn = self.config.openmax_meta_learn
        if self.openmax_meta_learn == None:
            print("Using fixed OpenMax hyper")
            if 'fixed' in self.config.weibull_tail_size:
                self.weibull_tail_size = int(self.config.weibull_tail_size.split("_")[-1])
            else:
                raise NotImplementedError()

            if 'fixed' in self.config.alpha_rank: 
                self.alpha_rank = int(self.config.alpha_rank.split('_')[-1])
            else:
                raise NotImplementedError()

            self.osdn_eval_threshold = self.config.osdn_eval_threshold
        else:
            print("Using meta learning on pseudo-open class examples")
            self.weibull_tail_size = None
            self.alpha_rank = None
            self.osdn_eval_threshold = None
            self.pseudo_open_set_metric = self.config.pseudo_open_set_metric

        self.mav_features_selection = self.config.mav_features_selection
        self.weibull_distributions = None # The dictionary that contains all weibull related information
        # self.training_features = None # A dictionary holding the features of all examples

    def _meta_learning(self, pseudo_open_model, full_train_set, pseudo_seen_classes):
        ''' Meta learning on self.curr_full_train_set and self.pseudo_open_set_classes
            Pick and update the best openmax hyper including:
                self.weibull_tail_size
                self.alpha_rank
                self.osdn_eval_threshold
        '''
        print("Perform cross validation using pseudo open class..")
        assert hasattr(self, 'dataloaders') # Should be updated after calling super._train()
        training_features = self._gather_correct_features(pseudo_open_model,
                                                          self.dataloaders['train'], # from super._train()
                                                          seen_classes=pseudo_seen_classes,
                                                          mav_features_selection=self.mav_features_selection) # A dict of (key: seen_class_indice, value: A list of feature vector that has correct prediction in this class)
        
        from global_setting import OPENMAX_META_LEARN
        meta_setting = OPENMAX_META_LEARN[self.openmax_meta_learn]
        list_w_tail_size = meta_setting['weibull_tail_size']
        list_alpha_rank = meta_setting['alpha_rank']
        list_threshold = meta_setting['osdn_eval_threshold']

        curr_train_set = torch.utils.data.Subset(
                             self.train_instance.train_dataset,
                             full_train_set
                         )
        meta_learn_result = [] # A list of tuple: (acc : float, hyper_setting : dict)
        for tail_size in list_w_tail_size:
            for alpha_rank in list_alpha_rank:
                for threshold in list_threshold:
                    self.weibull_tail_size = tail_size
                    self.alpha_rank = alpha_rank
                    self.osdn_eval_threshold = threshold

                    self.weibull_distributions = self._gather_weibull_distribution(
                                                     training_features,
                                                     distance_metric=self.distance_metric,
                                                     weibull_tail_size=self.weibull_tail_size
                                                 ) # A dict of (key: seen_class_indice, value: A per channel, MAV + weibull model)

                    eval_result = self._eval(
                                      pseudo_open_model,
                                      curr_train_set,
                                      pseudo_seen_classes,
                                      # verbose=False,
                                      verbose=True,
                                      training_features=training_features # If not None, can save time by not recomputing it
                                  )
                    if self.pseudo_open_set_metric == 'weighted':
                        acc = eval_result['overall_acc']
                    elif self.pseudo_open_set_metric == 'average':
                        acc = (eval_result['seen_closed_acc'] + eval_result['all_open_acc']) / 2.
                    elif self.pseudo_open_set_metric == '7_3':
                        acc = 0.7 * eval_result['seen_closed_acc'] + 0.3 * eval_result['all_open_acc']
                    else:
                        raise NotImplementedError()
                    meta_learn_result.append(
                        { 'acc':acc,
                          'tail':tail_size,
                          'alpha':alpha_rank,
                          'threshold':threshold
                        }
                    )
        meta_learn_result.sort(reverse=True, key=lambda x:x['acc'])
        print("Meta Learning result (sorted by acc):")
        print(meta_learn_result)
        best_res = meta_learn_result[0]
        self.weibull_tail_size = best_res['tail']
        self.alpha_rank = best_res['alpha']
        self.osdn_eval_threshold = best_res['threshold']
        print(f"Updated to : W_TAIL={self.weibull_tail_size}, ALPHA={best_res['alpha']}, THRESHOLD={best_res['threshold']}")

    def _gather_correct_features(self, model, train_loader, seen_classes=set(), mav_features_selection='correct'):
        assert len(seen_classes) > 0
        assert mav_features_selection in ['correct', 'none_correct_then_all', 'all']
        mapping_func = get_target_mapping_func(self.train_instance.classes,
                                               seen_classes,
                                               self.train_instance.open_classes)
        target_mapping_func = self._get_target_mapp_func(seen_classes)
        seen_class_softmax_indices = [mapping_func(i) for i in seen_classes]

        if mav_features_selection == 'correct':
            print("Gather feature vectors for each class that are predicted correctly")
        elif mav_features_selection == 'all':
            print("Gather all feature vectors for each class")
        elif mav_features_selection == 'none_correct_then_all':
            print("Gather correctly predicted feature vectors for each class. If none, then use all examples")
        model.eval()
        if mav_features_selection in ['correct', 'all']:
            features_dict = {seen_class_softmax_index: [] for seen_class_softmax_index in seen_class_softmax_indices}
        elif mav_features_selection == 'none_correct_then_all':
            features_dict = {seen_class_softmax_index: {'correct':[],'all':[]} for seen_class_softmax_index in seen_class_softmax_indices}

        if self.config.verbose:
            pbar = tqdm(train_loader, ncols=80)
        else:
            pbar = train_loader

        # cur_features = []
        # if self.use_feature_before_fc:
        #     def forward_hook_func(self, inputs, outputs):
        #         cur_features.append(inputs[0])
        #     handle = self.model.fc.register_forward_hook(forward_hook_func)

        for batch, data in enumerate(pbar):
            inputs, real_labels = data
            
            inputs = inputs.to(self.device)
            labels = target_mapping_func(real_labels.to(self.device))

            with torch.set_grad_enabled(False):
                outputs = model(inputs)

                cur_features = []
                _, preds = torch.max(outputs, 1)
                correct_indices = preds == labels.data

                for i in range(inputs.size(0)):
                    if mav_features_selection == 'correct':
                        if not correct_indices[i]:
                            continue
                        else:
                            features_dict[int(labels[i])].append(outputs[i].unsqueeze(0))
                    elif mav_features_selection == 'all':
                        features_dict[int(labels[i])].append(outputs[i].unsqueeze(0))
                    elif mav_features_selection == 'none_correct_then_all':
                        if correct_indices[i]:
                            features_dict[int(labels[i])]['correct'].append(outputs[i].unsqueeze(0))

                        features_dict[int(labels[i])]['all'].append(outputs[i].unsqueeze(0))

        if mav_features_selection == 'none_correct_then_all':
            none_correct_classes = [] # All None correct classes
            new_features_dict = {seen_class_softmax_index: None for seen_class_softmax_index in seen_class_softmax_indices}
            for class_index in seen_class_softmax_indices:
                if len(features_dict[class_index]['correct']) == 0:
                    none_correct_classes.append(class_index)
                    if len(features_dict[class_index]['all']) == 0:
                        import pdb; pdb.set_trace()  # breakpoint 98f3a0ff //
                        print(f"No training example for {class_index}?")
                    new_features_dict[class_index] = features_dict[class_index]['all']
                else:
                    new_features_dict[class_index] = features_dict[class_index]['correct']
            print("These classes has no correct feature. So we use all inputs.")
            print(none_correct_classes)
            features_dict = new_features_dict

        return features_dict

    def _gather_weibull_distribution(self, training_features, distance_metric='eucos', weibull_tail_size=20):
        assert distance_metric in ['eucos', 'eu', 'cos']
        weibull = {seen_class_index : {'mav': None, 'eu_distances': None, 'cos_distances': None, 'eucos_distances': None, 'weibull_model': None} 
                   for seen_class_index in training_features.keys()}
        for index in training_features.keys():
            if not len(training_features[index]) > 0:
                print(f"Error: No training examples for category {index}")
                import pdb; pdb.set_trace()  # breakpoint 18e1e416 //
            else:
                features_tensor = torch.cat(training_features[index], dim=0)
                mav = torch.mean(features_tensor, 0)
                mav_matrix = mav.unsqueeze(0).expand(features_tensor.size(0), -1)
                eu_distances = torch.sqrt(torch.sum((mav_matrix - features_tensor) ** 2, dim=1)) / self.div_eu # EU distance divided by 200.
                cos_distances = 1 - torch.nn.CosineSimilarity(dim=1)(mav_matrix, features_tensor)
                eucos_distances = eu_distances + cos_distances

                weibull[index]['mav'] = mav
                weibull[index]['eu_distances'] = eu_distances
                weibull[index]['cos_distances'] = cos_distances
                weibull[index]['eucos_distances'] = eucos_distances

                if distance_metric == 'eu':
                    distance_scores = list(eu_distances)
                elif distance_metric == 'eucos':
                    distance_scores = list(eucos_distances)
                elif distance_metric == 'cos':
                    distance_scores = list(cos_distances)
                mr = libmr.MR()
                tailtofit = sorted(distance_scores)[-weibull_tail_size:]
                mr.fit_high(tailtofit, len(tailtofit))
                weibull[index]['weibull_model'] = mr
        return weibull

    def compute_open_max(self, outputs, log_thresholds=True):
        """ Return (openset_score, openset_preds)
        """
        alpha_weights = [((self.alpha_rank+1) - i)/float(self.alpha_rank) for i in range(1, self.alpha_rank+1)]
        softmax_outputs = F.softmax(outputs, dim=1)
        _, softmax_rank = torch.sort(softmax_outputs, descending=True)
        
        open_scores = torch.zeros(outputs.size(0)).to(outputs.device)
        for batch_i in range(outputs.size(0)):
            # if self.use_feature_before_fc:
            #     batch_i_output = features[batch_i]
            # else:
            batch_i_output = outputs[batch_i]
            batch_i_rank = softmax_rank[batch_i]
            
            for i in range(len(alpha_weights)):
                class_index = int(batch_i_rank[i])
                alpha_i = alpha_weights[i]
                distance = self.distance_func(self.weibull_distributions[class_index]['mav'], batch_i_output)
                wscore = self.weibull_distributions[class_index]['weibull_model'].w_score(distance)
                modified_score = batch_i_output[class_index] * (1 - alpha_i*wscore)
                open_scores[batch_i] += batch_i_output[class_index] * alpha_i*wscore
                batch_i_output[class_index] = modified_score # should change the corresponding score in outputs

        # total_denominators = torch.sum(torch.exp(outputs), dim=1) + torch.exp(open_scores)
        openmax_outputs = F.softmax(torch.cat((outputs, open_scores.unsqueeze(1)), dim=1), dim=1)
        openmax_max, openmax_preds = torch.max(openmax_outputs, 1)
        openmax_top2, openmax_preds2 = torch.topk(openmax_outputs, 2, dim=1)
        closed_preds = torch.where(openmax_preds == self.num_seen_classes, 
                                   openmax_preds2[:, 1],
                                   openmax_preds)
        closed_maxs = torch.where(openmax_preds == self.num_seen_classes, 
                                  openmax_top2[:, 1],
                                  openmax_max)
        openmax_predicted_label = torch.where(openmax_preds == self.num_seen_classes, 
                                              torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device),
                                              openmax_preds)
        if log_thresholds:
            # First update the open set stats
            self._update_open_set_stats(openmax_max, openmax_preds)

            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
            self.thresholds_checkpoints[self.round]['open_set_score'] += (openmax_outputs[:, self.num_seen_classes]).tolist()
            self.thresholds_checkpoints[self.round]['closed_predicted'] += closed_preds.tolist()
            self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += closed_maxs.tolist()
            self.thresholds_checkpoints[self.round]['open_predicted'] += openmax_predicted_label.tolist()
            self.thresholds_checkpoints[self.round]['open_argmax_prob'] += openmax_max.tolist()

        # Return the prediction
        preds = torch.where((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_seen_classes), 
                            torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device),
                            openmax_preds)
        return openmax_outputs, preds

    def get_open_score_func(self):
        def open_score_func(inputs):
            outputs = self.model(inputs)
            scores, _ = self.compute_open_max(outputs, log_thresholds=False)
            return scores[:, -1]
        return open_score_func

    def _get_open_set_pred_func(self):
        """ Caveat: Open set class is represented as -1.
        """
        return lambda outputs, inputs : self.compute_open_max(outputs)[1]

    def _get_open_set_score_func(self):
        """ Return the Openmax score. Say seen class has length 100, then return a tensor of length 101, where index 101 is the open set score
        """
        return lambda outputs, inputs : self.compute_open_max(outputs)[0]

    def _eval(self, model, test_dataset, seen_classes, verbose=False, training_features=None):
        assert hasattr(self, 'dataloaders') # Should be updated after calling super._train()
        if training_features == None:
            training_features = self._gather_correct_features(model,
                                                              self.dataloaders['train'], # from super._train()
                                                              seen_classes=seen_classes,
                                                              mav_features_selection=self.mav_features_selection) # A dict of (key: seen_class_indice, value: A list of feature vector that has correct prediction in this class)

        self.weibull_distributions = self._gather_weibull_distribution(training_features,
                                                                       distance_metric=self.distance_metric,
                                                                       weibull_tail_size=self.weibull_tail_size) # A dict of (key: seen_class_indice, value: A per channel, MAV + weibull model)

        assert len(self.weibull_distributions.keys()) == len(seen_classes)
        self.num_seen_classes = len(seen_classes)
        self._reset_open_set_stats() # Open set status is the summary of 1/ Number of threshold reject 2/ Number of Open Class reject
        eval_result = super(OSDNNetwork, self)._eval(model, test_dataset, seen_classes, verbose=verbose)
        if verbose:
            print(f"Rejection details: Total rejects {self.open_set_stats['total_reject']}. "
                  f"By threshold ({self.osdn_eval_threshold}) {self.open_set_stats['threshold_reject']}. "
                  f"By being open class {self.open_set_stats['open_class_reject']}. "
                  f"By both {self.open_set_stats['both_reject']}. ")
        return eval_result

    def _update_open_set_stats(self, openmax_max, openmax_preds):
        # For each batch
        self.open_set_stats['threshold_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & ~(openmax_preds == self.num_seen_classes) ))
        self.open_set_stats['open_class_reject'] += float(torch.sum(~(openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_seen_classes) ))
        self.open_set_stats['both_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) & (openmax_preds == self.num_seen_classes) ))
        self.open_set_stats['total_reject'] += float(torch.sum((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_seen_classes) ))
        assert self.open_set_stats['threshold_reject'] + self.open_set_stats['open_class_reject'] + self.open_set_stats['both_reject'] == self.open_set_stats['total_reject']

    def _reset_open_set_stats(self):
        # threshold_reject and open_class_reject are mutually exclusive
        self.open_set_stats = {'threshold_reject': 0., 
                               'open_class_reject': 0.,
                               'both_reject': 0.,
                               'total_reject': 0.}


class OSDNNetworkModified(OSDNNetwork):
    def __init__(self, *args, **kwargs):
        # This is essentially using the open max algorithm. But this modified version 
        # doesn't change the activation score of the seen classes.
        super(OSDNNetworkModified, self).__init__(*args, **kwargs)

    def compute_open_max(self, outputs, log_thresholds=True):
        """ Return (openset_score, openset_preds)
        """
        alpha_weights = [((self.alpha_rank+1) - i)/float(self.alpha_rank) for i in range(1, self.alpha_rank+1)]
        softmax_outputs = F.softmax(outputs, dim=1)
        _, softmax_rank = torch.sort(softmax_outputs, descending=True)
        
        open_scores = torch.zeros(outputs.size(0)).to(outputs.device)
        for batch_i in range(outputs.size(0)):
            batch_i_output = outputs[batch_i]
            batch_i_rank = softmax_rank[batch_i]
            
            for i in range(len(alpha_weights)):
                class_index = int(batch_i_rank[i])
                alpha_i = alpha_weights[i]
                distance = self.distance_func(self.weibull_distributions[class_index]['mav'], batch_i_output)
                wscore = self.weibull_distributions[class_index]['weibull_model'].w_score(distance)
                open_scores[batch_i] += batch_i_output[class_index] * alpha_i*wscore

        # total_denominators = torch.sum(torch.exp(outputs), dim=1) + torch.exp(open_scores)
        openmax_outputs = F.softmax(torch.cat((outputs, open_scores.unsqueeze(1)), dim=1), dim=1)
        openmax_max, openmax_preds = torch.max(openmax_outputs, 1)
        openmax_top2, openmax_preds2 = torch.topk(openmax_outputs, 2, dim=1)

        closed_preds = torch.where(openmax_preds == self.num_seen_classes, 
                                   openmax_preds2[:, 1],
                                   openmax_preds)
        closed_maxs = torch.where(openmax_preds == self.num_seen_classes, 
                                  openmax_top2[:, 1],
                                  openmax_max)
        openmax_predicted_label = torch.where(openmax_preds == self.num_seen_classes, 
                                              torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device),
                                              openmax_preds)
        
        if log_thresholds:
            # First update the open set stats
            self._update_open_set_stats(openmax_max, openmax_preds)

            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
            assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['closed_predicted'])
            self.thresholds_checkpoints[self.round]['open_set_score'] += (openmax_outputs[:, self.num_seen_classes]).tolist()
            self.thresholds_checkpoints[self.round]['closed_predicted'] += closed_preds.tolist()
            self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += closed_maxs.tolist()
            self.thresholds_checkpoints[self.round]['open_predicted'] += openmax_predicted_label.tolist()
            self.thresholds_checkpoints[self.round]['open_argmax_prob'] += openmax_max.tolist()

        
        preds = torch.where((openmax_max < self.osdn_eval_threshold) | (openmax_preds == self.num_seen_classes), 
                            torch.LongTensor([UNSEEN_CLASS_INDEX]).to(outputs.device),
                            openmax_preds)
        return openmax_outputs, preds

def get_acc_from_performance_dict(dict):
    result_dict = {}
    for k in dict.keys():
        if dict[k]['count'] == 0:
            result_dict[k] = "N/A"
        else:
            result_dict[k] = float(dict[k]['corrects']/dict[k]['count'])
    return result_dict
