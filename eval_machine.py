import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy

import os
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list
from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX, PRETRAINED_MODEL_PATH
import libmr
import math


def get_eval_machine(open_set_method, trainset_info, trainer_config):
    return None
    
    if open_set_method == 'entropy':
        eval_machine_class = EntropyOpen
    elif open_set_method == "openmax":
        eval_machine_class = OpenmaxOpen
    elif open_set_method == 'softmax':
        eval_machine_class = SoftmaxOpen
    elif open_set_method == 'c2ae':
        eval_machine_class = C2AEOpen
    elif open_set_method == 'nn':
        eval_machine_class = NNOpen
    else:
        raise NotImplementedError()
    return eval_machine_class(trainset_info, trainer_config)

# class EvalMachine(object):
#     def __init__(self, trainset_info, trainer_config):
#         super().__init__()
#         self.trainset_info = trainset_info
#         self.trainer_config = trainer_config

#         self.batch = trainer_config['batch']
#         self.workers = trainer_config['workers']
#         self.device = trainer_config['device']

#     def _get_target_mapp_func(self, discovered_classes):
#         return get_target_mapping_func_for_tensor(self.trainset_info.classes,
#                                                   discovered_classes,
#                                                   self.trainset_info.open_classes,
#                                                   device=self.device)
    
#     def _get_target_unmapping_func_for_list(self, discovered_classes):
#         return get_target_unmapping_func_for_list(self.trainset_info.classes, discovered_classes)
         
#     def eval_open_set(self, discovered_classes, test_dataset, trainer_machine, result_path=None, verbose=True):
#         """ Performing open set evaluation
#         """
#         if os.path.exists(result_path):
#             print("Open set result already saved.")
#             self.open_set_result = torch.load(result_path)
#         else:
#             self.open_set_result = self._eval_open_set_helper(discovered_classes,
#                                                               test_dataset,
#                                                               trainer_machine,
#                                                               verbose=verbose)
#             torch.save(self.open_set_result, result_path)

#     def _eval_open_set_helper(self, discovered_classes, test_dataset, trainer_machine, verbose=True):
#         raise NotImplementedError()
    

# class NetworkOpen(EvalMachine):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def _eval_open_set_helper(self, discovered_classes, test_dataset, trainer_machine, verbose=True):
#         target_mapping_func = self._get_target_mapp_func(discovered_classes)
#         target_unmapping_func_for_list = self._get_target_unmapping_func_for_list(discovered_classes) # Only for transforming predicted label (in network indices) to real indices
#         dataloader = get_loader(test_dataset,
#                                 None,
#                                 shuffle=False,
#                                 batch_size=self.batch,
#                                 workers=self.workers)
        
#         with SetPrintMode(hidden=not verbose):
#             if verbose:
#                 pbar = tqdm(dataloader, ncols=80)
#             else:
#                 pbar = dataloader

#             performance_dict = {'corrects' : 0., 'not_seen' : 0., 'count' : 0.} # Accuracy of all non-hold out open class examples. If some classes not seen yet, accuracy = 0
#             open_set_result = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
#                                'real_labels' : [], # The real labels for CIFAR100 or other datasets.
#                                'open_set_score' : [], # Higher the score, more likely to be open set
#                                'closed_predicted' : [], # If fail the open set detection, then what's the predicted closed set label (network output)?
#                                'closed_predicted_real' : [], # If fail the open set detection, then what's the predicted closed set label (real labels)?
#                                'closed_argmax_prob' : [], # If fail the open set detection, then what's the probability for predicted closed set class (real labels)?
#                                'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
#                                'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
#                                'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
#                                'learningloss_pred_loss' : [], # Loss predicted by learning loss
#                                'actual_loss' : [], # actual losses
#                                } # A list of dictionary

#             with torch.no_grad():
#                 for batch, data in enumerate(pbar):
#                     inputs, real_labels = data

#                     inputs = inputs.to(self.device)
#                     labels = target_mapping_func(real_labels.to(self.device))

#                     outputs = self.classifier(self.backbone(inputs))
#                     _, preds = torch.max(outputs, 1)

#                     undiscovered_open_indices = labels == UNDISCOVERED_CLASS_INDEX
#                     discovered_closed_indices = labels >= 0
#                     hold_out_open_indices = labels == OPEN_CLASS_INDEX
#                     unlabeled_pool_class_indices = undiscovered_open_indices | discovered_closed_indices
#                     assert torch.sum(undiscovered_open_indices & discovered_closed_indices & hold_out_open_indices) == 0

#                     performance_dict['count'] += float(torch.sum(unlabeled_pool_class_indices))

#                     performance_dict['not_seen'] += torch.sum(undiscovered_open_indices).float()
#                     performance_dict['corrects'] += torch.sum(
#                                                                  torch.masked_select(
#                                                                      (preds==labels.data),
#                                                                      discovered_closed_indices
#                                                                  )
#                                                              ).float()
#             test_acc = performance_dict['corrects'] / performance_dict['count']
#             seen_rate = 1. - performance_dict['not_seen'] / performance_dict['count']
            
            
#             print(f"Test => "
#                   f"Closed set test Acc {test_acc}, "
#                   f"Discovered precentage {seen_rate}")

#             print(f"Test Accuracy {test_acc}.")
#         test_dict = {
#             'acc' : test_acc,
#             'seen' : seen_rate
#         }
#         return test_dict
#         open_set_prediction = self._get_open_set_pred_func()

# # def _get_open_set_pred_func(self):
#     #         assert self.config.network_eval_mode in ['threshold', 'dynamic_threshold', 'pseuopen_threshold']
# #         if self.config.network_eval_mode == 'threshold':
# #             # assert self.config.threshold_metric == "softmax"
# #             threshold = self.config.network_eval_threshold
# #         elif self.config.network_eval_mode == 'dynamic_threshold':
# #             # raise NotImplementedError()
# #             assert type(self.log) == list
# #             if len(self.log) == 0:
# #                 # First round, use default threshold
# #                 threshold = self.config.network_eval_threshold
# #                 print(f"First round. Use default threshold {threshold}")
# #             else:
# #                 try:
# #                     threshold = trainer_machine.get_dynamic_threshold(self.log, metric=self.config.threshold_metric, mode='weighted')
# #                 except trainer_machine.NoSeenClassException:
# #                     # Error when no new instances from seen class
# #                     threshold = self.config.network_eval_threshold
# #                     print(f"No seen class instances. Threshold set to {threshold}")
# #                 except trainer_machine.NoUnseenClassException:
# #                     threshold = self.config.network_eval_threshold
# #                     print(f"No unseen class instances. Threshold set to {threshold}")
# #                 else:
# #                     print(f"Threshold set to {threshold} based on all existing instances.")
# #         elif self.config.network_eval_mode in ['pseuopen_threshold']:
# #             # assert hasattr(self, 'pseuopen_threshold')
# #             # print(f"Using pseudo open set threshold of {self.pseuopen_threshold}")
# #             # threshold = self.pseuopen_threshold
# #             raise NotImplementedError()

# #         def open_set_prediction(outputs, inputs=None, features=None, label_for_learnloss=None):
# #             if self.icalr_strategy == 'proto':
# #                 # First step is to normalize the features
# #                 features = features / ((features*features).sum(1) ** 0.5).unsqueeze(1)

# #                 difference = (features.unsqueeze(1) - self.proto.unsqueeze(0)).abs()
# #                 distances = (difference*difference).sum(2)**0.5
# #                 outputs = -distances

# #             softmax_outputs = F.softmax(outputs, dim=1)
# #             softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
# #             if self.config.threshold_metric == 'softmax':
# #                 scores = softmax_max
# #             elif self.config.threshold_metric == 'entropy':
# #                 neg_entropy = softmax_outputs*softmax_outputs.log()
# #                 neg_entropy[softmax_outputs < 1e-5] = 0
# #                 scores = neg_entropy.sum(dim=1) # negative entropy!

# #             assert len(self.thresholds_checkpoints[self.round]['open_set_score']) == len(self.thresholds_checkpoints[self.round]['ground_truth'])
# #             self.thresholds_checkpoints[self.round]['open_set_score'] += (-scores).tolist()
# #             self.thresholds_checkpoints[self.round]['closed_predicted'] += softmax_preds.tolist()
# #             self.thresholds_checkpoints[self.round]['closed_argmax_prob'] += softmax_max.tolist()
# #             self.thresholds_checkpoints[self.round]['open_predicted'] += softmax_preds.tolist()
# #             self.thresholds_checkpoints[self.round]['open_argmax_prob'] += softmax_max.tolist()
# #             self.thresholds_checkpoints[self.round]['actual_loss'] += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()


# #             preds = torch.where(scores < threshold,
# #                                 torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device), 
# #                                 softmax_preds)
# #             return preds
# #         return open_set_prediction

# #         with SetPrintMode(hidden=not verbose):
# #             if verbose:
# #                 pbar = tqdm(dataloader, ncols=80)
# #             else:
# #                 pbar = dataloader

                
# #             with torch.no_grad():
# #                 for batch, data in enumerate(pbar):
# #                     inputs, real_labels = data
                    
# #                     inputs = inputs.to(self.device)
# #                     labels = target_mapping_func(real_labels.to(self.device))
# #                     labels_for_openset_pred = torch.where(
# #                                                   labels == OPEN_CLASS_INDEX,
# #                                                   torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(labels.device),
# #                                                   labels
# #                                               ) # This change hold out open set examples' indices to unseen open set examples indices
# #                     label_for_learnloss = labels.clone()
# #                     label_for_learnloss[label_for_learnloss<0] = 0
                    
# #                     outputs = model(inputs)
# #                     if self.icalr_strategy == 'proto':
# #                         preds = open_set_prediction(outputs, inputs=inputs, features=cur_features[0], label_for_learnloss=label_for_learnloss) # Open set index == UNDISCOVERED_CLASS_INDEX
# #                         cur_features = []
# #                     elif self.icalr_strategy in ['naive', 'smooth']:
# #                         preds = open_set_prediction(outputs, inputs=inputs, features=None, label_for_learnloss=label_for_learnloss) # Open set index == UNDISCOVERED_CLASS_INDEX
# #                     # loss = open_set_criterion(outputs, labels)

# #                     # labels_for_ground_truth = torch.where(
# #                     #                               (labels != OPEN_CLASS_INDEX) & (labels != UNDISCOVERED_CLASS_INDEX),
# #                     #                               torch.LongTensor([0]).to(labels.device),
# #                     #                               labels
# #                     #                           ) # This change hold out open set examples' indices to unseen open set examples indices

# #                     self.thresholds_checkpoints[self.round]['ground_truth'] += labels.tolist()
# #                     self.thresholds_checkpoints[self.round]['real_labels'] += real_labels.tolist()
                    
# #                     undiscovered_open_indices = labels == UNDISCOVERED_CLASS_INDEX
# #                     discovered_closed_indices = labels >= 0
# #                     hold_out_open_indices = labels == OPEN_CLASS_INDEX
# #                     unlabeled_pool_class_indices = undiscovered_open_indices | discovered_closed_indices
# #                     all_open_indices = undiscovered_open_indices | hold_out_open_indices
# #                     assert torch.sum(undiscovered_open_indices & discovered_closed_indices & hold_out_open_indices) == 0
                    


# #             self.thresholds_checkpoints[self.round]['closed_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['closed_predicted'])
# #             self.thresholds_checkpoints[self.round]['open_predicted_real'] = target_unmapping_func_for_list(self.thresholds_checkpoints[self.round]['open_predicted'])
            
        
# #         return epoch_result

#     elif open_set_method == "openmax":
#         eval_machine_class = OpenmaxOpen
#     elif open_set_method == 'softmax':
#         eval_machine_class = SoftmaxOpen
#     elif open_set_method == 'c2ae':
#         eval_machine_class = C2AEOpen
#     elif open_set_method == 'nn':
#         eval_machine_class = NNOpen