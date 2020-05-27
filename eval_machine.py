import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, roc_auc_score
import os
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func_for_tensor, get_target_unmapping_dict, get_target_mapping_func, get_target_unmapping_func_for_list
from global_setting import OPEN_CLASS_INDEX, UNDISCOVERED_CLASS_INDEX, PRETRAINED_MODEL_PATH
import libmr
import math

def calc_auc_score(x, y):
    # x and y should be bounded between [0,1]
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area

def get_eval_machine(open_set_method, trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path):
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
    return eval_machine_class(trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path)

class EvalMachine(object):
    def __init__(self, trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path):
        super().__init__()
        self.trainset_info = trainset_info
        self.trainer_config = trainer_config

        self.roc_path = open_result_roc_path
        self.goscr_path = open_result_goscr_path

        self.batch = trainer_config['batch']
        self.workers = trainer_config['workers']
        self.device = trainer_config['device']

    def _get_target_mapp_func(self, discovered_classes):
        return get_target_mapping_func_for_tensor(self.trainset_info.classes,
                                                  discovered_classes,
                                                  self.trainset_info.open_classes,
                                                  device=self.device)
    
    def _get_target_unmapping_func_for_list(self, discovered_classes):
        return get_target_unmapping_func_for_list(self.trainset_info.classes, discovered_classes)
         
    def eval_open_set(self, discovered_classes, test_dataset, trainer_machine, result_path=None, verbose=True):
        """ Performing open set evaluation
        """
        if os.path.exists(result_path):
            print("Open set result already saved.")
            self.open_set_result = torch.load(result_path)
        else:
            self.open_set_result = self._eval_open_set_helper(discovered_classes,
                                                              test_dataset,
                                                              trainer_machine,
                                                              verbose=verbose)
            torch.save(self.open_set_result, result_path)
            
            self._plot_open_set_result(self.open_set_result, self.roc_path, self.goscr_path)
        
        

    def _eval_open_set_helper(self, discovered_classes, test_dataset, trainer_machine, verbose=True):
        raise NotImplementedError()

class NetworkOpen(EvalMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _eval_open_set_helper(self, discovered_classes, test_dataset, trainer_machine, verbose=True):
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

            open_set_prediction = self._get_open_set_pred_func(trainer_machine)
            open_set_result = {'ground_truth' : [], # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
                               'real_labels' : [], # The real labels for CIFAR100 or other datasets.
                               'open_set_score' : [], # Higher the score, more likely to be open set
                               'closed_predicted' : [], # The predicted closed set label (indices in network output) if not open set?
                               'closed_predicted_real' : [], # The predicted closed set label (real labels) if not open set?
                               'closed_argmax_prob' : [], # The probability for predicted closed set class if not open set?
                               'open_predicted' : [], # What is the true predicted label including open set/ for k class method, this is same as above (network predicted output)
                               'open_predicted_real' : [], # What is the true predicted label including open set/ for k class method, this is same as above (real labels)
                               'open_argmax_prob' : [], # What is the probability of the true predicted label including open set/ for k class method, this is same as above
                               } # A list of dictionary

            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, real_labels = data

                    inputs = inputs.to(self.device)
                    labels = target_mapping_func(real_labels.to(self.device))
                    labels_for_openset_pred = torch.where(
                                                  labels == OPEN_CLASS_INDEX,
                                                  torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(labels.device),
                                                  labels
                                              ) # This change hold out open set examples' indices to unseen open set examples indices

                    open_set_result_i = open_set_prediction(inputs) # Open set index == UNDISCOVERED_CLASS_INDEX

                    open_set_result['ground_truth'] += labels.tolist()
                    open_set_result['real_labels'] += real_labels.tolist()
                    open_set_result['closed_predicted'] += open_set_result_i['closed_predicted']
                    open_set_result['open_predicted'] += open_set_result_i['open_predicted']
                    open_set_result['closed_predicted_real'] += target_unmapping_func_for_list(open_set_result_i['closed_predicted'])
                    open_set_result['open_predicted_real'] += target_unmapping_func_for_list(open_set_result_i['open_predicted'])
                    open_set_result['open_set_score'] += open_set_result_i['open_set_score']
                    open_set_result['closed_argmax_prob'] += open_set_result_i['closed_argmax_prob']
                    open_set_result['open_argmax_prob'] += open_set_result_i['open_argmax_prob']
        
        open_set_result['roc'] = self._parse_roc_result(open_set_result)
        open_set_result['goscr'] = self._parse_goscr_result(open_set_result)
        return open_set_result
    
    def _parse_roc_result(self, open_set_result):
        res = {'fpr' : None,
               'tpr' : None,
               'auroc' : None}
        # Discovered v.s. Hold-out open
        gt = np.array(open_set_result['ground_truth']) # 0 if closed set, UNDISCOVERED_CLASS_INDEX if unseen open set, OPEN_CLASS_INDEX if hold out open set
        open_scores = np.array(open_set_result['open_set_score'])
        gt[gt >= 0] = 0
        gt[gt == OPEN_CLASS_INDEX] = 1
        selected_indices = gt != UNDISCOVERED_CLASS_INDEX
        open_scores = open_scores[selected_indices]
        gt = gt[selected_indices]

        try:
            if np.any(np.isnan(open_scores)):
                print(f"There is {np.sum(np.isnan(open_scores))} NaN values. Replace them by the mean of remaining scores.")
                open_scores[np.where(np.isnan(open_scores))] = open_scores.nanmean()
            fpr, tpr, _ = roc_curve(gt, open_scores)
            auc_score = roc_auc_score(gt, open_scores)
        except:
            print(f"Wrong AUC!!!")
            import pdb; pdb.set_trace()

        res = {'fpr' : fpr, 'tpr' : tpr, 'auc_score' : auc_score}
        plt.figure(figsize=(10,10))
        axes = plt.gca()
        axes.set_ylim([0,1])
        axes.set_xlim([0,1])
        plt.title(f'ROC curve plot', y=0.96, fontsize=12)
        plt.xlabel("False Positive Rate (Closed set examples classified as open set)", fontsize=12)
        plt.ylabel("True Positive Rate (Open set example classified as open set)", fontsize=12)

        label_name = f"AUC_"+f"{auc_score:.3f}"
        plt.plot(fpr, tpr, label=label_name, linestyle='-')
        plt.legend(loc='upper left',
                   borderaxespad=0., fontsize=10)
        plt.tight_layout()
        plt.savefig(self.roc_path)
        plt.close('all')
        return res
    
    def _parse_goscr_result(self, open_set_result):
        # Discovered v.s. Hold-out open
        gt = np.array(open_set_result['ground_truth'])
        open_predicted = np.array(open_set_result['open_predicted'])
        open_scores = np.array(open_set_result['open_set_score'])
        selected_indices = gt != UNDISCOVERED_CLASS_INDEX
        gt = gt[selected_indices]
        open_predicted = open_predicted[selected_indices]
        open_scores = open_scores[selected_indices]
        # At this point, gt's open example have label OPEN_CLASS_INDEX.

        sorted_indices = np.argsort(open_scores) # Sort from smallest to largest
        FP = [0] # Number of wrongly classified open set example
        TC = [0] # Number of correctly classified closed set examples
        N = 0 # A counter of correctly classified closed set examples with open score higher than prior threshold
        N_below_threshold = 0 # A counter of correctly classified closed set examples
        threshold = open_scores.min() # We slide the threshold from low to high
        total_corrects = 0.
        if np.any(np.isnan(open_scores)):
            print(f"There is {np.sum(np.isnan(open_scores))} NaN values. Replace them by the mean of remaining scores.")
            try:
                open_scores[np.where(np.isnan(open_scores))] = open_scores.nanmean()
            except:
                print("Wrong AUC")
                import pdb; pdb.set_trace()

        for idx in sorted_indices:
            gt_label = gt[idx]
            curr_open_score = open_scores[idx]
            openpred_label = open_predicted[idx] # For K+1 method, this could be OPEN_CLASS_INDEX
            if gt_label == OPEN_CLASS_INDEX:
                threshold = curr_open_score
                FP.append(FP[-1]+1) # One more open set example wrongly classified
                TC.append(N) # N more correct classified closed example with open score <= prior threshold
            else:
                if gt_label == openpred_label:
                    # Correct prediction
                    total_corrects += 1
                    N_below_threshold += 1
                    if curr_open_score > threshold:
                        # if curr_open_score >= threshold: # TODO: Figure out which one to use
                        N = N_below_threshold

        num_closed_set = (gt != OPEN_CLASS_INDEX).sum()
        num_open_set = (gt == OPEN_CLASS_INDEX).sum()
        FPR = np.array(FP[1:]).astype(np.float32) / float(num_open_set)
        TCR = np.array(TC[1:]).astype(np.float32) / float(num_closed_set)
        auc_score = calc_auc_score(FPR, TCR)
        max_acc = total_corrects / num_closed_set
        res = {'fpr' : FPR, 'tcr' : TCR, 'max_acc' : max_acc, 'auc_score' : auc_score}

        plt.figure(figsize=(10,10))
        axes = plt.gca()
        axes.set_ylim([0,1])
        axes.set_xlim([0,1])
        plt.title(f'Open set classification rate plot', y=0.96, fontsize=12)
        axes.set_xscale('log')
        axes.autoscale(enable=True, axis='x', tight=True)
        plt.xlabel("False Positive Rate (Open set examples classified as closed set)", fontsize=12)
        plt.ylabel("Correct Classification Rate (Closed set examples classified into correct class)", fontsize=12)

        label_name = f"AUC_"+f"{auc_score:.3f}"#+"MAXACC_"+str(max_acc)
        plt.plot(FPR, TCR, label=label_name, linestyle='-')
        plt.legend(loc='upper left',
                   borderaxespad=0., fontsize=10)

        plt.tight_layout()
        plt.savefig(self.goscr_path)
        plt.close('all')
        return res

    def _get_open_set_pred_func(self, trainer_machine):
        raise NotImplementedError()           

# class OpenmaxOpen(NetworkOpen):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.div_eu = ?
    #     self.distance_func = ?
    #     self.openmax_meta_learn = self.config.openmax_meta_learn
    #     if self.openmax_meta_learn == None:
    #         print("Using fixed OpenMax hyper")
    #         if 'fixed' in self.config.weibull_tail_size:
    #             self.weibull_tail_size = int(self.config.weibull_tail_size.split("_")[-1])
    #         else:
    #             raise NotImplementedError()

    #         if 'fixed' in self.config.alpha_rank: 
    #             self.alpha_rank = int(self.config.alpha_rank.split('_')[-1])
    #         else:
    #             raise NotImplementedError()

    #         self.osdn_eval_threshold = self.config.osdn_eval_threshold
    #     else:
    #         print("Using meta learning on pseudo-open class examples")
    #         self.weibull_tail_size = None
    #         self.alpha_rank = None
    #         self.osdn_eval_threshold = None
    #         self.pseudo_open_set_metric = self.config.pseudo_open_set_metric

    #     self.mav_features_selection = self.config.mav_features_selection
    #     self.weibull_distributions = None # The dictionary that contains all weibull related information
    #     # self.training_features = None # A dictionary holding the features of all examples

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

    # def _get_open_set_pred_func(self, trainer_machine):
    #     def open_set_prediction(inputs):
    #         open_set_result_i = {}
    #         outputs = trainer_machine.get_class_scores(inputs)
    #         softmax_outputs = F.softmax(outputs, dim=1)
    #         softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
    #         if self.config.threshold_metric == 'softmax':
    #             scores = softmax_max
    #         elif self.config.threshold_metric == 'entropy':
    #             neg_entropy = softmax_outputs*softmax_outputs.log()
    #             neg_entropy[softmax_outputs < 1e-5] = 0
    #             scores = neg_entropy.sum(dim=1) # negative entropy!

    #         open_set_result_i['open_set_score']     += (-softmax_max).tolist()
    #         open_set_result_i['closed_predicted']   += softmax_preds.tolist()
    #         open_set_result_i['closed_argmax_prob'] += softmax_max.tolist()
    #         open_set_result_i['open_predicted']     += softmax_preds.tolist()
    #         open_set_result_i['open_argmax_prob']   += softmax_max.tolist()
    #         # open_set_result_i['actual_loss']        += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()
    #         return open_set_result_i
    #     return open_set_prediction

class SoftmaxOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_open_set_pred_func(self, trainer_machine):
        def open_set_prediction(inputs):
            open_set_result_i = {}
            outputs = trainer_machine.get_class_scores(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)

            open_set_result_i['open_set_score']     += (-softmax_max).tolist()
            open_set_result_i['closed_predicted']   += softmax_preds.tolist()
            open_set_result_i['closed_argmax_prob'] += softmax_max.tolist()
            open_set_result_i['open_predicted']     += softmax_preds.tolist()
            open_set_result_i['open_argmax_prob']   += softmax_max.tolist()
            # open_set_result_i['actual_loss']        += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()
            return open_set_result_i
        return open_set_prediction


class EntropyOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_open_set_pred_func(self, trainer_machine):
        def open_set_prediction(inputs):
            open_set_result_i = {}
            outputs = trainer_machine.get_class_scores(inputs)
            softmax_outputs = F.softmax(outputs, dim=1)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            neg_entropy = softmax_outputs*softmax_outputs.log()
            neg_entropy[softmax_outputs < 1e-5] = 0
            scores = neg_entropy.sum(dim=1) # negative entropy!

            open_set_result_i['open_set_score']     += (-scores).tolist()
            open_set_result_i['closed_predicted']   += softmax_preds.tolist()
            open_set_result_i['closed_argmax_prob'] += softmax_max.tolist()
            open_set_result_i['open_predicted']     += softmax_preds.tolist()
            open_set_result_i['open_argmax_prob']   += softmax_max.tolist()
            # open_set_result_i['actual_loss']        += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()
            return open_set_result_i
        return open_set_prediction

class NNOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class C2AEOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)