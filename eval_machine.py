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
from distance import eu_distance, cos_distance

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

def get_eval_machine(open_set_method, trainer_machine, trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path):
    if open_set_method == 'entropy':
        eval_machine_class = EntropyOpen
    elif open_set_method == "openmax":
        eval_machine_class = OpenmaxOpen
    elif open_set_method == 'softmax':
        eval_machine_class = SoftmaxOpen
    elif open_set_method == 'nn':
        eval_machine_class = NNOpen
    elif open_set_method == 'nn_cosine':
        eval_machine_class = NNCosineOpen
    else:
        raise NotImplementedError()
    return eval_machine_class(trainer_machine, trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path)

class EvalMachine(object):
    def __init__(self, trainer_machine, trainset_info, trainer_config, open_result_roc_path, open_result_goscr_path):
        super().__init__()
        self.trainer_machine = trainer_machine
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
         
    def eval_open_set(self, discovered_samples, discovered_classes, test_dataset, result_path=None, verbose=True):
        """ Performing open set evaluation
        """
        if os.path.exists(result_path):
            print(f"Open set result already saved at {result_path}.")
            self.open_set_result = torch.load(result_path)
        else:
            self.open_set_result = self._eval_open_set_helper(discovered_samples,
                                                              discovered_classes,
                                                              test_dataset,
                                                              verbose=verbose)
            torch.save(self.open_set_result, result_path)
            
        

    def _eval_open_set_helper(self, discovered_samples, discovered_classes, test_dataset, verbose=True):
        raise NotImplementedError()

class NetworkOpen(EvalMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _eval_open_set_helper(self, discovered_samples, discovered_classes, test_dataset, verbose=True, do_plot=True):
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

            open_set_prediction = self._get_open_set_pred_func()
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
        
        open_set_result['roc'] = self._parse_roc_result(open_set_result, do_plot=do_plot)
        open_set_result['goscr'] = self._parse_goscr_result(open_set_result, do_plot=do_plot)
        return open_set_result
    
    def _parse_roc_result(self, open_set_result, do_plot=True):
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
        if do_plot:
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
            print(f"ROC plot saved at {self.roc_path} with AUROC {auc_score:.3f}")
            plt.close('all')
        return res
    
    def _parse_goscr_result(self, open_set_result, do_plot=True):
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
        
        if do_plot:
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
            print(f"GOSCR plot saved at {self.goscr_path} with AUGOSCR {auc_score:.3f}")
            plt.close('all')
        return res

    def _get_open_set_pred_func(self):
        raise NotImplementedError()           

class OpenmaxOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.weibull_tail_sizes = [20]
        # self.alpha_ranks = [20, 80]
        # self.div_eus = [1000] # EU distance will be divided by this number
        self.weibull_tail_size = 20
        self.alpha_rank = 10
        self.div_eu = 1000.
        self.mav_features_selection = "none_correct_then_all"

    def _eval_open_set_helper(self, discovered_samples, discovered_classes, test_dataset, verbose=True):
        train_loader = self.trainer_machine.get_trainloader(discovered_samples, shuffle=False)
        self.num_discovered_classes = len(discovered_classes)

        # all_open_set_results = []
        # for weibull_tail_size in self.weibull_tail_sizes:
        #     for alpha_rank in self.alpha_ranks:
        #         for div_eu in self.div_eus:
        #             self.div_eu = div_eu
        #             self.alpha_rank = alpha_rank
        #             self.weibull_tail_size = weibull_tail_size
        #             self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu) + cos_distance(a,b)
        #             features_dict = self._gather_correct_features(train_loader, discovered_classes, mav_features_selection=self.mav_features_selection, verbose=verbose)
        #             self.weibull_distributions = self._gather_weibull_distribution(features_dict, div_eu)
        #             open_set_result_i = super()._eval_open_set_helper(discovered_samples, discovered_classes, test_dataset, verbose=verbose, do_plot=False)
        #             roc_i = open_set_result_i['roc']['auc_score']
        #             print(f"AUROC {roc_i}, div_eu {div_eu}, alpha {alpha_rank}, tail {weibull_tail_size}")
        #             d = {'roc' : roc_i,
        #                  'div_eu' : div_eu,
        #                  'alpha_rank' : alpha_rank,
        #                  'weibull_tail_size': weibull_tail_size}
        #             all_open_set_results.append(d)
        # all_open_set_results.sort(key=lambda x: x['roc'], reverse=True)
        # for res in all_open_set_results:
        #     tail = res['weibull_tail_size']
        #     score = res['roc']
        #     alpha_rank = res['alpha_rank']
        #     div_eu = res['div_eu']
        #     print(f"AUROC {score}, div_eu {div_eu}, alpha {alpha_rank}, tail {tail}")
        # res = all_open_set_results[0]
        # self.weibull_tail_size = res['weibull_tail_size']
        # self.alpha_rank = res['alpha_rank']
        # self.div_eu = res['div_eu']
        self.distance_func = lambda a, b: eu_distance(a,b,div_eu=self.div_eu) + cos_distance(a,b)
        features_dict = self._gather_correct_features(train_loader, discovered_classes, mav_features_selection=self.mav_features_selection, verbose=verbose)
        self.weibull_distributions = self._gather_weibull_distribution(features_dict, self.div_eu)
        return super()._eval_open_set_helper(discovered_samples, discovered_classes, test_dataset, verbose=verbose, do_plot=True)
    

    def _gather_correct_features(self, train_loader, discovered_classes=set(), mav_features_selection='correct', verbose=False):
        assert len(discovered_classes) > 0
        assert mav_features_selection in ['correct', 'none_correct_then_all', 'all']
        mapping_func = get_target_mapping_func(self.trainset_info.classes,
                                               discovered_classes,
                                               self.trainset_info.open_classes)
        target_mapping_func = self._get_target_mapp_func(discovered_classes)
        seen_class_softmax_indices = [mapping_func(i) for i in discovered_classes]

        if mav_features_selection == 'correct':
            print("Gather feature vectors for each class that are predicted correctly")
        elif mav_features_selection == 'all':
            print("Gather all feature vectors for each class")
        elif mav_features_selection == 'none_correct_then_all':
            print("Gather correctly predicted feature vectors for each class. If none, then use all examples")
        
        if mav_features_selection in ['correct', 'all']:
            features_dict = {seen_class_softmax_index: [] for seen_class_softmax_index in seen_class_softmax_indices}
        elif mav_features_selection == 'none_correct_then_all':
            features_dict = {seen_class_softmax_index: {'correct':[],'all':[]} for seen_class_softmax_index in seen_class_softmax_indices}

        if verbose:
            pbar = tqdm(train_loader, ncols=80)
        else:
            pbar = train_loader


        for batch, data in enumerate(pbar):
            inputs, real_labels = data
            
            inputs = inputs.to(self.device)
            labels = target_mapping_func(real_labels.to(self.device))

            with torch.set_grad_enabled(False):
                outputs = self.trainer_machine.get_class_scores(inputs)
                features = self.trainer_machine.get_features(inputs)
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

    def _gather_weibull_distribution(self, training_features, div_eu, weibull_tail_size=20):
        weibull = {seen_class_index : {'mav': None, 'eucos_distances': None, 'weibull_model': None} 
                   for seen_class_index in training_features.keys()}
        for index in training_features.keys():
            if not len(training_features[index]) > 0:
                print(f"Error: No training examples for category {index}")
                import pdb; pdb.set_trace()  # breakpoint 18e1e416 //
            else:
                features_tensor = torch.cat(training_features[index], dim=0)
                mav = torch.mean(features_tensor, 0)
                mav_matrix = mav.unsqueeze(0).expand(features_tensor.size(0), -1)
                eu_distances = torch.sqrt(torch.sum((mav_matrix - features_tensor) ** 2, dim=1)) / div_eu # EU distance divided by div_eu.
                cos_distances = 1 - torch.nn.CosineSimilarity(dim=1)(mav_matrix, features_tensor)
                eucos_distances = eu_distances + cos_distances

                weibull[index]['mav'] = mav
                # weibull[index]['eucos_distances'] = eucos_distances

                distance_scores = list(eucos_distances)
                mr = libmr.MR()
                tailtofit = sorted(distance_scores)[-weibull_tail_size:]
                mr.fit_high(tailtofit, len(tailtofit))
                weibull[index]['weibull_model'] = mr
        return weibull

    def compute_open_max(self, inputs, log_thresholds=True):
        """ Return (openset_score, openset_preds)
        """
        open_set_result_i = {}
        outputs = self.trainer_machine.get_class_scores(inputs)
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
                modified_score = batch_i_output[class_index] * (1 - alpha_i*wscore)
                open_scores[batch_i] += batch_i_output[class_index] * alpha_i*wscore
                batch_i_output[class_index] = modified_score # should change the corresponding score in outputs

        # total_denominators = torch.sum(torch.exp(outputs), dim=1) + torch.exp(open_scores)
        openmax_outputs = F.softmax(torch.cat((outputs, open_scores.unsqueeze(1)), dim=1), dim=1)
        openmax_max, openmax_preds = torch.max(openmax_outputs, 1)
        openmax_top2, openmax_preds2 = torch.topk(openmax_outputs, 2, dim=1)
        closed_preds = torch.where(openmax_preds == self.num_discovered_classes, 
                                   openmax_preds2[:, 1],
                                   openmax_preds)
        closed_maxs = torch.where(openmax_preds == self.num_discovered_classes, 
                                  openmax_top2[:, 1],
                                  openmax_max)
        openmax_predicted_label = torch.where(openmax_preds == self.num_discovered_classes, 
                                              torch.LongTensor([UNDISCOVERED_CLASS_INDEX]).to(outputs.device),
                                              openmax_preds)
        if log_thresholds:
            open_set_result_i['open_set_score'] = (openmax_outputs[:, self.num_discovered_classes]).tolist()
            open_set_result_i['closed_predicted'] = closed_preds.tolist()
            open_set_result_i['closed_argmax_prob'] = closed_maxs.tolist()
            open_set_result_i['open_predicted'] = openmax_predicted_label.tolist()
            open_set_result_i['open_argmax_prob'] = openmax_max.tolist()

        return open_set_result_i

    def _get_open_set_pred_func(self):
        """ Caveat: Open set class is represented as -1.
        """
        return lambda inputs : self.compute_open_max(inputs)



class SoftmaxOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_open_set_pred_func(self):
        def open_set_prediction(inputs):
            open_set_result_i = {}
            softmax_outputs = self.trainer_machine.get_prob_scores(inputs)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)

            open_set_result_i['open_set_score']     = (-softmax_max).tolist()
            open_set_result_i['closed_predicted']   = softmax_preds.tolist()
            open_set_result_i['closed_argmax_prob'] = softmax_max.tolist()
            open_set_result_i['open_predicted']     = softmax_preds.tolist()
            open_set_result_i['open_argmax_prob']   = softmax_max.tolist()
            # open_set_result_i['actual_loss']        += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()
            return open_set_result_i
        return open_set_prediction


class EntropyOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_open_set_pred_func(self):
        def open_set_prediction(inputs):
            open_set_result_i = {}
            softmax_outputs = self.trainer_machine.get_prob_scores(inputs)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)
            neg_entropy = softmax_outputs*softmax_outputs.log()
            neg_entropy[softmax_outputs < 1e-5] = 0
            scores = neg_entropy.sum(dim=1) # negative entropy!

            open_set_result_i['open_set_score']     = (-scores).tolist()
            open_set_result_i['closed_predicted']   = softmax_preds.tolist()
            open_set_result_i['closed_argmax_prob'] = softmax_max.tolist()
            open_set_result_i['open_predicted']     = softmax_preds.tolist()
            open_set_result_i['open_argmax_prob']   = softmax_max.tolist()
            # open_set_result_i['actual_loss']        += (torch.nn.CrossEntropyLoss(reduction='none')(outputs, label_for_learnloss)).tolist()
            return open_set_result_i
        return open_set_prediction



from query_machine import distance_matrix, cosine_distance
class NNOpen(NetworkOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_function = distance_matrix # Euclidean distance matrix

    def _get_features(self, dataloader, trainer_machine, verbose=True):
        features = torch.Tensor([]).to(self.device)
        for batch, data in enumerate(dataloader):
            inputs, _ = data
            
            with torch.no_grad():
                cur_features = trainer_machine.get_features(inputs.to(self.device))

            features = torch.cat((features, cur_features),dim=0)

        return features
    
    def _eval_open_set_helper(self, discovered_samples, discovered_classes, test_dataset, verbose=True):
        train_loader = self.trainer_machine.get_trainloader(discovered_samples, shuffle=False)
        test_loader = get_loader(test_dataset,
                                 None,
                                 shuffle=False,
                                 batch_size=self.batch,
                                 workers=self.workers)        
        test_features = self._get_features(test_loader, self.trainer_machine, verbose=verbose).cpu()
        labeled_features = self._get_features(train_loader, self.trainer_machine, verbose=verbose).cpu()
        self.min_dist = self.distance_function(test_features, labeled_features).min(dim=1)[0].cpu().tolist()
        return super()._eval_open_set_helper(discovered_samples, discovered_classes, test_dataset, verbose=verbose, do_plot=True)

    def _get_open_set_pred_func(self):
        def open_set_prediction(inputs):
            open_set_result_i = {}
            softmax_outputs = self.trainer_machine.get_prob_scores(inputs)
            softmax_max, softmax_preds = torch.max(softmax_outputs, 1)

            size_of_inputs = inputs.shape[0]
            open_set_result_i['open_set_score']     = self.min_dist[:size_of_inputs]
            self.min_dist = self.min_dist[size_of_inputs:]
            open_set_result_i['closed_predicted']   = softmax_preds.tolist()
            open_set_result_i['closed_argmax_prob'] = softmax_max.tolist()
            open_set_result_i['open_predicted']     = softmax_preds.tolist()
            open_set_result_i['open_argmax_prob']   = softmax_max.tolist()
            return open_set_result_i
        return open_set_prediction

class NNCosineOpen(NNOpen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_function = cosine_distance # Euclidean distance matrix
