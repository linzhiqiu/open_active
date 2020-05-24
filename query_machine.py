import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys, random, os
import trainer_machine
import learning_loss
from instance_info import BasicInfoCollector, ClusterInfoCollector, LearningLossInfoCollector
from open_info import OpenCollector
from utils import get_subset_loader, get_target_unmapping_dict, SetPrintMode

def get_query_machine(query_method, trainset_info, trainer_config):
    """Return a QueryMachine object
        Args:
            query_method (str) : The querying method
            trainset_info (TrainsetInfo) : The details of the train set
    """
    if query_method == 'random':
        query_machine_class = RandomQuery
    elif query_method == 'entropy':
        query_machine_class = EntropyQuery
    elif query_method == 'softmax':
        query_machine_class = SoftmaxQuery
    elif query_method == 'uldr':
        query_machine_class = ULDRQuery
    elif query_method == 'learnloss':
        query_machine_class = LearnLossQuery
    elif query_method == 'coreset':
        query_machine_class = CoresetQuery
    else:
        raise NotImplementedError()
    return query_machine_class(trainset_info, trainer_config)

class QueryMachine(object):
    """Abstract class for query algorithms"""
    def __init__(self, trainset_info, trainer_config):
        """Base class for active querying
            Args:
                query_method (str) : Specify the active learning method
                trainset_info (TrainsetInfo) : 
        """
        super(QueryMachine, self).__init__()
        self.trainset_info = trainset_info
        self.batch = trainer_config['batch']
        self.workers = trainer_config['workers']
        self.device = trainer_config['device']

    def query(self, trainer_machine, budget, discovered_samples, discovered_classes, query_result_path=None, verbose=True):
        """Perform querying
            Returns:
                new_discovered_samples (list of int) : New labeled samples (including old ones)
                new_discovered_classes (set of int) : New labeled classes (including old ones)
        """
        if os.path.exists(query_result_path):
            print(f"Load from pre-existing active query results from {query_result_path}")
            ckpt_dict = torch.load(query_result_path)
            return ckpt_dict['new_discovered_samples'], ckpt_dict['new_discovered_classes']
        else:
            print(f"First time performing active querying under budget {budget}.")
            with SetPrintMode(hidden=not verbose):
                new_discovered_samples, new_discovered_classes = self._query_helper(trainer_machine,
                                                                                    budget,
                                                                                    discovered_samples,
                                                                                    discovered_classes,
                                                                                    verbose=verbose)

                classes_diff = new_discovered_classes.difference(discovered_classes)
                print(f"Recognized class from {len(discovered_classes)} to {len(new_discovered_classes)}")
                assert len(set(new_discovered_samples)) ==  len(new_discovered_samples)

                assert len(set(new_discovered_samples)) == len(new_discovered_samples)
                assert len(new_discovered_samples) - len(discovered_samples) == budget

            # Save new samples and classes
            torch.save({
                'new_discovered_samples' : new_discovered_samples,
                'new_discovered_classes' : new_discovered_classes,
            }, query_result_path)    
            return new_discovered_samples, new_discovered_classes

    def _get_favorable_rankings(self, trainer_machine, inputs):
        """Returns a 1-D tensor specifying the query favorability score for inputs
            Args:
                trainer_machine (TrainerMachine)
                inputs (A batch of images)
        """
        raise NotImplementedError()
    
    def _get_loader(self, samples, use_tqdm=True):
        dataloader = get_subset_loader(self.trainset_info.train_dataset,
                                        samples,
                                        None, # target transform is None,
                                        batch_size=self.batch,
                                        shuffle=False, # Very important!
                                        workers=self.workers)
        if use_tqdm: pbar = tqdm(dataloader, ncols=80)
        else: pbar = dataloader
        return pbar
    
    def _get_features(self, samples, trainer_machine, verbose=True):
        dataloader = self._get_loader(samples, use_tqdm=verbose)

        features = torch.Tensor([]).to(self.device)
        for batch, data in enumerate(dataloader):
            inputs, _ = data
            
            with torch.no_grad():
                cur_features = trainer_machine.get_features(inputs.to(self.device))

            features = torch.cat((features, cur_features),dim=0)

        return features

    def _query_helper(self, trainer_machine, budget, discovered_samples, discovered_classes, verbose=True):
        """Return new labeled samples from unlabeled_pool.
            Returns:
                new_discovered_samples (list of int): New labeled samples
                new_discovered_classes (set of int): New labeled classes
            Args:
                trainer_machine (TrainerMachine) : Must have backbone and classifier
                budget (int)
                discovered_samples (list)
                discovered_classes (set)
                verbose (bool)
        """
        unlabeled_pool = self.trainset_info.query_samples.difference(discovered_samples)
        unlabeled_pool = list(unlabeled_pool)
        unmapping = get_target_unmapping_dict(self.trainset_info.classes, discovered_classes)
        if len(unlabeled_pool) < budget:
            print("Remaining data is fewer than the budget constraint. Label all.")
            return list(self.trainset_info.query_samples), self.trainset_info.query_classes

        # Keep a rank for each sample. Higher ranks means more preferable to label.
        rankings = self._get_favorable_rankings(unlabeled_pool,
                                                trainer_machine,
                                                discovered_samples,
                                                verbose=verbose)
        
        rankings = list(rankings[:budget])

        new_discovered_samples = set(discovered_samples.copy())
        new_discovered_classes = discovered_classes.copy()
        for idx in rankings:
            new_sample = unlabeled_pool[int(idx)]
            new_discovered_samples.add(new_sample)
            new_discovered_classes.add(self.trainset_info.train_labels[new_sample])
        return list(new_discovered_samples), new_discovered_classes
            

class RandomQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(RandomQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose):
        dataloader = self._get_loader(unlabeled_pool, use_tqdm=verbose)

        favorable_scores = torch.Tensor().to(self.device)
        
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                inputs, _ = data
                
                scores = torch.rand(inputs.shape[0]).to(self.device)
                favorable_scores = torch.cat((favorable_scores, scores))
        sorted_favorable_scores, rankings = torch.sort(favorable_scores, descending=True)
        return rankings

class EntropyQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(EntropyQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True):
        dataloader = self._get_loader(unlabeled_pool, use_tqdm=verbose)
        favorable_scores = torch.Tensor().to(self.device)
        
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                inputs, _ = data
                class_scores = trainer_machine.get_class_scores(inputs.to(self.device))
                prob_scores = F.softmax(class_scores, dim=1)
                entropy = -(prob_scores*prob_scores.log()).sum(1)
                favorable_scores = torch.cat((favorable_scores, entropy))
        sorted_favorable_scores, rankings = torch.sort(favorable_scores, descending=True)
        return rankings

class SoftmaxQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(SoftmaxQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True):
        dataloader = self._get_loader(unlabeled_pool, use_tqdm=verbose)
        favorable_scores = torch.Tensor().to(self.device)
        
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                inputs, _ = data
                class_scores = trainer_machine.get_class_scores(inputs.to(self.device))
                prob_scores = F.softmax(class_scores, dim=1)
                prob_max, _ = torch.max(prob_scores, dim=1)
                favorable_scores = torch.cat((favorable_scores, -prob_max))
        sorted_favorable_scores, rankings = torch.sort(favorable_scores, descending=True)
        return rankings

class ULDRQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(ULDRQuery, self).__init__(*args, **kwargs)
        self.gaussian_kernel = 100

    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True):
        unlabeled_features = self._get_features(unlabeled_pool, trainer_machine, verbose=verbose).cpu()
        labeled_features = self._get_features(discovered_samples, trainer_machine, verbose=verbose).cpu()
        
        u_to_l = torch.exp(-distance_matrix(unlabeled_features, labeled_features)/(2*self.gaussian_kernel**2)).cpu()
        u_to_u = torch.exp(-distance_matrix(unlabeled_features, unlabeled_features)/(2*self.gaussian_kernel**2)).cpu()
        u_to_l_sum = u_to_l.sum(1)
        u_to_u_sum = u_to_u.sum(1) - 1.
        
        idx_uu_ul_tuples = torch.FloatTensor([[i, float(u_to_u_sum[i]), float(u_to_l_sum[i])] for i in range(u_to_l_sum.shape[0])]).cpu()
        uldr_ratios = idx_uu_ul_tuples[:,1]/idx_uu_ul_tuples[:,2]
        _, rankings = uldr_ratios.sort(descending=True)
        final_rankings = []

        if verbose:
            pbar = tqdm(range(0, len(idx_uu_ul_tuples)), ncols=80)
        else:
            pbar = range(0, len(idx_uu_ul_tuples))
        
        for _, _ in enumerate(pbar):
            first_idx = idx_uu_ul_tuples[rankings[0]][0]
            final_rankings.append(first_idx)
            idx_uu_ul_tuples = idx_uu_ul_tuples[idx_uu_ul_tuples[:,0] != first_idx]
            idx_uu_ul_tuples[:, 1] = idx_uu_ul_tuples[:, 1] - u_to_u[idx_uu_ul_tuples[:, 0].long(), int(first_idx)]
            idx_uu_ul_tuples[:, 2] = idx_uu_ul_tuples[:, 2] + u_to_u[idx_uu_ul_tuples[:, 0].long(), int(first_idx)]
            
            uldr_ratios = idx_uu_ul_tuples[:,1]/idx_uu_ul_tuples[:,2]
            _, rankings = uldr_ratios.sort(descending=True)

        return final_rankings

class LearnLossQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(LearnLossQuery, self).__init__(*args, **kwargs)
        raise NotImplementedError()
        # if self.active_random_sampling == 'fixed_10K':
        #     if len(unlabeled_pool) > 10000:
        #         random.shuffle(unlabeled_pool)
        #         unlabeled_pool = unlabeled_pool[:10000]
        # elif self.active_random_sampling == "1_out_of_5":
        #     if len(unlabeled_pool)*0.2 > self.config.budget:
        #         random.shuffle(unlabeled_pool)
        #         unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]

class CoresetQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(CoresetQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True):
        unlabeled_features = self._get_features(unlabeled_pool, trainer_machine, verbose=verbose).cpu()
        labeled_features = self._get_features(discovered_samples, trainer_machine, verbose=verbose).cpu()
        
        unlabel_to_label_dist = distance_matrix(unlabeled_features, labeled_features).min(dim=1)[0].cpu()
        unlabel_to_unlabel_dist = distance_matrix(unlabeled_features, unlabeled_features).cpu()
        
        
        sorted_min_dist_to_labeled, rankings = torch.sort(unlabel_to_label_dist, descending=True)
        idx_score_pairs = [(int(idx), float(sorted_min_dist_to_labeled[i])) for i, idx in enumerate(rankings)]
        final_rankings = []

        if verbose:
            pbar = tqdm(range(0, len(idx_score_pairs)), ncols=80)
        else:
            pbar = range(0, len(idx_score_pairs))
        for _, _ in enumerate(pbar):
            first_idx = idx_score_pairs[0][0]
            final_rankings.append(first_idx)

            idx_score_pairs = idx_score_pairs[1:]
            new_idx_score_pairs = []
            for i, (idx, min_dist_to_old_labeled) in enumerate(idx_score_pairs):
                min_score = min(min_dist_to_old_labeled,
                                float(unlabel_to_unlabel_dist[first_idx, idx]))
                new_idx_score_pairs.append((idx, min_score))
            new_idx_score_pairs.sort(key=lambda x: x[1], reverse=True)
            idx_score_pairs = new_idx_score_pairs

        return final_rankings
    

# class CoresetMeasure(LabelPicker):
#     def __init__(self, *args, **kwargs):
#         super(CoresetMeasure, self).__init__(*args, **kwargs)
#         assert isinstance(self.trainer_machine, trainer_machine.Network)
#         assert self.config.label_picker == 'coreset_measure'
#         self.coreset_feature = self.config.coreset_feature


#         # Make sure Network (parent class) is at last branch
#         if isinstance(self.trainer_machine, trainer_machine.OSDNNetwork):
#             pass            
#         elif isinstance(self.trainer_machine, trainer_machine.ClusterNetwork):
#             import pdb; pdb.set_trace()  # breakpoint 37c80b49 //
            
#         elif isinstance(self.trainer_machine, learning_loss.NetworkLearningLoss):
#             import pdb; pdb.set_trace()  # breakpoint 7988875e //
#         elif isinstance(self.trainer_machine, trainer_machine.Network):
#             print("Using Network's coreset_measure")
#         else:
#             raise NotImplementedError()

#     def get_features(self, samples):
#         self.model.eval()
#         dataloader = get_subset_loader(self.train_instance.train_dataset,
#                                        samples,
#                                        None, # target transform is None,
#                                        batch_size=self.config.batch,
#                                        shuffle=False,
#                                        workers=self.config.workers)
#         pbar = dataloader


#         cur_features = []
#         if self.coreset_feature == 'before_fc':
#             def forward_hook_func(self, inputs, outputs):
#                 cur_features.append(inputs[0])
#         elif self.coreset_feature == 'after_fc':
#             def forward_hook_func(self, inputs, outputs):
#                 cur_features.append(outputs)
#         hook_handle = self.model.fc.register_forward_hook(forward_hook_func)
        
#         features = torch.Tensor([])
#         for batch, data in enumerate(pbar):
#             cur_features = []
#             inputs, _ = data
            
#             inputs = inputs.to(self.config.device)

#             with torch.set_grad_enabled(False):
#                 outputs = self.model(inputs)

#             features = torch.cat((features, cur_features[0].cpu()),dim=0)

#         hook_handle.remove()
#         return features, dataloader

#     def select_new_data(self, discovered_samples, discovered_classes):
#         self.model = self.trainer_machine.model
#         self.unmapping = get_target_unmapping_dict(self.train_instance.classes, discovered_classes)

#         unlabeled_pool = self.train_instance.query_samples.difference(discovered_samples)
#         unlabeled_pool = list(unlabeled_pool)
#         undiscovered_classes = self.train_instance.query_classes.difference(discovered_classes)
#         if len(unlabeled_pool) < self.config.budget:
#             print("Remaining data is fewer than the budget constraint. Label all.")
#             return unlabeled_pool, undiscovered_classes
#         elif self.config.budget == 0:
#             print("No label budget")
#             return set(), discovered_classes

#         if self.active_random_sampling == 'fixed_10K':
#             if len(unlabeled_pool) > 10000:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:10000]
#         elif self.active_random_sampling == "1_out_of_5":
#             if len(unlabeled_pool)*0.2 > self.config.budget:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]



#         unlabeled_features, unlabeled_dataloader = self.get_features(unlabeled_pool)
#         unlabeled_features = unlabeled_features.cpu()
#         labeled_features, _ = self.get_features(discovered_samples)
#         labeled_features = labeled_features.cpu()

#         unlabel_to_label_dist = distance_matrix(unlabeled_features, labeled_features).min(dim=1)[0]
#         unlabel_to_unlabel_dist = distance_matrix(unlabeled_features, unlabeled_features)

#         # Normalize the distances to 0-1 by dividing max

#         dist_max = torch.max(unlabel_to_label_dist.max(), unlabel_to_unlabel_dist.max())
#         unlabel_to_label_dist = unlabel_to_label_dist/dist_max
#         unlabel_to_unlabel_dist = unlabel_to_unlabel_dist/dist_max

#         if self.open_active_setup in ['half', 'hr100', 'hr200', 'hr300', 'hr400', 'open']:
#             open_collector = OpenCollector(self.trainer_machine)
#             open_scores = open_collector.gather_open_info(
#                               unlabeled_dataloader,
#                               device=self.config.device
#                           )
#             open_scores = open_scores - open_scores.min()
#             open_scores = open_scores / open_scores.max()
#         else:
#             open_scores = None

#         sorted_dist, rankings = torch.sort(unlabel_to_label_dist, descending=True)
#         if self.open_active_setup in ['half', 'hr100', 'hr200', 'hr300', 'hr400', 'open']:
#             score_ranking_pair = [(int(r), float(sorted_dist[i]), float(open_scores[r])) for i, r in enumerate(rankings)]
#             if self.open_active_setup == 'open':
#                 score_ranking_pair.sort(key=lambda x: x[2], reverse=True)
#             elif self.open_active_setup in ['hr100', 'hr200', 'hr300',] and self.trainer_machine.round < int(self.open_active_setup[2:]):
#                 random.shuffle(score_ranking_pair)
#             else:
#                 score_ranking_pair.sort(key=lambda x: x[1]+x[2], reverse=True)
#         else:
#             score_ranking_pair = [(int(r), float(sorted_dist[i]), 0) for i, r in enumerate(rankings)]
        
#         new_samples = set() # New labeled samples
#         new_classes = set() # New classes (may include seen classes)
#         while len(new_samples) < self.config.budget:
#             real_idx = score_ranking_pair[0][0]
#             new_sample = unlabeled_pool[real_idx]
#             new_samples.add(new_sample)
#             new_classes.add(self.train_instance.train_labels[new_sample])

#             score_ranking_pair = score_ranking_pair[1:]
#             score_ranking_pair_new = []
#             for i, (r, s, o) in enumerate(score_ranking_pair):
#                 min_score = min(s, float(unlabel_to_unlabel_dist[real_idx, r]))
#                 score_ranking_pair_new.append((r, min_score, o))
#             if self.open_active_setup in ['half'] or (self.open_active_setup in ['hr100','hr200','hr300', 'hr400'] and self.trainer_machine.round >= int(self.open_active_setup[2:])):
#                 score_ranking_pair_new.sort(key=lambda x: x[1]+x[2], reverse=True)
#             elif self.open_active_setup in ['active'] or (self.open_active_setup in ['ar100','ar200','ar300'] and self.trainer_machine.round >= int(self.open_active_setup[2:])):
#                 score_ranking_pair_new.sort(key=lambda x: x[1], reverse=True)
#             elif self.open_active_setup in ['open']:
#                 score_ranking_pair_new.sort(key=lambda x: x[2], reverse=True)
#             else:
#                 pass # Keep it random shuffled
#             score_ranking_pair = score_ranking_pair_new
#         return new_samples, new_classes
    
#     def get_logging_str(self, verbose=False):
#         logging_strs = []
#         if verbose:
#             logging_strs += ["coreset"]
#             logging_strs += [self.config.coreset_measure,
#                              self.config.active_random_sampling,
#                              self.config.coreset_feature]
#             logging_strs += ['oa', self.config.open_active_setup]
#         else:
#             raise NotImplementedError()
#         return "_".join(logging_strs)

# def distance_matrix(A, B):
#     # A is m x d pytorch matrix, B is n x d pytorch matrix.
#     # Result is m x n pytorch matrix
#     A_2 = (A**2).sum(dim=1)
#     B_2 = (B**2).sum(dim=1)
#     A_B_2 = A_2.view(A.shape[0], 1) + B_2

#     dists = A_B_2 + -2. * torch.mm(A, B.t())
#     return dists

# def highest_loss(outputs):
#     _, losses = outputs
#     return losses.view(losses.size(0))

# def lowest_loss(outputs):
#     _, losses = outputs
#     return -losses.view(losses.size(0))

# class UncertaintyMeasure(LabelPicker):
#     def __init__(self, *args, **kwargs):
#         super(UncertaintyMeasure, self).__init__(*args, **kwargs)
#         assert isinstance(self.trainer_machine, trainer_machine.Network)
#         assert self.config.label_picker == 'uncertainty_measure'

#         # Make sure Network (parent class) is at last branch
#         # Each branch must consider the following branches
#         if self.config.trainer in uncertainty_type_dict['learning_loss']:
#             print("Using LearningLoss's uncertainty_measure")
#             assert self.config.uncertainty_measure in ['highest_loss', 'lowest_loss', 'random_query']
#             self.info_collector_class = LearningLossInfoCollector
#             def learning_loss_measure_func(outputs):
#                 network_outputs, losses = outputs
#                 if self.config.uncertainty_measure == 'highest_loss': 
#                     return -losses.view(losses.size(0)) # Higher the loss, more uncertain score
#                 elif self.config.uncertainty_measure == 'lowest_loss':
#                     return losses.view(losses.size(0))
#                 elif self.config.uncertainty_measure == 'random_query':
#                     return torch.rand_like(losses.view(losses.size(0)))
#             self.measure_func = learning_loss_measure_func
#         elif self.config.trainer in uncertainty_type_dict['osdn']:
#             assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
#             self.info_collector_class = BasicInfoCollector            
#             def openmax_measure_func(outputs):
#                 if self.config.trainer in uncertainty_type_dict['icalr']:
#                     openmax_outputs = self.trainer_machine.compute_open_max(outputs, None, log_thresholds=False)[0]
#                 else:
#                     openmax_outputs = self.trainer_machine.compute_open_max(outputs, log_thresholds=False)[0]
#                 score, _ = torch.max(openmax_outputs, 1)
#                 if self.config.uncertainty_measure == 'least_confident':
#                     return score
#                 elif self.config.uncertainty_measure == 'most_confident':
#                     return -score
#                 elif self.config.uncertainty_measure == 'random_query':
#                     return torch.rand_like(score, device=score.device)
#             self.measure_func = openmax_measure_func
#         elif self.config.trainer in uncertainty_type_dict['cluster']:
#             assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
#             self.info_collector_class = ClusterInfoCollector
#             def cluster_measure_func(outputs):
#                 score, _ = torch.max(outputs, 1)
#                 if self.config.uncertainty_measure == 'least_confident':
#                     return score
#                 elif self.config.uncertainty_measure == 'most_confident':
#                     return -score
#                 elif self.config.uncertainty_measure == 'random_query':
#                     return torch.rand_like(score, device=score.device)
#             self.measure_func = cluster_measure_func
#         elif self.config.trainer in uncertainty_type_dict['network'] or self.config.trainer in uncertainty_type_dict['sigmoid']:
#             print("Using Network's uncertainty_measure")
#             assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
#             self.info_collector_class = BasicInfoCollector
#             def network_measure_func(outputs):
#                 softmax_outputs = F.softmax(outputs, dim=1)
#                 score, _ = torch.max(softmax_outputs, 1)
#                 if self.config.uncertainty_measure == 'least_confident':
#                     return score
#                 elif self.config.uncertainty_measure == 'most_confident':
#                     return -score
#                 elif self.config.uncertainty_measure == 'entropy':
#                     entropy = (softmax_outputs*softmax_outputs.log()).sum(1) # This is the negative entropy
#                     return entropy
#                 elif self.config.uncertainty_measure == 'random_query':
#                     return torch.rand_like(score, device=score.device)
#             self.measure_func = network_measure_func
#         else:
#             raise NotImplementedError()

#     def select_new_data(self, discovered_samples, discovered_classes):
#         self.model = self.trainer_machine.model
#         self.unmapping = get_target_unmapping_dict(self.train_instance.classes, discovered_classes)

#         unlabeled_pool = self.train_instance.query_samples.difference(discovered_samples)
#         unlabeled_pool = list(unlabeled_pool)
#         undiscovered_classes = self.train_instance.query_classes.difference(discovered_classes)
#         if len(unlabeled_pool) < self.config.budget:
#             print("Remaining data is fewer than the budget constraint. Label all.")
#             return unlabeled_pool, undiscovered_classes
#         elif self.config.budget == 0:
#             print("No label budget")
#             return set(), discovered_classes

#         if self.active_random_sampling == 'fixed_10K':
#             if len(unlabeled_pool) > 10000:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:10000]
#         elif self.active_random_sampling == "1_out_of_5":
#             if len(unlabeled_pool)*0.2 > self.config.budget:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]

#         dataloader = get_subset_loader(self.train_instance.train_dataset,
#                                        unlabeled_pool,
#                                        None, # target transform is None,
#                                        batch_size=self.config.batch,
#                                        shuffle=False, # Very important!
#                                        workers=self.config.workers)

#         info_collector = self.info_collector_class(
#                              self.trainer_machine.round,
#                              self.unmapping,
#                              discovered_classes,
#                              self.measure_func
#                          )
#         active_scores, info = info_collector.gather_instance_info(
#                                   dataloader,
#                                   self.model,
#                                   device=self.config.device
#                               )
#         active_scores = active_scores - active_scores.min()
#         active_scores = active_scores / active_scores.max()

#         open_collector = OpenCollector(self.trainer_machine)

#         if self.open_active_setup in ['half', 'open', 'hr100', 'hr200', 'hr300', 'hr400']:
#             open_scores = open_collector.gather_open_info(
#                               dataloader,
#                               device=self.config.device
#                           )
#             open_scores = open_scores - open_scores.min()
#             open_scores = open_scores / open_scores.max()

#         if self.open_active_setup == 'active':
#             scores = active_scores
#         elif self.open_active_setup == 'open':
#             scores = -open_scores
#         elif self.open_active_setup == 'half':
#             scores = active_scores-open_scores
#         elif self.open_active_setup in ['hr100', 'hr200', 'hr300', 'hr400']:
#             if self.trainer_machine.round < int(self.open_active_setup[2:]):
#                 print(f"Smaller than {int(self.open_active_setup[2:])}. Use random query.")
#                 scores = torch.rand_like(active_scores)
#             else:
#                 scores = active_scores-open_scores
#         elif self.open_active_setup in ['ar100', 'ar200', 'ar300']:
#             if self.trainer_machine.round < int(self.open_active_setup[2:]):
#                 scores = torch.rand_like(active_scores)
#             else:
#                 scores = active_scores

#         sorted_scores, rankings = torch.sort(scores, descending=False)
#         rankings = list(rankings[:self.config.budget])

#         new_samples = set() # New labeled samples
#         new_classes = set() # New classes (may include seen classes)
#         for idx in rankings:
#             new_sample = unlabeled_pool[idx]
#             self.trainer_machine.log.append(info[idx])
#             new_samples.add(new_sample)
#             new_classes.add(self.train_instance.train_labels[new_sample])
#         return new_samples, new_classes

# class CoresetMeasure(LabelPicker):
#     def __init__(self, *args, **kwargs):
#         super(CoresetMeasure, self).__init__(*args, **kwargs)
#         assert isinstance(self.trainer_machine, trainer_machine.Network)
#         assert self.config.label_picker == 'coreset_measure'
#         self.coreset_feature = self.config.coreset_feature


#         # Make sure Network (parent class) is at last branch
#         if isinstance(self.trainer_machine, trainer_machine.OSDNNetwork):
#             pass            
#         elif isinstance(self.trainer_machine, trainer_machine.ClusterNetwork):
#             import pdb; pdb.set_trace()  # breakpoint 37c80b49 //
            
#         elif isinstance(self.trainer_machine, learning_loss.NetworkLearningLoss):
#             import pdb; pdb.set_trace()  # breakpoint 7988875e //
#         elif isinstance(self.trainer_machine, trainer_machine.Network):
#             print("Using Network's coreset_measure")
#         else:
#             raise NotImplementedError()

#     def get_features(self, samples):
#         self.model.eval()
#         dataloader = get_subset_loader(self.train_instance.train_dataset,
#                                        samples,
#                                        None, # target transform is None,
#                                        batch_size=self.config.batch,
#                                        shuffle=False,
#                                        workers=self.config.workers)
#         pbar = dataloader


#         cur_features = []
#         if self.coreset_feature == 'before_fc':
#             def forward_hook_func(self, inputs, outputs):
#                 cur_features.append(inputs[0])
#         elif self.coreset_feature == 'after_fc':
#             def forward_hook_func(self, inputs, outputs):
#                 cur_features.append(outputs)
#         hook_handle = self.model.fc.register_forward_hook(forward_hook_func)
        
#         features = torch.Tensor([])
#         for batch, data in enumerate(pbar):
#             cur_features = []
#             inputs, _ = data
            
#             inputs = inputs.to(self.config.device)

#             with torch.set_grad_enabled(False):
#                 outputs = self.model(inputs)

#             features = torch.cat((features, cur_features[0].cpu()),dim=0)

#         hook_handle.remove()
#         return features, dataloader

#     def select_new_data(self, discovered_samples, discovered_classes):
#         self.model = self.trainer_machine.model
#         self.unmapping = get_target_unmapping_dict(self.train_instance.classes, discovered_classes)

#         unlabeled_pool = self.train_instance.query_samples.difference(discovered_samples)
#         unlabeled_pool = list(unlabeled_pool)
#         undiscovered_classes = self.train_instance.query_classes.difference(discovered_classes)
#         if len(unlabeled_pool) < self.config.budget:
#             print("Remaining data is fewer than the budget constraint. Label all.")
#             return unlabeled_pool, undiscovered_classes
#         elif self.config.budget == 0:
#             print("No label budget")
#             return set(), discovered_classes

#         if self.active_random_sampling == 'fixed_10K':
#             if len(unlabeled_pool) > 10000:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:10000]
#         elif self.active_random_sampling == "1_out_of_5":
#             if len(unlabeled_pool)*0.2 > self.config.budget:
#                 random.shuffle(unlabeled_pool)
#                 unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]



#         unlabeled_features, unlabeled_dataloader = self.get_features(unlabeled_pool)
#         unlabeled_features = unlabeled_features.cpu()
#         labeled_features, _ = self.get_features(discovered_samples)
#         labeled_features = labeled_features.cpu()

#         unlabel_to_label_dist = distance_matrix(unlabeled_features, labeled_features).min(dim=1)[0]
#         unlabel_to_unlabel_dist = distance_matrix(unlabeled_features, unlabeled_features)

#         # Normalize the distances to 0-1 by dividing max

#         dist_max = torch.max(unlabel_to_label_dist.max(), unlabel_to_unlabel_dist.max())
#         unlabel_to_label_dist = unlabel_to_label_dist/dist_max
#         unlabel_to_unlabel_dist = unlabel_to_unlabel_dist/dist_max

#         if self.open_active_setup in ['half', 'hr100', 'hr200', 'hr300', 'hr400', 'open']:
#             open_collector = OpenCollector(self.trainer_machine)
#             open_scores = open_collector.gather_open_info(
#                               unlabeled_dataloader,
#                               device=self.config.device
#                           )
#             open_scores = open_scores - open_scores.min()
#             open_scores = open_scores / open_scores.max()
#         else:
#             open_scores = None

#         sorted_dist, rankings = torch.sort(unlabel_to_label_dist, descending=True)
#         if self.open_active_setup in ['half', 'hr100', 'hr200', 'hr300', 'hr400', 'open']:
#             score_ranking_pair = [(int(r), float(sorted_dist[i]), float(open_scores[r])) for i, r in enumerate(rankings)]
#             if self.open_active_setup == 'open':
#                 score_ranking_pair.sort(key=lambda x: x[2], reverse=True)
#             elif self.open_active_setup in ['hr100', 'hr200', 'hr300',] and self.trainer_machine.round < int(self.open_active_setup[2:]):
#                 random.shuffle(score_ranking_pair)
#             else:
#                 score_ranking_pair.sort(key=lambda x: x[1]+x[2], reverse=True)
#         else:
#             score_ranking_pair = [(int(r), float(sorted_dist[i]), 0) for i, r in enumerate(rankings)]
        
#         new_samples = set() # New labeled samples
#         new_classes = set() # New classes (may include seen classes)
#         while len(new_samples) < self.config.budget:
#             real_idx = score_ranking_pair[0][0]
#             new_sample = unlabeled_pool[real_idx]
#             new_samples.add(new_sample)
#             new_classes.add(self.train_instance.train_labels[new_sample])

#             score_ranking_pair = score_ranking_pair[1:]
#             score_ranking_pair_new = []
#             for i, (r, s, o) in enumerate(score_ranking_pair):
#                 min_score = min(s, float(unlabel_to_unlabel_dist[real_idx, r]))
#                 score_ranking_pair_new.append((r, min_score, o))
#             if self.open_active_setup in ['half'] or (self.open_active_setup in ['hr100','hr200','hr300', 'hr400'] and self.trainer_machine.round >= int(self.open_active_setup[2:])):
#                 score_ranking_pair_new.sort(key=lambda x: x[1]+x[2], reverse=True)
#             elif self.open_active_setup in ['active'] or (self.open_active_setup in ['ar100','ar200','ar300'] and self.trainer_machine.round >= int(self.open_active_setup[2:])):
#                 score_ranking_pair_new.sort(key=lambda x: x[1], reverse=True)
#             elif self.open_active_setup in ['open']:
#                 score_ranking_pair_new.sort(key=lambda x: x[2], reverse=True)
#             else:
#                 pass # Keep it random shuffled
#             score_ranking_pair = score_ranking_pair_new
#         return new_samples, new_classes
    
#     def get_logging_str(self, verbose=False):
#         logging_strs = []
#         if verbose:
#             logging_strs += ["coreset"]
#             logging_strs += [self.config.coreset_measure,
#                              self.config.active_random_sampling,
#                              self.config.coreset_feature]
#             logging_strs += ['oa', self.config.open_active_setup]
#         else:
#             raise NotImplementedError()
#         return "_".join(logging_strs)

def distance_matrix(A, B):
    # A is m x d pytorch matrix, B is n x d pytorch matrix.
    # Result is m x n pytorch matrix
    A_2 = (A**2).sum(dim=1)
    B_2 = (B**2).sum(dim=1)
    A_B_2 = A_2.view(A.shape[0], 1) + B_2
    dists = A_B_2 + -2. * torch.mm(A, B.t())
    return dists

# if __name__ == '__main__':
#     import pdb; pdb.set_trace()  # breakpoint a576792d //
#     A = torch.randn(5,3)
#     B = torch.randn(2,3)
#     distance_matrix(A,B)
