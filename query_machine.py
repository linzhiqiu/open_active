import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys, random, os
import trainer_machine
from utils import get_subset_loader, get_target_unmapping_dict, SetPrintMode

def get_query_machine(query_method, dataset_info, trainer_config):
    """Return a QueryMachine object
    Args:
        query_method (str) : The querying method
        dataset_info (dataset_factory.DatasetInfo) : The details of the dataset
    """
    if query_method == 'random':
        query_machine_class = RandomQuery
    elif query_method == 'entropy':
        query_machine_class = EntropyQuery
    elif query_method == 'softmax':
        query_machine_class = SoftmaxQuery
    elif query_method == 'uldr':
        query_machine_class = ULDRQuery
    elif query_method == 'uldr_norm_cosine':
        query_machine_class = ULDRNormCosQuery
    elif query_method == 'learnloss':
        query_machine_class = LearnLossQuery
    elif query_method == 'coreset':
        query_machine_class = CoresetQuery
    elif query_method == 'coreset_norm_cosine':
        query_machine_class = CoresetNormCosQuery
    else:
        raise NotImplementedError()
    return query_machine_class(dataset_info, trainer_config)

class QueryMachine(object):
    """Abstract class for query algorithms"""

    def __init__(self, dataset_info, trainer_config):
        """Template class for all active querying methods
        Args:
            query_method (str) : Specify the active learning method
            dataset_info (dataset_factory.DatasetInfo): Dataset Information
        """
        super(QueryMachine, self).__init__()
        self.dataset_info = dataset_info
        self.batch = trainer_config.batch
        self.workers = trainer_config.workers
        self.device = trainer_config.device

    def query(self,
              trainer_machine,
              budget,
              discovered_samples,
              discovered_classes,
              query_result_path=None,
              verbose=True):
        """Perform a round of active querying

        Args:
            trainer_machine (trainer_machine.TrainerMachine): TrainerMachine instance
            budget (int): Budget to query
            discovered_samples (list[int]): All discovered (labeled) training samples
            discovered_classes (list[int]): All classes with discovered samples
            query_result_path (str, optional): Where the result will be saved
            verbose (bool, optional): Whether to print more information. Defaults to False

        Returns:
            (list[int], set[int]): (new_discovered_samples, new_discovered_classes)
                                   new_discovered_samples is newly labeled samples,
                                   new_discovered_classes is newly labeled classes (including old ones)
        """                         
        if os.path.exists(query_result_path):
            print(f"Load from pre-existing active query results from {query_result_path}")
            ckpt_dict = torch.load(query_result_path)
            return ckpt_dict['new_discovered_samples'], ckpt_dict['new_discovered_classes']
        else:
            print(f"First time performing active querying under budget {budget}.")
            with SetPrintMode(hidden=not verbose):
                new_discovered_samples, new_discovered_classes = self._query_helper(
                                                                     trainer_machine,
                                                                     budget,
                                                                     discovered_samples,
                                                                     discovered_classes,
                                                                     verbose=verbose
                                                                 )

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
        dataloader = get_subset_loader(self.dataset_info.train_dataset,
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
        unlabeled_pool = self.dataset_info.trainset_info.query_samples.difference(discovered_samples)
        unlabeled_pool = list(unlabeled_pool)
        unmapping = get_target_unmapping_dict(
            self.dataset_info.class_info.classes, discovered_classes)
        if budget == 0:
            print("Budget is 0. No querying.")
            return discovered_samples, discovered_classes
        if len(unlabeled_pool) <= budget:
            print("Remaining data is fewer/ equal than the budget constraint. Label all.")
            return list(self.dataset_info.trainset_info.query_samples), self.dataset_info.class_info.query_classes

        # Keep a rank for each sample. Higher ranks means more preferable to label.
        rankings = self._get_favorable_rankings(unlabeled_pool,
                                                trainer_machine,
                                                discovered_samples,
                                                budget=budget,
                                                verbose=verbose)
        
        rankings = list(rankings[:budget])

        new_discovered_samples = set(discovered_samples.copy())
        new_discovered_classes = discovered_classes.copy()
        for idx in rankings:
            new_sample = unlabeled_pool[int(idx)]
            new_discovered_samples.add(new_sample)
            new_discovered_classes.add(self.dataset_info.trainset_info.train_labels[new_sample])
        return list(new_discovered_samples), new_discovered_classes
            

class RandomQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(RandomQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True, budget=None):
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
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True, budget=None):
        dataloader = self._get_loader(unlabeled_pool, use_tqdm=verbose)
        favorable_scores = torch.Tensor().to(self.device)
        
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                inputs, _ = data
                prob_scores = trainer_machine.get_prob_scores(inputs.to(self.device))
                entropy = -(prob_scores*prob_scores.log()).sum(1)
                favorable_scores = torch.cat((favorable_scores, entropy))
        sorted_favorable_scores, rankings = torch.sort(favorable_scores, descending=True)
        return rankings

class SoftmaxQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(SoftmaxQuery, self).__init__(*args, **kwargs)
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True, budget=None):
        dataloader = self._get_loader(unlabeled_pool, use_tqdm=verbose)
        favorable_scores = torch.Tensor().to(self.device)
        
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                inputs, _ = data
                prob_scores = trainer_machine.get_prob_scores(inputs.to(self.device))
                prob_max, _ = torch.max(prob_scores, dim=1)
                favorable_scores = torch.cat((favorable_scores, -prob_max))
        sorted_favorable_scores, rankings = torch.sort(favorable_scores, descending=True)
        return rankings

class ULDRQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(ULDRQuery, self).__init__(*args, **kwargs)
        self.gaussian_kernel = 100.
        self.distance_matrix = distance_matrix

    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True, budget=None):
        unlabeled_features = self._get_features(unlabeled_pool, trainer_machine, verbose=verbose).cpu()
        labeled_features = self._get_features(discovered_samples, trainer_machine, verbose=verbose).cpu()
        

        ul_dist = self.distance_matrix(unlabeled_features, labeled_features).cpu()
        uu_dist = self.distance_matrix(unlabeled_features, unlabeled_features).cpu()
        print(f"The average pairwise distance between unlabeled to unlabeled is {float(((uu_dist.sum(1)-1.)/uu_dist.shape[1]).mean())}")
        print(f"The average pairwise distance between unlabeled to labeled is {float(ul_dist.mean())}")
        
        u_to_l = torch.exp(-ul_dist/(2*self.gaussian_kernel**2)).cpu()
        u_to_u = torch.exp(-uu_dist/(2*self.gaussian_kernel**2)).cpu()
        u_to_l_sum = u_to_l.sum(1)
        u_to_u_sum = u_to_u.sum(1) - 1.
        idx_uu_ul_tuples = torch.FloatTensor([[i, float(u_to_u_sum[i]), float(u_to_l_sum[i])] for i in range(u_to_l_sum.shape[0])]).cpu()
        uldr_ratios = idx_uu_ul_tuples[:,1]/idx_uu_ul_tuples[:,2]
        _, rankings = uldr_ratios.sort(descending=True)
        final_rankings = []

        if verbose:
            pbar = tqdm(range(0, budget), ncols=80)
        else:
            pbar = range(0, budget)
        
        for _, _ in enumerate(pbar):
            first_idx = idx_uu_ul_tuples[rankings[0]][0]
            final_rankings.append(first_idx)
            idx_uu_ul_tuples = idx_uu_ul_tuples[idx_uu_ul_tuples[:,0] != first_idx]
            idx_uu_ul_tuples[:, 1] = idx_uu_ul_tuples[:, 1] - u_to_u[idx_uu_ul_tuples[:, 0].long(), int(first_idx)]
            idx_uu_ul_tuples[:, 2] = idx_uu_ul_tuples[:, 2] + u_to_u[idx_uu_ul_tuples[:, 0].long(), int(first_idx)]
            
            uldr_ratios = idx_uu_ul_tuples[:,1]/idx_uu_ul_tuples[:,2]
            _, rankings = uldr_ratios.sort(descending=True)

        return final_rankings

class ULDRNormCosQuery(ULDRQuery):
    def __init__(self, *args, **kwargs):
        super(ULDRNormCosQuery, self).__init__(*args, **kwargs)
        self.distance_matrix = cosine_distance
        self.gaussian_kernel = .5

class CoresetQuery(QueryMachine):
    def __init__(self, *args, **kwargs):
        super(CoresetQuery, self).__init__(*args, **kwargs)
        self.distance_matrix = distance_matrix
    
    def _get_favorable_rankings(self, unlabeled_pool, trainer_machine, discovered_samples, verbose=True, budget=None):
        unlabeled_features = self._get_features(unlabeled_pool, trainer_machine, verbose=verbose).cpu()
        labeled_features = self._get_features(discovered_samples, trainer_machine, verbose=verbose).cpu()
        
        unlabel_to_label_dist = self.distance_matrix(unlabeled_features, labeled_features).min(dim=1)[0].cpu()
        unlabel_to_unlabel_dist = self.distance_matrix(unlabeled_features, unlabeled_features).cpu()
        
        final_rankings = []

        if verbose:
            pbar = tqdm(range(0, budget), ncols=80)
        else:
            pbar = range(0, budget)
        
        idx_score_pairs = torch.FloatTensor([[i, float(unlabel_to_label_dist[i])] for i in range(unlabel_to_label_dist.shape[0])]).cpu()
        _, rankings = idx_score_pairs[:, 1].sort(descending=True)
        for _, _ in enumerate(pbar):
            first_idx = int(idx_score_pairs[rankings[0]][0])
            final_rankings.append(first_idx)
            
            idx_score_pairs = idx_score_pairs[idx_score_pairs[:,0] != first_idx]
            
            idx_score_pairs[:, 1] = torch.min(idx_score_pairs[:, 1], unlabel_to_unlabel_dist[idx_score_pairs[:,0].long(), int(first_idx)])
            _, rankings = idx_score_pairs[:, 1].sort(descending=True)

        return final_rankings
    
class CoresetNormCosQuery(CoresetQuery):
    def __init__(self, *args, **kwargs):
        super(CoresetNormCosQuery, self).__init__(*args, **kwargs)
        self.distance_matrix = cosine_distance
    
def distance_matrix(A, B):
    # A is m x d pytorch matrix, B is n x d pytorch matrix.
    # Result is m x n pytorch matrix
    A_2 = (A**2).sum(dim=1)
    B_2 = (B**2).sum(dim=1)
    A_B_2 = A_2.view(A.shape[0], 1) + B_2
    dists = A_B_2 + -2. * torch.mm(A, B.t())
    return dists

def cosine_distance(A, B):
    # whether the A and B are normalized unit vectors
    A_normalized = torch.nn.functional.normalize(A, p=2, dim=1, eps=1e-12)
    B_normalized = torch.nn.functional.normalize(B, p=2, dim=1, eps=1e-12)
    return 1. - torch.matmul(A_normalized, B_normalized.t())