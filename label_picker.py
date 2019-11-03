import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys, random
import trainer_machine
import learning_loss
from instance_info import BasicInfoCollector, ClusterInfoCollector, LearningLossInfoCollector
from utils import get_subset_loader, get_target_unmapping_dict

class LabelPicker(object):
    """Abstract class"""
    def __init__(self, config, train_instance, trainer_machine):
        super(LabelPicker, self).__init__()
        self.round = trainer_machine.round # To keep track of the new instances added each round
        self.config = config
        self.train_instance = train_instance
        self.trainer_machine = trainer_machine # Should have a variable log to store all new instance's information
        self._generate_new_log()

        # New: After reading active learning papers
        self.active_random_sampling = self.config.active_random_sampling

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def select_new_data(self, s_train, seen_classes):
        # Require: Update the self.trainer_machine.log in each call
        raise NotImplementedError()

    def _generate_new_log(self):
        # Generate new log in self.trainer_machine
        self.trainer_machine.log = []

def least_confident(outputs):
    softmax_outputs = F.softmax(outputs, dim=1)
    score, _ = torch.max(softmax_outputs, 1)
    return score

def most_confident(outputs):
    softmax_outputs = F.softmax(outputs, dim=1)
    score, _ = torch.max(softmax_outputs, 1)
    return -score

def entropy(outputs):
    # The negative entropy
    softmax_outputs = F.softmax(outputs, dim=1)
    entropy = (softmax_outputs*softmax_outputs.log()).sum(1) # This is the negative entropy
    return entropy

def random_query(outputs):
    return torch.rand(outputs.size(0), device=outputs.device)

def highest_loss(outputs):
    _, losses = outputs
    return losses.view(losses.size(0))

def lowest_loss(outputs):
    _, losses = outputs
    return -losses.view(losses.size(0))

from global_setting import uncertainty_type_dict
class UncertaintyMeasure(LabelPicker):
    def __init__(self, *args, **kwargs):
        super(UncertaintyMeasure, self).__init__(*args, **kwargs)
        assert isinstance(self.trainer_machine, trainer_machine.Network)
        assert self.config.label_picker == 'uncertainty_measure'
        # Make sure Network (parent class) is at last branch
        if self.config.trainer in uncertainty_type_dict['osdn']:
            assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
            self.info_collector_class = BasicInfoCollector
            def openmax_measure_func(outputs):
                if self.config.trainer in uncertainty_type_dict['icalr']:
                    openmax_outputs = self.trainer_machine.compute_open_max(outputs, None)[0]
                else:
                    openmax_outputs = self.trainer_machine.compute_open_max(outputs)[0]
                score, _ = torch.max(openmax_outputs, 1)
                if self.config.uncertainty_measure == 'least_confident':
                    return score
                elif self.config.uncertainty_measure == 'most_confident':
                    return -score
                elif self.config.uncertainty_measure == 'random_query':
                    return torch.rand_like(score, device=score.device)
            self.measure_func = openmax_measure_func
        elif self.config.trainer in uncertainty_type_dict['cluster']:
            assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
            self.info_collector_class = ClusterInfoCollector
            def cluster_measure_func(outputs):
                score, _ = torch.max(outputs, 1)
                if self.config.uncertainty_measure == 'least_confident':
                    return score
                elif self.config.uncertainty_measure == 'most_confident':
                    return -score
                elif self.config.uncertainty_measure == 'random_query':
                    return torch.rand_like(score, device=score.device)
            self.measure_func = cluster_measure_func
        elif self.config.trainer in uncertainty_type_dict['learning_loss']:
            print("Using LearningLoss's uncertainty_measure")
            assert self.config.uncertainty_measure in ['highest_loss', 'lowest_loss', 'least_confident', 'most_confident', 'random_query']
            self.info_collector_class = LearningLossInfoCollector
            self.measure_func = getattr(sys.modules[__name__], self.config.uncertainty_measure)
            # raise NotImplementedError()
        elif self.config.trainer in uncertainty_type_dict['network'] or self.config.trainer in uncertainty_type_dict['sigmoid']:
            print("Using Network's uncertainty_measure")
            assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query', 'entropy']
            self.info_collector_class = BasicInfoCollector
            self.measure_func = getattr(sys.modules[__name__], self.config.uncertainty_measure)
        else:
            raise NotImplementedError()

    def get_checkpoint(self):
        # TODO: Add something meaningful
        return None

    def load_checkpoint(self, checkpoint):
        pass

    def select_new_data(self, s_train, seen_classes):
        self.model = self.trainer_machine.model
        self.unmapping = get_target_unmapping_dict(self.train_instance.classes, seen_classes)

        unlabeled_pool = self.train_instance.query_samples.difference(s_train)
        unlabeled_pool = list(unlabeled_pool)
        unseen_classes = self.train_instance.query_classes.difference(seen_classes)
        if len(unlabeled_pool) < self.config.budget:
            print("Remaining data is fewer than the budget constraint. Label all.")
            return unlabeled_pool, unseen_classes
        elif self.config.budget == 0:
            print("No label budget")
            return set(), seen_classes

        if self.active_random_sampling == 'fixed_10K':
            if len(unlabeled_pool) > 10000:
                random.shuffle(unlabeled_pool)
                unlabeled_pool = unlabeled_pool[:10000]
        elif self.active_random_sampling == "1_out_of_5":
            if len(unlabeled_pool)*0.2 > self.config.budget:
                random.shuffle(unlabeled_pool)
                unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]

        dataloader = get_subset_loader(self.train_instance.train_dataset,
                                       unlabeled_pool,
                                       None, # target transform is None,
                                       batch_size=self.config.batch,
                                       shuffle=False,
                                       workers=self.config.workers)

        info_collector = self.info_collector_class(
                             self.trainer_machine.round,
                             self.unmapping,
                             seen_classes,
                             self.measure_func
                         )
        scores, info = info_collector.gather_instance_info(
                           dataloader,
                           self.model,
                           device=self.config.device
                       )

        sorted_scores, rankings = torch.sort(scores, descending=False)
        rankings = list(rankings[:self.config.budget])

        t_train = set() # New labeled samples
        t_classes = set() # New classes (may include seen classes)
        for idx in rankings:
            new_sample = unlabeled_pool[idx]
            self.trainer_machine.log.append(info[idx])
            t_train.add(new_sample)
            t_classes.add(self.train_instance.train_labels[new_sample])
        return t_train, t_classes

class CoresetMeasure(LabelPicker):
    def __init__(self, *args, **kwargs):
        super(CoresetMeasure, self).__init__(*args, **kwargs)
        assert isinstance(self.trainer_machine, trainer_machine.Network)
        assert self.config.label_picker == 'coreset_measure'
        self.coreset_feature = self.config.coreset_feature
        # Make sure Network (parent class) is at last branch
        if isinstance(self.trainer_machine, trainer_machine.OSDNNetwork):
            import pdb; pdb.set_trace()  # breakpoint b5ff94cd //
            
        elif isinstance(self.trainer_machine, trainer_machine.ClusterNetwork):
            import pdb; pdb.set_trace()  # breakpoint 37c80b49 //
            
        elif isinstance(self.trainer_machine, learning_loss.NetworkLearningLoss):
            import pdb; pdb.set_trace()  # breakpoint 7988875e //

        elif isinstance(self.trainer_machine, trainer_machine.Network):
            print("Using Network's coreset_measure")
        else:
            raise NotImplementedError()

    def get_checkpoint(self):
        # TODO: Add something meaningful
        return None

    def load_checkpoint(self, checkpoint):
        pass

    def get_features(self, samples):
        self.model.eval()
        dataloader = get_subset_loader(self.train_instance.train_dataset,
                                       samples,
                                       None, # target transform is None,
                                       batch_size=self.config.batch,
                                       shuffle=False,
                                       workers=self.config.workers)
        pbar = dataloader


        cur_features = []
        if self.coreset_feature == 'before_fc':
            def forward_hook_func(self, inputs, outputs):
                cur_features.append(inputs[0])
        elif self.coreset_feature == 'after_fc':
            def forward_hook_func(self, inputs, outputs):
                cur_features.append(outputs)
        hook_handle = self.model.fc.register_forward_hook(forward_hook_func)
        
        features = torch.Tensor([])
        for batch, data in enumerate(pbar):
            cur_features = []
            inputs, _ = data
            
            inputs = inputs.to(self.config.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)

            features = torch.cat((features, cur_features[0].cpu()),dim=0)

        hook_handle.remove()
        return features

    def select_new_data(self, s_train, seen_classes):
        self.model = self.trainer_machine.model
        self.unmapping = get_target_unmapping_dict(self.train_instance.classes, seen_classes)

        unlabeled_pool = self.train_instance.query_samples.difference(s_train)
        unlabeled_pool = list(unlabeled_pool)
        unseen_classes = self.train_instance.query_classes.difference(seen_classes)
        if len(unlabeled_pool) < self.config.budget:
            print("Remaining data is fewer than the budget constraint. Label all.")
            return unlabeled_pool, unseen_classes
        elif self.config.budget == 0:
            print("No label budget")
            return set(), seen_classes

        if self.active_random_sampling == 'fixed_10K':
            if len(unlabeled_pool) > 10000:
                random.shuffle(unlabeled_pool)
                unlabeled_pool = unlabeled_pool[:10000]
        elif self.active_random_sampling == "1_out_of_5":
            if len(unlabeled_pool)*0.2 > self.config.budget:
                random.shuffle(unlabeled_pool)
                unlabeled_pool = unlabeled_pool[:int(len(unlabeled_pool)*.2)]

        unlabeled_features = self.get_features(unlabeled_pool).cpu()
        labeled_features = self.get_features(s_train).cpu()

        unlabel_to_label_dist = distance_matrix(unlabeled_features, labeled_features).min(dim=1)[0]
        unlabel_to_unlabel_dist = distance_matrix(unlabeled_features, unlabeled_features)


        sorted_dist, rankings = torch.sort(unlabel_to_label_dist, descending=True)
        score_ranking_pair = [(int(r), float(sorted_dist[i])) for i, r in enumerate(rankings)]
        t_train = set() # New labeled samples
        t_classes = set() # New classes (may include seen classes)
        while len(t_train) < self.config.budget:
            real_idx = score_ranking_pair[0][0]
            new_sample = unlabeled_pool[real_idx]
            t_train.add(new_sample)
            t_classes.add(self.train_instance.train_labels[new_sample])


            score_ranking_pair = score_ranking_pair[1:]
            score_ranking_pair_new = []
            for i, (r, s) in enumerate(score_ranking_pair):
                min_score = min(s, float(unlabel_to_unlabel_dist[real_idx, r]))
                score_ranking_pair_new.append((r, min_score))
            score_ranking_pair_new.sort(key=lambda x: x[1], reverse=True)
            score_ranking_pair = score_ranking_pair_new
        # sorted_scores, rankings = torch.sort(scores, descending=False)
        # rankings = list(rankings[:self.config.budget])

        
        # for idx in rankings:
        #     new_sample = unlabeled_pool[idx]
        #     self.trainer_machine.log.append(info[idx])
        #     t_train.add(new_sample)
        #     t_classes.add(self.train_instance.train_labels[new_sample])
        return t_train, t_classes

def distance_matrix(A, B):
    # A is m x d pytorch matrix, B is n x d pytorch matrix.
    # Result is m x n pytorch matrix
    A_2 = (A**2).sum(dim=1)
    B_2 = (B**2).sum(dim=1)
    A_B_2 = A_2.view(A.shape[0], 1) + B_2

    dists = A_B_2 + -2. * torch.mm(A, B.t())
    return dists

if __name__ == '__main__':
    import pdb; pdb.set_trace()  # breakpoint a576792d //
    A = torch.randn(5,3)
    B = torch.randn(2,3)
    distance_matrix(A,B)
