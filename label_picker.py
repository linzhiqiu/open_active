import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import trainer_machine
from instance_info import BasicInfoCollector
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

def random_query(outputs):
    return torch.rand(outputs.size(0), device=outputs.device)

class UncertaintyMeasure(LabelPicker):
    def __init__(self, *args, **kwargs):
        super(UncertaintyMeasure, self).__init__(*args, **kwargs)
        assert isinstance(self.trainer_machine, trainer_machine.Network)
        assert self.config.label_picker == 'uncertainty_measure'
        if isinstance(self.trainer_machine, trainer_machine.Network):
            self.measure_func = getattr(sys.modules[__name__], self.config.uncertainty_measure)
        elif isinstance(self.trainer_machine, trainer_machine.OSDNNetwork):
            assert self.config.uncertainty_measure in ['least_confident', 'most_confident', 'random_query']
            def openmax_measure_func(outputs):
                openmax_outputs = self.trainer_machine.compute_open_max(outputs)[0]
                score, _ = torch.max(openmax_outputs, 1)
                if self.config.uncertainty_measure == 'least_confident':
                    return score
                elif self.config.uncertainty_measure == 'most_confident':
                    return -score
                elif self.config.uncertainty_measure == 'random_query':
                    return torch.rand_like(score, device=score.device)
            self.measure_func = openmax_measure_func
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

        dataloader = get_subset_loader(self.train_instance.train_dataset,
                                       unlabeled_pool,
                                       None, # target transform is None,
                                       batch_size=self.config.batch,
                                       shuffle=False,
                                       workers=workers)

        info_collector = BasicInfoCollector(
                             self.trainer_machine.round,
                             self.unmapping,
                             seen_classes,
                             self.measure_func
                         )
        scores, info = info_collector.gather_instance_info(
                           dataloader,
                           model,
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


if __name__ == '__main__':
    import pdb; pdb.set_trace()  # breakpoint a576792d //

