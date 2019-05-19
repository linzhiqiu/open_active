import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import trainer_machine
from utils import get_subset_dataloaders

def get_target_unmapping_dict(classes, seen_classes):
    """ Return a dictionary that map 0-len(seen_classes) to true seen_classes indices.
        Always return the same indices as long as seen classes (which is a set) is the same.
        Args:
            classes: The list of all classes
            seen_classes: The set of all seen classes
    """
    seen_classes = list(seen_classes)
    mapping = {idx : -1 if idx not in seen_classes else seen_classes.index(idx)
               for idx in classes}
    unmapping = {mapping[true_index] : true_index for true_index in mapping.keys()}
    if -1 in unmapping.keys():
        del unmapping[-1]
    return unmapping

class InstanceInfo(object):
    def __init__(self):
        super(InstanceInfo, self).__init__()

class BasicInstanceInfo(InstanceInfo):
    """ Store the most basic information of a new instance to be added
    """
    def __init__(self, round_index, true_label, predicted_label, softmax_score, seen):
        super(BasicInstanceInfo, self).__init__()
        self.round_index = round_index
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.softmax_score = softmax_score
        self.seen = seen # -1 if unseen, 1 if seen

class LabelPicker(object):
    """Abstract class"""
    def __init__(self, config, train_instance, trainer_machine):
        super(LabelPicker, self).__init__()
        self.round = 0 # To keep track of the new instances added each round
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
        self.round += 1
        self.model = self.trainer_machine.model
        self.unmapping = get_target_unmapping_dict(self.train_instance.classes, seen_classes)

        unlabeled_pool = self.train_instance.train_samples.difference(s_train)
        unlabeled_pool = list(unlabeled_pool)
        unseen_classes = self.train_instance.classes.difference(seen_classes)
        if len(unlabeled_pool) < self.config.budget:
            print("Remaining data is fewer than the budget constraint. Label all.")
            return unlabeled_pool, self.trainer_machine.classes

        dataloader = get_subset_dataloaders(self.train_instance.train_dataset,
                                            unlabeled_pool,
                                            None, # target transform is None
                                            self.config.batch,
                                            workers=self.config.workers,
                                            shuffle=False)['train']
        pbar = tqdm(dataloader, ncols=80)
        # Score each examples in the unlabeled pool
        scores = torch.Tensor().to(self.config.device)
        info = []
        with torch.no_grad():
            for batch, data in enumerate(pbar):
                inputs, labels = data
                
                inputs = inputs.to(self.config.device)

                outputs = self.model(inputs)
                scores_batch_i = self.measure_func(outputs)
                # batch_instances_info is a list of information for every example in this batch.
                batch_instances_info = self._get_batch_instances_info(outputs, labels, seen_classes)
                info += batch_instances_info

                scores = torch.cat((scores,scores_batch_i))

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

    def _get_batch_instances_info(self, outputs, labels, seen_classes):
        # Return a list with length == outputs.size(0). Each element is the information of that specific example.
        # Each element is represented by (round_index, true_label, predicted_label, softmax_score, -1 if in unseen class else 1)
        batch_instances_info = []
        softmax_outputs = F.softmax(outputs, dim=1)
        prob_scores, predicted_labels = torch.max(softmax_outputs, 1)
        for i in range(outputs.size(0)):
            prob_score_i = float(prob_scores[i])
            predicted_label_i = int(self.unmapping[int(predicted_labels[i])])
            label_i = int(labels[i])
            instance_info = BasicInstanceInfo(self.round, label_i, predicted_label_i, prob_score_i, label_i in seen_classes)
            batch_instances_info.append(instance_info)
        return batch_instances_info


if __name__ == '__main__':
    import pdb; pdb.set_trace()  # breakpoint a576792d //

