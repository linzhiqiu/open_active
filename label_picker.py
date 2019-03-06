import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import trainer_machine
from utils import get_subset_dataloaders

class LabelPicker(object):
    """Abstract class"""
    def __init__(self, config, train_instance, trainer_machine):
        super(LabelPicker, self).__init__()
        self.config = config
        self.train_instance = train_instance
        self.trainer_machine = trainer_machine

    def get_checkpoint(self):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def select_new_data(self, s_train, seen_classes):
        raise NotImplementedError()



def least_confident(outputs):
    softmax_outputs = F.softmax(outputs, dim=1)
    score, _ = torch.max(softmax_outputs, 1)
    return score

def most_confident(outputs):
    softmax_outputs = F.softmax(outputs, dim=1)
    score, _ = torch.max(softmax_outputs, 1)
    return -score

class UncertaintyMeasure(LabelPicker):
    def __init__(self, *args, **kwargs):
        super(UncertaintyMeasure, self).__init__(*args, **kwargs)
        assert self.config.label_picker == 'uncertainty_measure'
        self.measure_func = getattr(sys.modules[__name__], self.config.uncertainty_measure)

    def get_checkpoint(self):
        # TODO: Add something meaningful
        return None

    def load_checkpoint(self, checkpoint):
        pass

    def select_new_data(self, s_train, seen_classes):
        if isinstance(self.trainer_machine, trainer_machine.Network):
            self.model = self.trainer_machine.model

            unlabeled_pool = self.train_instance.train_samples.difference(s_train)
            unlabeled_pool = list(unlabeled_pool)
            unseen_classes = self.train_instance.classes.difference(seen_classes)
            if len(unlabeled_pool) < self.config.budget:
                print("Remaining data is fewer than the budget constraint. Label all.")
                return unlabeled_pool, self.trainer_machine.classes

            dataloader = get_subset_dataloaders(self.train_instance.train_dataset,
                                                unlabeled_pool,
                                                None,
                                                self.config.batch,
                                                workers=self.config.workers,
                                                shuffle=False)['train']
            pbar = tqdm(dataloader, ncols=80)
            # Score each examples in the unlabeled pool
            scores = torch.Tensor().to(self.config.device)
            with torch.no_grad():
                for batch, data in enumerate(pbar):
                    inputs, _ = data
                    
                    inputs = inputs.to(self.config.device)

                    outputs = self.model(inputs)
                    scores_batch_i = self.measure_func(outputs)
                    scores = torch.cat((scores,scores_batch_i))
            _, rankings = torch.sort(scores, descending=False)
            rankings = list(rankings[:self.config.budget])

            t_train = set() # New labeled samples
            t_classes = set() # New classes (may include seen classes)
            for idx in rankings:
                new_sample = unlabeled_pool[idx]
                t_train.add(new_sample)
                t_classes.add(self.train_instance.train_labels[new_sample])
            return t_train, t_classes
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    import pdb; pdb.set_trace()  # breakpoint a576792d //

