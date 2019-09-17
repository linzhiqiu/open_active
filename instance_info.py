import torch
import torch.nn.functional as F
from tqdm import tqdm

class InstanceInfo(object):
    def __init__(self):
        super(InstanceInfo, self).__init__()

class BasicInstanceInfo(InstanceInfo):
    """ Store the most basic information of a new instance to be added
    """
    def __init__(self, round_index, true_label, predicted_label, softmax, entropy, seen):
        super(BasicInstanceInfo, self).__init__()
        self.round_index = round_index
        self.true_label = true_label # in original index
        self.predicted_label = predicted_label # in original index
        self.softmax = softmax # Caveat: For SigmoidInfoCollector, this is the max sigmoid score.
        self.entropy = entropy # In fact, negative entropy, so the higher(more pos) the more certain
        self.seen = seen # -1 if unseen, 1 if seen

class ClusterInstanceInfo(InstanceInfo):
    """ Store the most basic information of a new instance to be added
    """
    def __init__(self, round_index, true_label, predicted_label, softmax, entropy, gaussian, seen):
        super(ClusterInstanceInfo, self).__init__()
        self.round_index = round_index
        self.true_label = true_label # in original index
        self.predicted_label = predicted_label # in original index
        self.softmax = softmax # Caveat: For ClusterInfoCollector, this is just the prob score
        self.entropy = entropy # In fact, negative entropy, so the higher(more pos) the more certain
        self.gaussian = gaussian
        self.seen = seen # -1 if unseen, 1 if seen


class InfoCollector(object):
    ''' A class that collect list of InstanceInfo (optionally a tensor of scores that can be used to sort the instances)
    '''
    def __init__(self):
        super(InfoCollector, self).__init__()

    def gather_instance_info(self, dataloader, model):
        raise NotImplementedError()

class BasicInfoCollector(InfoCollector):
    def __init__(self, round_index, unmapping_dict, seen_classes, measure_func=lambda x:torch.Tensor([0])):
        '''Args:
            measure_func : Input --> a batch of outputs, Output --> A tensor of float scores
        '''
        super(BasicInfoCollector, self).__init__()
        self.round_index = round_index
        self.seen_classes = seen_classes
        self.unmapping_dict = unmapping_dict
        self.measure_func = measure_func

    def _basic_batch_instances_info_func(self, outputs, labels):
        '''A func takes a batch of outputs, labels,
           then output a list of InstanceInfo objects
        '''
        # Return a list with length == outputs.size(0). Each element is the information of that specific example.
        # Each element is represented by (round_index, true_label, predicted_label, softmax, entropy, -1 if in unseen class else 1)
        batch_instances_info = []
        softmax_outputs = F.softmax(outputs, dim=1)
        prob_scores, predicted_labels = torch.max(softmax_outputs, 1)
        for i in range(outputs.size(0)):
            softmax_i = softmax_outputs[i]
            entropy_i = float((softmax_i*softmax_i.log()).sum()) # This is the negative entropy
            prob_score_i = float(prob_scores[i])
            predicted_label_i = int(self.unmapping_dict[int(predicted_labels[i])])
            label_i = int(labels[i])
            instance_info = BasicInstanceInfo(self.round_index, label_i, predicted_label_i, prob_score_i, entropy_i, label_i in self.seen_classes)
            batch_instances_info.append(instance_info)
        return batch_instances_info

    def gather_instance_info(self, dataloader, model, device='cuda'):
        ''' Return the result of running model on dataloader
            Args:
                dataloader : Cannot use shuffle=True or sampler. Assume same order as the dataset.
                batch_instances_info_func : A func takes a batch of outputs, labels,
                                            then output a list of InstanceInfo objects
            Returns:
                scores : A 1-D tensor representing the scores of all instances
                info : A list of InstanceInfo objects
        '''
        pbar = tqdm(dataloader, ncols=80)
        # Score each examples in the unlabeled pool
        scores = torch.Tensor().to(device)
        info = []
        with torch.no_grad():
            for batch, data in enumerate(pbar):
                inputs, labels = data
                
                inputs = inputs.to(device)

                outputs = model(inputs)

                scores_batch_i = self.measure_func(outputs).to(device)
                scores = torch.cat((scores,scores_batch_i))

                # batch_instances_info is a list of information for every example in this batch.
                batch_instances_info = self._basic_batch_instances_info_func(outputs, labels)
                info += batch_instances_info
        return scores, info

import math
class ClusterInfoCollector(BasicInfoCollector):
    def __init__(self, gamma, *args, **kwargs):
        self.gamma = gamma
        super(ClusterInfoCollector, self).__init__(*args, **kwargs)

    def _basic_batch_instances_info_func(self, outputs, labels):
        '''A func takes a batch of outputs, labels,
           then output a list of InstanceInfo objects
        '''
        # Return a list with length == outputs.size(0). Each element is the information of that specific example.
        # Each element is represented by (round_index, true_label, predicted_label, max_cluster_score, entropy, -1 if in unseen class else 1)
        batch_instances_info = []
        # softmax_outputs = F.softmax(outputs, dim=1)
        normalized_outputs = F.softmax(outputs, dim=1)
        # normalized_outputs = outputs / outputs.sum(1, keepdim=True)
        gaussians = (outputs.exp() / ((math.pi / self.gamma)**.5)).mean(1)
        prob_scores, predicted_labels = torch.max(normalized_outputs, 1)
        for i in range(outputs.size(0)):
            gaussian_i = float(gaussians[i])
            prob_i = normalized_outputs[i]
            entropy_i = float((prob_i*prob_i.log()).sum()) # This is the negative entropy
            prob_score_i = float(prob_scores[i])
            predicted_label_i = int(self.unmapping_dict[int(predicted_labels[i])])
            label_i = int(labels[i])
            instance_info = ClusterInstanceInfo(self.round_index, label_i, predicted_label_i, prob_score_i, entropy_i, gaussian_i, label_i in self.seen_classes)
            batch_instances_info.append(instance_info)
        return batch_instances_info

class SigmoidInfoCollector(BasicInfoCollector):
    def __init__(self, *args, **kwargs):
        super(SigmoidInfoCollector, self).__init__(*args, **kwargs)

    def _basic_batch_instances_info_func(self, outputs, labels):
        '''A func takes a batch of outputs, labels,
           then output a list of InstanceInfo objects
        '''
        # Return a list with length == outputs.size(0). Each element is the information of that specific example.
        # Each element is represented by (round_index, true_label, predicted_label, max_cluster_score, entropy, -1 if in unseen class else 1)
        batch_instances_info = []
        softmax_outputs = F.softmax(outputs, dim=1)
        sigmoid_outputs = torch.nn.Sigmoid()(outputs)
        # normalized_outputs = outputs / outputs.sum(1, keepdim=True)
        prob_scores, predicted_labels = torch.max(softmax_outputs, 1)
        max_sigmoid_scores, _ = torch.max(sigmoid_outputs, 1)
        for i in range(outputs.size(0)):
            prob_i = prob_scores[i]
            open_i = max_sigmoid_scores[i]
            entropy_i = float((prob_i*prob_i.log()).sum()) # This is the negative entropy
            prob_score_i = float(prob_scores[i])
            predicted_label_i = int(self.unmapping_dict[int(predicted_labels[i])])
            label_i = int(labels[i])
            instance_info = BasicInstanceInfo(self.round_index, label_i, predicted_label_i, open_i, entropy_i, label_i in self.seen_classes)
            batch_instances_info.append(instance_info)
        return batch_instances_info

class C2AEInfoCollector(BasicInfoCollector):
    def __init__(self, *args, **kwargs):
        super(C2AEInfoCollector, self).__init__(*args, **kwargs)

    def _basic_batch_instances_info_func(self, outputs, labels):
        '''A func takes a batch of outputs, labels,
           then output a list of InstanceInfo objects
        '''
        # Return a list with length == outputs.size(0). Each element is the information of that specific example.
        # Each element is represented by (round_index, true_label, predicted_label, max_cluster_score, entropy, -1 if in unseen class else 1)
        raise NotImplementedError()
        batch_instances_info = []
        softmax_outputs = F.softmax(outputs, dim=1)
        sigmoid_outputs = torch.nn.Sigmoid()(outputs)
        # normalized_outputs = outputs / outputs.sum(1, keepdim=True)
        prob_scores, predicted_labels = torch.max(softmax_outputs, 1)
        max_sigmoid_scores, _ = torch.max(sigmoid_outputs, 1)
        for i in range(outputs.size(0)):
            prob_i = prob_scores[i]
            open_i = max_sigmoid_scores[i]
            entropy_i = float((prob_i*prob_i.log()).sum()) # This is the negative entropy
            prob_score_i = float(prob_scores[i])
            predicted_label_i = int(self.unmapping_dict[int(predicted_labels[i])])
            label_i = int(labels[i])
            instance_info = BasicInstanceInfo(self.round_index, label_i, predicted_label_i, open_i, entropy_i, label_i in self.seen_classes)
            batch_instances_info.append(instance_info)
        return batch_instances_info