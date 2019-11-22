import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split
from trainer_machine import Network, train_epochs
import argparse
import numpy as np
from dataset_factory import get_dataset_factory
from utils import get_subset_dataloaders, get_subset_loader, get_loader, SetPrintMode, get_target_mapping_func, get_target_unmapping_dict
# from torchvision import models
from models.resnet import ResNet18


class OpenActiveManager:
    """
    An instance that represents the state of an open active task
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 total_num_classes=100,
                 init_num_samples=1000,
                 init_num_classes=40,
                 budget_per_iteration=20,
                 num_iterations=400,
                 num_hold_out_classes=10,
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.total_num_classes = total_num_classes
        self.num_hold_out_classes = num_hold_out_classes
        self.unseen_data = []  # Indices of the training data unseen
        self.seen_data = []  # Indices of the training data seen
        self.hold_out_data = []  # Indices of the training data unseen
        self.train_labels = []  # Labels of each of the training data
        self.holder_out_classes = []  # Hold out classes
        self.train_classes = []  # Labels that are not held out
        self._set_hold_out_samples(train_dataset)
        self._set_init_seen_sample(train_dataset,
                                   init_num_classes,
                                   int(init_num_samples/init_num_classes),
                                  )

        self.check_data(verbose=True)


    def _set_hold_out_samples(self, train_dataset):
        self.hold_out_classes = np.random.choice(list(range(self.total_num_classes)),
                                                 size=self.num_hold_out_classes,
                                                 replace=False)
        self.train_classes = [label for label in range(self.total_num_classes) if label not in self.hold_out_classes]
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
        self.train_labels = []  # Labels of all training data
        for i, (_, label) in enumerate(train_data):
            self.train_labels.append(int(label[0]))
            if int(label[0]) in self.hold_out_classes:
                self.hold_out_data.append(i)
            else:
                self.unseen_data.append(i)


    def _set_init_seen_sample(self,
                              train_dataset,
                              num_init_seen_classes,
                              num_init_samples_per_class,
                              ):
        label_to_data_indices = {label: [] for label in self.train_classes}
        for idx in self.unseen_data:
            label_to_data_indices[self.train_labels[idx]].append(idx)
        # Get the initial seen classes
        seen_classes = np.random.choice(self.train_classes,
                                        size=num_init_seen_classes,
                                        replace=False)
        for label in seen_classes:
            # Get samples from these classes
            seen_samples = np.random.choice(label_to_data_indices[label],
                                            size=num_init_samples_per_class,
                                            replace=False)
            self.seen_data += list(seen_samples)
        
        # Exclude these samples from unseen_data
        self.unseen_data = [idx for idx in self.unseen_data if idx not in self.seen_data]
    
    def update_new_seen_data(self, new_seen_data):
        self.seen_data += list(new_seen_data)
        self.unseen_data = [i for i in self.unseen_data if i not in new_seen_data]


    def get_seen_classes(self):
        seen_classes = set()
        for idx in self.seen_data:
            seen_classes.add(self.train_labels[idx])
        return sorted(list(seen_classes))

    def get_unseen_classes(self):
        seen_classes = self.get_seen_classes()
        unseen_classes = [i for i in range(self.total_num_classes) if i not in seen_classes and i not in self.hold_out_classes]
        return sorted(unseen_classes)

    def get_seen_samples(self):
        return sorted(self.seen_data)

    def get_unseen_samples(self):
        return sorted(self.unseen_data)
    
    def get_seen_samples_dataloader(self, batch_size=32):
        sampler = SubsetRandomSampler(self.seen_data)
        train_data = torch.utils.data.DataLoader(self.train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 sampler=sampler)
        return train_data

    def check_data(self, verbose=False):
        assert(len(self.train_labels) == len(self.train_dataset))
        assert(len(self.unseen_data)+len(self.seen_data)+len(self.hold_out_data) == len(self.train_labels))
        assert(len(self.get_seen_classes())+len(self.get_unseen_classes())+self.num_hold_out_classes == self.total_num_classes)

        seen_data_set = set(self.seen_data)
        unseen_data_set = set(self.unseen_data)
        hold_out_data_set = set(self.hold_out_data)
        assert(seen_data_set.isdisjoint(unseen_data_set))
        assert(seen_data_set.isdisjoint(hold_out_data_set))
        assert(unseen_data_set.isdisjoint(hold_out_data_set))

        if verbose:
            print(f"{len(self.seen_data)} seen data, {len(self.unseen_data)} unseen data, {len(self.hold_out_data)} hold out data.")
            print(f"{len(self.get_seen_classes())} seen classes, {len(self.get_unseen_classes())} unseen classes, {self.num_hold_out_classes} hold out classes.")
            print("Passing data check!")


def evaluation(eval_dataset, model, criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    eval_data = torch.utils.data.DataLoader(eval_dataset,
                                            batch_size=32,
                                            sampler=None)
    with torch.no_grad():
        for batch_data, batch_label in eval_data:
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            output = model(batch_data)
            pred = F.log_softmax(output, dim=1)
            eval_loss += F.cross_entropy(pred, batch_label, reduction='sum').item()
            pred = pred.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(batch_label.view_as(pred)).sum().item()
    eval_loss /= len(eval_dataset)
    accuracy = correct / len(eval_dataset)

    print(f"Eval accuracy={accuracy}, loss={eval_loss}")


def main():
    QUERY_BUDGET = 20
    NUM_ITERATIONS = 400
    dataset_factory = get_dataset_factory("CIFAR100", "./data", "default")
    train_dataset, test_dataset = dataset_factory.get_dataset()
    open_active = OpenActiveManager(train_dataset, test_dataset)
    model = ResNet18(num_classes=100).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    # Initial training
    for epoch in range(50):
        for batch_data, batch_label in open_active.get_seen_samples_dataloader(batch_size=32):
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            optimizer.zero_grad()

            output = model(batch_data)
            pred = F.log_softmax(output, dim=1)
            loss = criterion(pred, batch_label)
            loss.backward()
            optimizer.step()
        print(f"Initial training, epoch={epoch}, loss={loss}")
    evaluation(test_dataset, model, criterion)

    # Each round of active learning
    for i in range(NUM_ITERATIONS):
        unseen_samples = open_active.get_unseen_samples()
        # Randomly sampling unseen samples
        new_samples = np.random.choice(unseen_samples, QUERY_BUDGET, replace=False)
        open_active.update_new_seen_data(new_samples)
        open_active.check_data()
        model.train()
        for epoch in range(20):
            for batch_data, batch_label in open_active.get_seen_samples_dataloader(batch_size=32):
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
                optimizer.zero_grad()

                output = model(batch_data)
                pred = F.log_softmax(output, dim=1)
                loss = criterion(pred, batch_label)
                loss.backward()
                optimizer.step()
            print(f"Round={i}, epoch={epoch}, loss={loss}")
        evaluation(test_dataset, model, criterion)
            
    # print(train_dataset, test_dataset)

if __name__ == '__main__':
    main()
