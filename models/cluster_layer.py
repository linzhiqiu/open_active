import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterLayer(torch.nn.Module):
    def __init__(self, cluster_num, feature_size, distance_func, gamma=1.0, init_mode='zeros'):
        super(ClusterLayer, self).__init__()
        assert init_mode == 'zeros'
        self.distance_func = distance_func
        self.gamma = gamma
        self.clusters = torch.nn.Parameter(torch.zeros(cluster_num, feature_size))

    def forward(self, x):
        x = x.unsqueeze(1).expand(x.size(0), self.clusters.size()[0], x.size(1))
        distances = - self.distance_func(x, self.clusters.unsqueeze(0)) # A size (#batch, #clusters) tensor
        # distances = torch.exp(self.gamma*distances)
        distances = self.gamma*distances
        # output = distances / distances.sum(1, keepdim=True)
        return distances
