import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidLayer(torch.nn.Module):
    def __init__(self, output_size, feature_size):
        super(SigmoidLayer, self).__init__()

    def forward(self, x):
        return x
