import torch

def eu_distance(a,b):
    return torch.sqrt(torch.sum((a - b) ** 2)) / 200.

def cos_distance(a,b):
    return 1 - torch.nn.CosineSimilarity(dim=0)(a, b)