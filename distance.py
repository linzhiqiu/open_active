import torch

def eu_distance(a,b):
    return torch.sqrt(torch.sum((a - b) ** 2)) / 200.

def cos_distance(a,b):
    return 1 - torch.nn.CosineSimilarity(dim=0)(a, b)

def eu_distance_batch(a, b, factor=200.):
    return torch.sqrt(torch.sum((a-b) ** 2, dim=len(a.size())-1)) / factor

def cos_distance_batch(a, b):
    return 1. - torch.nn.CosineSimilarity(dim=len(a.size())-1)(a, b)

