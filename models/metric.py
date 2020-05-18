import torch 
from torch.nn import functional as F

def squared_l2_distance(A, B):
    norm_A = torch.sum(A ** 2, dim=1).view(-1, 1)
    norm_B = torch.sum(B ** 2, dim=1).view(1, -1)

    AB = torch.matmul(A, B.t())
    square_dist = torch.clamp(norm_A + norm_B - 2*AB, min=0)
    return square_dist

def cosine_similarity(A, B, normalized=True):
    # whether the A and B are normalized unit vectors
    if normalized:
        A_normalized = A
        B_normalized = B
    else:
        A_normalized = F.normalize(A, p=2, dim=1, eps=1e-12)
        B_normalized = F.normalize(B, p=2, dim=1, eps=1e-12)
    return torch.matmul(A_normalized, B_normalized.t())


def predict(centroids, query, metric='squared_l2'):
    if metric == 'squared_l2':
        similarity = -squared_l2_distance(query, centroids)
    elif metric == 'cosine_with_normalized_features':
        similarity = cosine_similarity(query, centroids, normalized=True)
    elif metric == 'cosine_with_unnormalized_features':
        similarity = cosine_similarity(query, centroids, normalized=False)
    else:
        raise ValueError('Invalid metric specified')
    return similarity
