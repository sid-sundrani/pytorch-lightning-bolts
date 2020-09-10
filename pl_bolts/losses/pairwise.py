import torch
import torch.nn.functional as F


def cosine_similarity_loss(self, a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    loss = (a * b).sum(-1).mean()
    return loss


def cosine_distance_loss(self, a, b):
