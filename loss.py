import torch.nn as nn


def consistency_loss_mse(view1, view2):
    return nn.MSELoss()(view1, view2)


def consistency_loss_cosine(view1, view2):
    cos = nn.CosineSimilarity(dim=1)
    return (1 - cos(view1, view2)).mean()
