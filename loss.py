import torch
import torch.nn as nn


def consistency_loss_mse(view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between the two views' logits."""
    return nn.MSELoss()(view1, view2)


def consistency_loss_cosine(view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity between the two views' logits, averaged over the batch."""
    cos = nn.CosineSimilarity(dim=1)
    return (1 - cos(view1, view2)).mean()
