import torch.nn.functional as F
import torch
import numpy as np


def weight_loss(weights, loss, logits, normalize=True, *args, **kwargs):

    if not np.isclose(torch.sum(weights).item(), 1) and normalize:
        weights = F.normalize(weights, p=1, dim=0)

    return torch.sum(weights * loss(logits, reduction="none", *args, **kwargs))

