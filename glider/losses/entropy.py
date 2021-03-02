import torch.nn.functional as F
import torch


def entropy(logits, eps=1e-7, reduction='mean'):
    h = -torch.sum(F.softmax(logits, dim=1) * torch.log_softmax(torch.add(logits, eps), dim=1), dim=1)

    if reduction not in {'none', 'mean', 'sum'}:
        raise ValueError('{} is not a valid value for reduction'.format(reduction))

    if reduction == 'mean':
        h = torch.mean(h)

    elif reduction == 'sum':
        h = torch.sum(h)

    return h

