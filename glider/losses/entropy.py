import torch.nn.functional as F
import torch


def entropy(logits, eps=1e-7, reduction='mean', probabilities=False):

    if len(logits.shape) == 1:
        if probabilities:
            return -torch.sum(logits * torch.log(logits))
        else:
            return -torch.sum(F.softmax(logits) * torch.log_softmax(logits, dim=-1))

    if probabilities:
        h = -torch.sum(logits * torch.log(torch.add(logits, eps)), dim=1)
    else:
        h = -torch.sum(F.softmax(logits, dim=1) * torch.log_softmax(torch.add(logits, eps), dim=1), dim=1)

    if reduction not in {'none', 'mean', 'sum'}:
        raise ValueError('{} is not a valid value for reduction'.format(reduction))

    if reduction == 'mean':
        h = torch.mean(h)

    elif reduction == 'sum':
        h = torch.sum(h)

    return h

