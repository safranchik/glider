import torch
import torch.nn.functional as F


def expected_cross_entropy(predictions, Y_bar, reduction='mean'):

    # handles the case when Y_bar is a vector of hard labels
    if len(Y_bar.shape) == 1:
        return F.cross_entropy(predictions, Y_bar)

    if reduction not in {'none', 'mean', 'sum'}:
        raise ValueError('{} is not a valid value for reduction'.format(predictions))

    predictions = torch.sum(-F.log_softmax(predictions, dim=1) * Y_bar, dim=1)

    if reduction == 'mean':
        predictions = torch.mean(predictions)

    elif reduction == 'sum':
        predictions = torch.sum(predictions)

    return predictions
