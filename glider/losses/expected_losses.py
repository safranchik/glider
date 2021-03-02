import torch
import torch.nn.functional as F


def expected_cross_entropy(predictions, y_bar, reduction="mean"):

    # handles the case when y_bar is a vector of hard labels
    if len(y_bar.shape) <= 1:
        return F.cross_entropy(predictions, y_bar, reduction=reduction)

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    assert y_bar.shape[1] == predictions.shape[1]

    loss = torch.sum(-F.log_softmax(predictions, dim=1) * y_bar.flatten(), dim=1)

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss
