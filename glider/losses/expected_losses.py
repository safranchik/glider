import torch
import torch.nn.functional as F


def expected_cross_entropy(input, target, reduction="mean"):
    # handles the case when y_bar is a vector of hard labels

    if len(target.shape) <= 1:
        return F.cross_entropy(input, target, reduction=reduction)

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    assert target.shape[1] == input.shape[1]

    loss = torch.sum(-target * F.log_softmax(input, dim=1), dim=1)

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


def expected_nll_loss(input, target, reduction="mean"):
    # handles the case when y_bar is a vector of hard labels
    if len(target.shape) <= 1:
        return F.nll_loss(input, target, reduction=reduction)

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    assert target.shape[1] == input.shape[1]

    loss = torch.sum(-target * input, dim=1)

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss
