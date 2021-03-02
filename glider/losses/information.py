import torch
from glider.losses import entropy


def entropy_weighted(loss_class, predictions, y=None, reduction="mean", eps=1e-7):

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    if not len(predictions):
        return torch.tensor([0], requires_grad=True, dtype=torch.float32)

    if y is None:
        loss = torch.log(loss_class(predictions, reduction='none') + eps)
    else:
        # import pdb; pdb.set_trace()
        loss = (2.3-entropy(predictions, reduction="none")) * loss_class(predictions, y, reduction='none')

        # loss = (1-rho) * torch.sqrt(loss) + rho * loss

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


def information(loss_class, predictions, y=None, rho=0, reduction="mean", eps=1e-7, mode="log"):

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    if not len(predictions):
        return torch.tensor([0], requires_grad=True, dtype=torch.float32)

    if y is None:
        loss = torch.log(loss_class(predictions, reduction='none') + eps)
    else:
        loss = loss_class(predictions, y, reduction='none')
        if mode == "log":
            loss = torch.log(loss + eps)
        elif mode=="sqrt":
            loss = torch.sqrt(loss)
        # import pdb; pdb.set_trace()
        # loss = torch.log(loss + eps - rho) + loss
        # import pdb; pdb.set_trace()
        # loss = (1-rho) * torch.sqrt(loss) + rho * loss

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


def plastic_loss(loss_class, predictions, y=None, reduction="mean", rho=0.5, eps=1e-7):

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    if not len(predictions):
        return torch.tensor([0], requires_grad=True, dtype=torch.float32)

    if y is None:
        loss = torch.log(loss_class(predictions, reduction='none') + eps)
    else:
        loss = loss_class(predictions, y, reduction='none')
        loss = rho *  2.3 *torch.log(loss + eps) + (1-rho) * loss

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss

def sqrt_loss(loss_class, predictions, y=None, reduction="mean", eps=1e-7):

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    if not len(predictions):
        return torch.tensor([0], requires_grad=True, dtype=torch.float32)

    if y is None:
        loss = torch.log(loss_class(predictions, reduction='none') + eps)
    else:
        loss = torch.sqrt(loss_class(predictions, y, reduction='none'))

    if reduction == "mean":
        loss = torch.mean(loss)

    elif reduction == "sum":
        loss = torch.sum(loss)

    return loss


