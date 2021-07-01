import numpy as np
import torch
import torch.nn.functional as F

def harden(vec):

    # vector is torch tensor
    if torch.torch.is_tensor(vec):
        if vec.dim() == 2:
            vec = torch.argmax(vec, dim=1)

    else:
        # vector is a numpy array or an iterable type other than a tensor
        vec = np.array(vec)
        if vec.ndim == 2:
            vec = np.argmax(vec, axis=1)

    return vec


def soften(vec, num_classes):

    # vector is torch tensor
    if torch.is_tensor(vec):
        if vec.dim() == 1:
            vec = F.one_hot(vec.to(torch.int64), num_classes=num_classes)
    else:
        # vector is numpy array or an iterable type other than a tensor
        vec = np.array(vec)
        if vec.ndim == 1:
            vec = np.squeeze(np.eye(num_classes)[vec.reshape(-1)])

    return vec



