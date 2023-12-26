import torch
import numpy as np


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def to_one_hot(inp, num_classes):
    if inp is not None:
        y_onehot = torch.FloatTensor(inp.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
        return torch.autograd.Variable(y_onehot.cuda(), requires_grad=False)
    else:
        return None


def mixup_process(out, target, lam):
    indices = np.random.permutation(out.size(0))
    out = out * lam + out[indices] * (1 - lam)
    target = target * lam + target[indices] * (1 - lam)
    return out, target
