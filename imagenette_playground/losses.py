import torch
from torch.nn import functional as F


class Loss:

    def losses(self, logits, targets):
        raise NotImplementedError

    def __call__(self, logits, targets, reduction='mean'):
        losses = self.losses(logits, targets)
        if reduction == 'mean':
            return losses.mean()
        if reduction == 'sum':
            return losses.sum()
        if reduction == 'none':
            return losses
        raise ValueError(f'{reduction} is not a valid reduction mode. Should be "mean", "sum" or "none".')


class CrossEntropy(Loss):

    def losses(self, logits, targets):
        return F.cross_entropy(logits, targets, reduction='none')
