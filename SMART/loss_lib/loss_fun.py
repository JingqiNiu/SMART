
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_channel(pred_channels):
    task_num = len(pred_channels)
    pred_channels = np.cumsum([0] + pred_channels)
    slice_list = []
    for idx in range(task_num):
        ss = slice(pred_channels[idx] ,pred_channels[idx+1])
        slice_list.append ( ss)
    return slice_list

def get_smooth_label(num_class, smooth=0):
    if num_class == 1:
        return [smooth, 1-smooth]
    table = np.eye(num_class, dtype=float)
    for i in range(num_class):
        table[i, i] -= smooth
        if i == 0:
            table[i, i+1] = smooth

        elif i == num_class-1:
            table[i, i-1] = smooth
        else:
            table[i, i-1] = smooth / 2.0
            table[i, i+1] = smooth / 2.0
    table = torch.tensor(table)
    return table


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
            self.alpha = self.alpha / self.alpha.sum()
            self.alpha = self.alpha.view(-1, 1)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        predargmax = torch.argmax(preds_softmax, dim = 1)
        # torch.set_printoptions(precision=1)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1, 1))
        loss = -torch.mul(torch.pow((1-preds_softmax),
                                    self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BCEFocalLoss(torch.nn.Module):
    """
    """

    def __init__(self, gamma=2, alpha=0.5, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        if not alpha:
            alpha = 0.5
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
            (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class SoftCrossEntropy(nn.Module):
    """
    inputs shape [N,num_class]
    target shape [N,num_class]
    alpha [num_class]
    """

    def __init__(self, alpha=None, smooth=0, reduction='mean'):
        super(SoftCrossEntropy, self).__init__()
        self.alpha = alpha
        if isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = torch.FloatTensor(alpha)
            self.alpha = self.alpha / self.alpha.sum()
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, target):
        table = get_smooth_label(
            inputs.shape[1], self.smooth).to(target.device)
        target = table[target]
        logit = F.softmax(inputs, dim=1)
        print('the labels is',target.T)
        predargmax = torch.argmax(logit, dim = 1)
        print('the predargmax is',predargmax)
        print('the softmax is', (logit.T))
        log_likelihood = -(logit + 1e-10).log()
        if not self.alpha is None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            log_likelihood = log_likelihood * self.alpha
        loss = torch.mul(log_likelihood, target)

        if self.reduction == 'mean':
            loss = torch.sum(loss) / inputs.shape[0]
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
