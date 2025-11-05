import inspect
import torch
import torch.nn as nn
from copy import deepcopy
from .loss_fun import FocalLoss, BCEFocalLoss, SoftCrossEntropy,calc_channel
from .ordinal_regression import OrdinalRegressionLoss
LOSS_LIB = dict()

for m in inspect.getmembers(torch.nn, inspect.isclass):
    if m[0].endswith('Loss'):
        LOSS_LIB [m[0] ] = m[1]  # add all loss in torch.nn
""" 
['AdaptiveLogSoftmaxWithLoss', 'BCELoss', 'BCEWithLogitsLoss', 'CTCLoss', 'CosineEmbeddingLoss', 'CrossEntropyLoss', 'HingeEmbeddingLoss', 'KLDivLoss',
 'L1Loss', 'MSELoss', 'MarginRankingLoss', 'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss', 'MultiMarginLoss', 'NLLLoss', 'PoissonNLLLoss', 'SmoothL1Loss', 
'SoftMarginLoss', 'TripletMarginLoss', 'TripletMarginWithDistanceLoss']
"""

# Manually register
LOSS_LIB['bce_focal_loss'] = BCEFocalLoss
LOSS_LIB['OrdinalRegressionLoss'] = OrdinalRegressionLoss

def build_from_cfg(class_entrance, cfg):
    """
    Class instantiation

    Args:
        class_entrance ( class ): not string ,the actual class that can be called 
        cfg (dict): The parameters of the  '__init__' in  class

    Returns:
        Instances of the class
    """
    fun_param = deepcopy(cfg)

    sig = inspect.signature(class_entrance)
    sig_keys = sig.parameters.keys()

    if 'kwargs' not in sig_keys:
        for key in list (fun_param.keys()) :
            if key not in sig_keys:
                print(f'{key} not in {sig_keys}')
                fun_param.pop(key)  # Remove redundant parameters

    return class_entrance(**fun_param)


def build_loss(name, cfg):
    """
    Args:
        name (str): the name of the loss 
        cfg (dict): The parameters of the  '__init__' in  class

    """
    class_entrance = LOSS_LIB[name]
    return build_from_cfg(class_entrance, cfg)

class Multi_Task_Loss(nn.Module):
    def __init__(self,loss_cfg,channels,task_weight=None ):
        super().__init__()
        cfg_list = deepcopy(loss_cfg)
        self.slice_list =  calc_channel(channels)

        loss_fun_list = []
        for item in cfg_list:
            if isinstance(item,str):
                loss_fun = build_loss(item,{})
            elif isinstance(item,dict):
                name = item.pop('name')
                loss_fun = build_loss(name,item)
            else:
                raise TypeError
            loss_fun_list.append(loss_fun)
        self.loss_fun_list = loss_fun_list
        self.task_num = len(self.loss_fun_list)
        if task_weight is not None:
            self.task_weight = task_weight
        else:
            self.task_weight = [1] * self.task_num

    def forward(self, preds, labels):
        loss_list = []
        idx = 0
        for loss_fun,ch,weight in zip(self.loss_fun_list,self.slice_list,self.task_weight):
            pred = preds[:,ch].squeeze(dim=1)
            label = labels[:,idx]
            # change label dtype
            if isinstance(loss_fun,torch.nn.BCEWithLogitsLoss):
                label = label.float()
            elif isinstance(loss_fun,torch.nn.CrossEntropyLoss):
                label = label.long()
            one_task_loss = loss_fun(pred,label) * weight
            loss_list.append(one_task_loss )
            idx += 1
        loss = torch.stack(loss_list).mean()
        return loss

LOSS_LIB['multi_task_loss'] = Multi_Task_Loss

class Select_Loss(nn.Module):
    def __init__(self, cfg, num_classes):
        """
        Args:
            cfg (dict):  contain  loss parameters
            num_classes (int): the number of classes
        """
        super().__init__()
        cfg = deepcopy(cfg)
        assert num_classes >= 1
        loss_name = cfg.pop('loss_name')
        class_weight = cfg.pop('loss_weight', None)
        smooth = cfg.pop("smooth", 0)
        print(f'## Loss: {loss_name},config: {cfg}')

        if loss_name == 'focal_loss':
            if num_classes >= 2:
                criterion = FocalLoss(num_classes=num_classes, alpha=class_weight)
            else:
                criterion = BCEFocalLoss(alpha=class_weight)

        elif loss_name == 'ce_loss':
            if num_classes >= 2:
                if class_weight is not None or smooth > 0:
                    criterion = SoftCrossEntropy(class_weight, smooth)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.BCEWithLogitsLoss()
        
        else:
            criterion = build_loss(loss_name, cfg)

        self.criterion = criterion
        self.num_classes = num_classes
        self.loss_name = loss_name
    def forward(self,pre, label):
        return self.call(pre, label)
    def call(self, pre, label):
        if self.num_classes == 1:
            label = label.float()
            label = label.unsqueeze(axis=1) if label.dim() == 1 else label
        loss = self.criterion(pre, label)
        return loss
