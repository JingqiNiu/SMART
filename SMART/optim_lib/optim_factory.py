from torch_optimizer import RAdam
import torch
import inspect
import copy
OPTIM_LIB = dict()

for m in inspect.getmembers(torch.optim, inspect.isclass):
    OPTIM_LIB[m[0].lower()] = m[1]  # add optimizer in torch.optim

OPTIM_LIB['radam'] = RAdam

""" 
['radam','asgd', 'adadelta', 'adagrad', 'adam', 'adamw', 'adamax', 'lbfgs', 'optimizer', 'rmsprop', 'rprop', 'sgd', 'sparseadam']
"""
def build_optim(parameters,cfg):
    cfg = copy.deepcopy(cfg)
    cfg['params'] = parameters
    
    name = cfg.pop('optim_name')
    cfg.setdefault('weight_decay',0.0001)
    if (name == 'sgd') :
        cfg.setdefault('momentum',0.9)
    return OPTIM_LIB[name](**cfg)
