import torch
import inspect
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

SCHE_LIB = dict()

for m in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
    SCHE_LIB[m[0]] = m[1]  # add all lr_scheduler in torch.optim.lr_scheduler
""" 
['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'Counter', 'CyclicLR', 'ExponentialLR', 'LambdaLR', 'MultiStepLR', 
'MultiplicativeLR', 'OneCycleLR', 'Optimizer', 'ReduceLROnPlateau', 'StepLR', '_LRScheduler']
"""

def build_scheduler(name, cfg):
    class_name = SCHE_LIB[name]
    return class_name(**cfg)

def select_scheduler(optimizer,hparams):

    scheduler_cfg = deepcopy(hparams['scheduler'])
    scheduler_name = scheduler_cfg.pop('scheduler_name')

    if scheduler_name == 'lr_plateau':
        lr_conf = scheduler_cfg['plateau_cfg']
        default_cfg = dict(optimizer=optimizer, mode='max', factor=0.5, patience=5,
                cooldown=1,min_lr=1e-7,  verbose=True)
        default_cfg.update(lr_conf)
        scheduler = ReduceLROnPlateau(**default_cfg)

    elif scheduler_name == 'lr_step':
        lr_conf = scheduler_cfg['scheduler_cfg']
        scheduler = StepLR(optimizer, step_size=lr_conf['step_size'], gamma=lr_conf['gamma'])
        
    elif scheduler_name == 'cosine':
        eta_min = scheduler_cfg['scheduler_cfg'].get('eta_min',1e-7)
        scheduler = CosineAnnealingLR( optimizer, hparams['trainer']['max_epochs'], eta_min=eta_min)
    else:
        scheduler = build_scheduler(scheduler_name, scheduler_cfg['scheduler_cfg'])
    return scheduler