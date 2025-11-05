
from .transforms_define import default_train, default_val,resize_crop_train
from .transforms_define import your_own_train


DEFINE_TRANS = dict()
DEFINE_TRANS['default_train'] = default_train
DEFINE_TRANS['default_val'] = default_val
DEFINE_TRANS['resize_crop_train'] = resize_crop_train
DEFINE_TRANS['resize_crop_val'] = default_val
# DEFINE_TRANS['your_own_train'] = your_own_train
# DEFINE_TRANS['your_own_val']   = default_val


def build_transforms(mode, cfg):

    target_size = cfg['target_size']
    prob = cfg.get('aug_prob', 0.5)
    aug_m = cfg.get('aug_m', 2)
    transforms_name = cfg.get('transform_define', 'default')

    transforms_train_fun = DEFINE_TRANS[transforms_name + '_train']
    transforms_val_fun = DEFINE_TRANS[transforms_name + '_val']
    if mode == 'train':
        return transforms_train_fun(target_size, prob, aug_m)
    else:
        return transforms_val_fun(target_size)
