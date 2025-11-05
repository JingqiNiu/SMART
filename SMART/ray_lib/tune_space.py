from ray import tune
from .ray_utils import init_value_from_config


def default_space():
    search_apace = {}
    # search space for lightning module
    search_apace['transform#aug_prob'] = tune.uniform(0.2, 0.8)
    search_apace['transform#aug_m'] = tune.uniform(1, 3)
    search_apace['data#sample_distribution'] = tune.uniform(0.5, 1.0)
    search_apace['lr'] = tune.loguniform(1e-5, 1e-3)
    search_apace['optimizer#optim_name'] = tune.choice([ 'adam', 'adamw'])
    search_apace['loss#loss_name'] = tune.choice(['ce_loss', 'focal_loss'])
    search_apace['data#batch_size'] = tune.choice([4, 8, 16])
    search_apace['model#model_name'] = tune.choice(['efficientnet-b1','efficientnet-b2','efficientnet-b3'])

    hp_mutations_for_pbt = {
        'transform#aug_prob': tune.uniform(0.2, 0.8),
        'transform#aug_m': tune.uniform(1, 3),
        "lr": tune.loguniform(1e-5, 1e-2)
    }
    return search_apace, hp_mutations_for_pbt


SPACE_LIB = dict()
SPACE_LIB['default'] = default_space


def init_search_space(space_define, module_hparams):
    '''
    description:  create search_space and 'points_to_evaluate' for search_alg
    param :
        space_define(str): the key of SPACE_LIB
        module_hparams: dict, from module_config.yaml
    return :
        search_apace: dict, the parameter space to search
        init_start_params:list, 'points_to_evaluate' in search_alg

    '''
    search_apace, hp_mutations_for_pbt = SPACE_LIB[space_define]()

    # use 'module_config' to  make 'init_start_params',
    init_start_params = [{}]
    # 'init_start_params' has the same value as module_hparams
    init_start_params[0] = init_value_from_config(module_hparams, search_apace)

    return search_apace, hp_mutations_for_pbt, init_start_params
