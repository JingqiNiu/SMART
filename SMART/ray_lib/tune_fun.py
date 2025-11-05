
import copy
import os
from ray.tune.schedulers import PopulationBasedTraining, AsyncHyperBandScheduler, FIFOScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback


def get_pb2_bounds(hp_mutations_for_pbt):
    """
    description:  get the bounds in hp_mutations_for_pbt
    param :
       hp_mutations_for_pbt:dict
    return :
        bounds:dict, {key:[lower , upper] ,}
    """
    pbt_dict = hp_mutations_for_pbt
    bounds = {}
    for key in pbt_dict:
        bounds[key] = [pbt_dict[key].lower, pbt_dict[key].upper]
    return bounds


def select_scheduler(name, *args):
    '''
    description: select scheduler for tune
    param :
        name : str,['pbt','pb2','hbs','fifo' ]
        *args: list, hp_pbt ,hp_hbs
    return :
        scheduler , ray_callback
    '''
    hp_pbt, hp_hbs = args[0], args[1]
    report_call_back_metrics = {
        "val_loss": "val_loss",
        "final_score": "final_score",
        "train_loss": "train_loss_step",
    }

    if name == 'pbt':
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=hp_pbt['interval'],
            resample_probability=hp_pbt['prob'],
            hyperparam_mutations=hp_pbt['hp_mutations'])
    elif name == 'pb2':
        hyperparam_bounds = get_pb2_bounds(hp_pbt['hp_mutations'])
        scheduler = PB2(
            time_attr="training_iteration",
            perturbation_interval=hp_pbt['interval'],
            hyperparam_bounds=hyperparam_bounds)  # pb2 do have 'resample_probability'
    elif name == 'hbs':
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=hp_hbs['max_epochs'],
            grace_period=hp_hbs['grace_period'])
    elif name == 'fifo':
        scheduler = FIFOScheduler()
    else:
        print('write wrong name:', name)
        raise NotImplementedError

    if name == 'pbt' or name == 'pb2':
        ray_callback = TuneReportCheckpointCallback(
            report_call_back_metrics, filename="checkpoint", on="validation_end")
    else:
        ray_callback = TuneReportCallback(report_call_back_metrics, on="validation_end")

    return scheduler, ray_callback


def select_search_alg(name, init_start_params):
    """
    name: str,['hyper','none']
    init_start_params: list, Initialize the optimal search space from module_conf
    """
    name = name.lower()
    if name == 'hyper':
        search_alg = HyperOptSearch(points_to_evaluate=init_start_params)
    elif name == 'none':
        search_alg = None
    else:
        print('write wrong name:', name)
        raise NotImplementedError
    return search_alg
