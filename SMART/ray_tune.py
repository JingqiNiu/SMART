
import copy
import os
from ray import tune
# from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune import CLIReporter
import ray
from train import train_model
from analysis_trails import analysis_exp, rm_ck_in_pbt
from ray_lib import init_search_space,select_search_alg,select_scheduler

def tune_model(tune_hparams, module_hparams):
    """
    tune_hparams: dict, params for tune
    module_hparams: dict ,params for lightning_module and trainer
    """
    # ray.init(address='ray://192.168.100.12:9999')
    # ray.init(address='auto', _redis_password='5241590000000000')
    module_hparams['tune_hp'] = copy.deepcopy(tune_hparams)
    trainer_hp = module_hparams['trainer']

    search_hp, hp_mutations_for_pbt, init_start_params = init_search_space(
        tune_hparams['space_define'], module_hparams)

    hp_pbt = tune_hparams['hp_pbt']
    hp_pbt['hp_mutations'] = hp_mutations_for_pbt
    hp_hbs = {
        'max_epochs': trainer_hp['max_epochs'],
        'grace_period': trainer_hp['min_epochs']
    }
    hp_scheduler = [hp_pbt, hp_hbs]

    scheduler, ray_callback = select_scheduler(tune_hparams['scheduler'], *hp_scheduler)
    search_alg = select_search_alg(tune_hparams['search_alg'], init_start_params)

    metric_columns = ["training_iteration", "final_score",
                      "val_loss", "train_loss", ]
    reporter = CLIReporter(parameter_columns=list(search_hp.keys()),
                           metric_columns=metric_columns)

    run_hp = tune_hparams['run_hp']
    train_model_with_parameters = tune.with_parameters(train_model,
                                                       module_hparams=module_hparams, ray_callback=ray_callback)
    # if trainer_hp['accelerator'] == 'ddp':  # ddp + ray tune  is not available yet
    #     train_model_with_parameters = DistributedTrainableCreator(train_model_with_parameters,
    #                                                               num_workers=1,
    #                                                               backend='nccl',
    #                                                               num_cpus_per_worker=run_hp['resources_per_trial']['cpu'],
    #                                                               num_gpus_per_worker=run_hp['resources_per_trial']['gpu'])
    #     run_hp['resources_per_trial'] = None

    run_hp['run_or_experiment'] = train_model_with_parameters
    run_hp['scheduler'] = scheduler
    run_hp['search_alg'] = search_alg
    run_hp['progress_reporter'] = reporter
    run_hp['config'] = search_hp
    run_hp['local_dir'] = module_hparams['root_folder']
    run_hp['name'] = module_hparams['sub_folder']

    analysis = tune.run(**run_hp)
    # after each trial,remove temporary checkpoint in pb2 or pbt
    if tune_hparams['scheduler'] == 'pb2' or tune_hparams['scheduler'] == 'pbt':
        keep_checkpoints_num = tune_hparams['run_hp']['keep_checkpoints_num']
        trail_dirs = analysis.dataframe()['logdir'].to_list()
        rm_ck_in_pbt(trail_dirs, keep_checkpoints_num)

    save_csv_dir = os.path.join(module_hparams['root_folder'], module_hparams['sub_folder'])
    analysis_exp(analysis, metric_columns, save_csv_dir)
