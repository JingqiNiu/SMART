
import os
import sys
import time
import datetime
import numpy as np
import copy
import yaml

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load

import ray
from ray import tune
from lightning_module import PL_Module
from utils import Logger, copy_file_or_dir
from config_helper import config_from_search_space


def train_model(search_hp, checkpoint_dir=None, module_hparams=None, ray_callback=None):
    """
    search_hp:  
    checkpoint_dir: this is  for PopulationBasedTraining. we do not use it to load checkpoint.
    module_hparams: config for lightning_module
    ray_callback: tune ReportCallback
    """
    module_hparams = config_from_search_space(module_hparams, search_hp) # use search_hp to replace module_hparams
    trainer_hp = copy.deepcopy(module_hparams['trainer'])
    print('the module_hparams is',module_hparams)
    if ray_callback is not None:  # use ray tune
        print(f'!!This trail use  gpu: {ray.get_gpu_ids()} !!')
        time.sleep(np.random.randint(10))  # avoid read data at the same time
        saving_root = tune.get_trial_dir()
        csv_logger = CSVLogger(saving_root, name="", version="")

        logger_lists = [csv_logger]
        callbacks_lists = [ray_callback]
    else:  # not use ray tune
        print('Not using ray tune')
        saving_root = os.path.join(module_hparams['root_folder'], module_hparams['sub_folder'])
        os.makedirs(saving_root, exist_ok=True)
        time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        sys.stdout = Logger(os.path.join(saving_root, f'record_log_{time_str}.txt'))

        tb_logger = TensorBoardLogger(saving_root, name="", version="")
        csv_logger = CSVLogger(saving_root, name="", version="")

        logger_lists = [tb_logger, csv_logger]
        callbacks_lists = list()

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(saving_root, 'checkpoints'),
        filename='{epoch:04d}_{final_score:.3f}_{val_loss:.3f}_{train_loss:.3f}',
        save_top_k=module_hparams['save_top_k_checkpoint'], verbose=True, monitor='final_score',
        save_last=True, period=1, mode='max', save_weights_only=False)
    # some non-essential callbacks
    early_stop_callback = EarlyStopping(monitor="final_score", min_delta=0.0001,
                                        patience=module_hparams['earlystop_patience'], verbose=True, mode='max')
    gpu_stats_callback = GPUStatsMonitor(inter_step_time=False, intra_step_time=True)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch', log_momentum=False)
    callbacks_lists += [early_stop_callback, checkpoint_callback,lr_monitor_callback]  # add non-essential callbacks , here

    trainer_hp['logger'] = logger_lists
    trainer_hp['callbacks'] = callbacks_lists
    trainer_hp['default_root_dir'] = saving_root

    trainer = Trainer(**trainer_hp)

    original_yaml_path = os.path.join(saving_root, 'hparams_original.yaml')
    with open(original_yaml_path, 'w') as f_write:
        yaml.safe_dump(module_hparams, f_write, sort_keys=False)

    if module_hparams['save_code']:
        code_dir = os.getcwd()
        code_save_dir = os.path.join(saving_root, 'save_code')
        copy_file_or_dir(module_hparams['save_code'], code_dir, code_save_dir)

    if checkpoint_dir:  # load checkpoint for PopulationBasedTraining
        ckpt = pl_load(os.path.join(checkpoint_dir), map_location=lambda storage, loc: storage)
        model = PL_Module._load_model_state(ckpt,  module_hparams=module_hparams)
        trainer.current_epoch = ckpt["epoch"]
        print('@@@@@@@@@ We load ',checkpoint_dir)
    else:
        model = PL_Module(module_hparams)

    trainer.tune(model)  # for auto_lr_find  and auto_scale_batch_size
    trainer.fit(model)  # trianing process
    print('Best ck path : ', checkpoint_callback.best_model_path)


    print(f'############# Here We Evaluate the Inner TestSet, Best ckpt {checkpoint_callback.best_model_path} ######')
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    # model.creat_attention_map()
    print(f'############# Here We Evaluate the Validation Set, Best ckpt {checkpoint_callback.best_model_path} ###############')
    model.test_out_dir = os.path.join(saving_root, 'Validation_result/')
    trainer.test(test_dataloaders=model.val_dataloader(),
                    ckpt_path=checkpoint_callback.best_model_path)
    print(f'############# Here We Evaluate the eval_training_set Set, Best ckpt {checkpoint_callback.best_model_path} ###############')
    model.test_out_dir = os.path.join(saving_root, 'evaluate_train_result/')
    trainer.test(test_dataloaders=model.evaluate_train_dataloader(),
                    ckpt_path=checkpoint_callback.best_model_path)
        

    
    print(f'############## Here We Evaluate the extra_data Set, Best ckpt {checkpoint_callback.best_model_path} ###############')
    data_loader_dict = model.extra_dataloader()

    for name in data_loader_dict:
        print('############', name, '#####################')
        extra_data_loader = data_loader_dict[name]
        model.test_out_dir = os.path.join(saving_root, *['extra_test_results', name])
        trainer.test(test_dataloaders =extra_data_loader, ckpt_path=checkpoint_callback.best_model_path)
        model.creat_attention_map(extra_data_loader)   
