import os
import numpy as np
from copy import deepcopy

import torch
from pytorch_lightning.core.lightning import LightningModule

from loss_lib import Select_Loss
from data_lib import train_loader, basic_loader
from data_lib.parse_data_infor import get_data_infor, data_str_list_to_dict
from metrics_lib.save_exp_results import Post_Process
from models_lib import build_model, load_pretrain_model
from optim_lib import build_optim, select_scheduler
from config_helper import check_set_parameter
from cam import attention_map


def distributed_concat(input_obj, world_size):
    if torch.is_tensor(input_obj):
        output_tensors = [torch.zeros_like(input_obj) for _ in range(world_size)]
        torch.distributed.all_gather(output_tensors, input_obj)
        concat = torch.cat(output_tensors, dim=0)
    else:
        output_obj = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(output_obj, input_obj)
        concat = []
        for i in output_obj:
            concat.extend(i)
    return concat


class PL_Module(LightningModule):
    def __init__(self, config):
        """
        config: dict  
        """
        super(PL_Module, self).__init__()
        config_new = deepcopy(config)
        if 'test_csv_path' in config_new.keys():
            self.test_csv_path = config_new['test_csv_path']
        else:
            self.test_csv_path = None
        if 'EXtest_csv_path' in config_new.keys():
            self.EXtest_csv_path = config_new['EXtest_csv_path']
        else:
            self.EXtest_csv_path = None
        config = check_set_parameter(config)
        self.save_hyperparameters(config)  # pass config to self.haparams
        config = deepcopy(config)
        self.num_classes = config['model']['num_classes']
        self.transforms_cfg = config['transform']
        self.data_cfg = config['data']
        self.add_info = config['add_info']
        if 'resume' in config:
            self.model = build_model(**config['model'], add_info = config['add_info'], config = config)
        else:
            self.model = build_model(**config['model'], add_info = config['add_info'], config = config)
        self.loss_term = Select_Loss(config['loss'], self.num_classes)
        self.dataset_define = config.get('dataset_define' ,None)
        self.test_out_dir = None  # init

    def setup(self, stage) -> None:
        self.use_ddp = self.trainer.use_ddp
        self.world_size = self.trainer.world_size

        self.bs, res_bs = divmod(self.hparams['data']['batch_size'], self.world_size)
        self.data_cfg['batch_size'] = self.bs
        assert res_bs == 0, "In DDP, batch size must be an integer multiple of the number of GPUs"
        if self.trainer.auto_lr_find:
            # self._hparams_initial will save to .yaml
            self._hparams_initial['lr'] = self.hparams['lr']
        if self.test_out_dir is None:
                self.test_out_dir = os.path.join( self.trainer.default_root_dir, 'test_result/')
        self.post_class = Post_Process(self.hparams['post_process'], self.global_rank,self.test_out_dir )
        return super().setup(stage=stage)


    def configure_optimizers(self):
        lr = self.hparams["lr"]

        optim_cfg = dict(lr=lr)
        optim_cfg.update(self.hparams["optimizer"])
        optimizer = build_optim(self.parameters(), optim_cfg)
        scheduler = select_scheduler(optimizer, self.hparams)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'final_score'}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, x,x2):
        # print('the x is',torch.unique(x))
        # print('the patient info', torch.unique(x2))
        if self.add_info == True:
            out = self.model(x,x2)
        else:
            out = self.model(x)
        # print('!!!!! the model is',self.model)
        return out

    def share_step(self, batch):
        x, y = batch['image'], batch['label']
        result = self.forward(x,batch['img_infor'])
        return {"label": y, "result": result}

    def training_step(self, batch, batch_idx):
        # print('the batch shape is',batch['image'].shape)
        path = batch['path']
        epoch = self.current_epoch
        # save_path = f'/home/jingqi/Retina_Classification_Train_UWF/epoch_training_data/epoch_{epoch}.txt'
        # with open(save_path, 'a') as f:
        #     for p in path:
        #         f.write(f'{p}\n')
        return self.share_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch)

    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        path = batch['path']
        result = self.forward(x,batch['img_infor'])
        return {"label": y, "result": result, "path": path}

    def training_step_end(self, outputs):
        # print('the outputs shape is',outputs['result'].shape)
        # print('the outputs shape is',outputs['result'])
        # print('the label is',outputs['label'].shape)
        # print('the outputs is',outputs['label'])
        loss = self.loss_term.call(outputs['result'], outputs['label'])
        self.log("train_loss", loss, on_step=True, prog_bar=True, sync_dist=self.use_ddp)
        return {"loss": loss}

    def validation_step_end(self, outputs):
        return outputs

    def test_step_end(self, outputs, *args, **kwargs):
        return outputs

    def epoch_end_fun(self, outputs):
        label = torch.cat([x["label"] for x in outputs])
        result = torch.cat([x["result"] for x in outputs])
        if self.use_ddp:
            label = distributed_concat(label, self.world_size)
            result = distributed_concat(result, self.world_size)
        val_loss = self.loss_term.call(result, label)
        paths = []
        if 'path' in outputs[0]:
            for item in outputs:
                paths.extend(item['path'])
            if isinstance(paths[0], tuple):
                paths = [item[0] for item in paths]
            if self.use_ddp:
                paths = distributed_concat(paths, self.world_size)
        final_score = self.post_class.on_epoch( label ,result,paths)
        return val_loss ,final_score


    def validation_epoch_end(self, outputs):
        self.print(f'##validation. current_epoch: {self.current_epoch} ##')

        val_loss ,final_score = self.epoch_end_fun(outputs)
        logs = {"val_loss": val_loss,  "final_score": final_score}
        self.log_dict(logs, prog_bar=True, sync_dist=self.use_ddp)
        return None

    def test_epoch_end(self, outputs):
        print('############ this is the testing #############################')
        test_loss ,final_score = self.epoch_end_fun(outputs)
        logs = {"test_loss": test_loss, "test_final_score": final_score}
        self.log_dict(logs, prog_bar=True, sync_dist=self.use_ddp)
        return None
        
    def creat_attention_map(self, data_loader=None):
        if (self.hparams['cam_name'] is None):
            self.print('cam_name ==None', 'not use cam')
            return
        elif self.trainer.use_dp and (self.global_rank > 0):
            return

        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            self.model = load_pretrain_model(self.model, best_model_path)
        else:
            self.print('No best checkpoint in training, using original model')

        cam_dataloader = data_loader if data_loader else self.get_basic_dataloader(self.hparams['test_data_name'], self.hparams['label_name'])
        attention_map(self.model, cam_dataloader, self.device, self.test_out_dir,
                      cam_name=self.hparams['cam_name'], conv_layer=self.hparams['cam_layer'])


    def train_dataloader(self):
        train_csv_infor = get_data_infor(self.hparams['train_data_name'], self.hparams['label_name'] , self.dataset_define)
        self.data_cfg['sample_seed'] = self.current_epoch
        return train_loader(train_csv_infor,  self.transforms_cfg, self.data_cfg, self.trainer.world_size, self.global_rank)

    def get_basic_dataloader(self, data_name, label_name, csv_path = None):
        csv_infor = get_data_infor(data_name, label_name ,self.dataset_define, csv_path  = csv_path)
        # print('((((((((()))))))))', csv_infor, data_name)
        return basic_loader(csv_infor, self.transforms_cfg, self.data_cfg, num_replicas=self.trainer.world_size, rank=self.global_rank)

    def val_dataloader(self):
        return self.get_basic_dataloader(self.hparams['val_data_name'], self.hparams['label_name'])

    def test_dataloader(self):
        if 'test_csv_path' in self.hparams.keys():
            test_csv_path = self.hparams['test_csv_path']
        else:
            test_csv_path = None
        return self.get_basic_dataloader(self.hparams['test_data_name'], self.hparams['label_name'], csv_path = test_csv_path)

    def evaluate_train_dataloader(self):
        """
        This data_loader is about the training set, and no data enhancement is used
        """
        return self.get_basic_dataloader(self.hparams['train_data_name'], self.hparams['label_name'])

    def extra_dataloader(self):
        """
        Run test on several extra data_loader

        Returns:
            data_loader_dict: dict ,Contains several data_loader for test
        """

        extra_data_name = data_str_list_to_dict(self.hparams['extra_data_name'])
        extra_label_name = self.hparams['extra_label_name']

        if isinstance(extra_label_name, str):
            extra_label_name = [extra_label_name]
        if len(extra_label_name) == 1:
            extra_label_name = [extra_label_name * len(extra_data_name)]
        assert len(extra_data_name) == len(extra_label_name)
        if 'EXtest_csv_path' in self.hparams.keys():
            if self.hparams['EXtest_csv_path'] != None:
                EXtest_csv_path = self.hparams['EXtest_csv_path']
                EXtest_csv_path = True
            else:
                test_csv_path = None
                EXtest_csv_path = False
        else:
            test_csv_path = None
            EXtest_csv_path = False
            
        data_loader_dict = {}
        for index, key in enumerate(extra_data_name):
            label_name = extra_label_name[index]
            data_infor_dict = {key: extra_data_name[key]}
            if EXtest_csv_path == True:
                data_loader_dict[key] = self.get_basic_dataloader(data_infor_dict, label_name, csv_path = EXtest_csv_path)
            else:
                data_loader_dict[key] = self.get_basic_dataloader(data_infor_dict, label_name)
        return data_loader_dict
